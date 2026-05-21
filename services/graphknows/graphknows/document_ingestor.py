"""Document ingestion: URL/path → Chunk → Entity dual-write to Memgraph + Neo4j."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ONTOLOGY = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "CONCEPT",
    "POLICY",
    "LAW",
    "INSTITUTION",
    "TECHNOLOGY",
    "PRODUCT",
]

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50

_SYNTHETIC_TEXT = "The European Central Bank sets monetary policy. " * 100


@dataclass
class ChunkData:
    id: str
    text: str
    char_offset: int
    token_count: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class EntityMention:
    name: str
    entity_type: str
    chunk_id: str
    score: float = 1.0


@dataclass
class IngestResult:
    document_id: str
    title: str
    url: str
    chunk_count: int
    entity_count: int
    mention_count: int
    errors: list[str] = field(default_factory=list)


class DocumentIngestor:
    """Ingest documents from URL or local path into Neo4j + Memgraph.

    Pipeline: fetch -> chunk -> extract entities -> embed -> dual-write.

    Attributes:
        _neo4j: Synchronous Neo4j driver instance.
        _mg: Optional synchronous Memgraph driver (same Neo4j protocol).
        _embed: Optional embedding model (e.g. SentenceTransformer).
    """

    def __init__(
        self,
        neo4j_driver: Any | None = None,
        memgraph_driver: Any | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self._neo4j = neo4j_driver
        self._mg = memgraph_driver
        self._embed = embedding_model
        self._gliner_cache: dict[tuple[str, ...], Any] = {}
        self._spacy_cache: Any = None

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch(self, source: str) -> tuple[str, str]:
        """Fetch raw text from *source* (URL, local path, or synthetic marker).

        Args:
            source: URL (http/https), local file path, ``.pdf`` path,
                    or ``__synthetic__`` for testing.

        Returns:
            ``(raw_text, resolved_source)`` where *resolved_source* is the
            actual URL or absolute filesystem path.

        Raises:
            RuntimeError: If the source cannot be fetched or decoded.
        """
        if source == "__synthetic__":
            return _SYNTHETIC_TEXT, "synthetic://test"

        if source.startswith(("http://", "https://")):
            return self._fetch_url(source)

        if source.lower().endswith(".pdf"):
            return self._fetch_pdf(source)

        # Plain text file
        path = os.path.abspath(source)
        try:
            with open(path, encoding="utf-8") as f:
                return f.read(), path
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {path}") from None
        except UnicodeDecodeError:
            raise RuntimeError(f"File is not valid UTF-8: {path}") from None

    def _fetch_url(self, url: str) -> tuple[str, str]:
        import httpx

        try:
            resp = httpx.get(
                url,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Graphknows/1.0 (https://graphknows.io; research@graphknows.io)"
                    ),
                },
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"HTTP fetch failed for {url}: {exc}") from exc

        content_type = resp.headers.get("content-type", "").lower()
        if "application/pdf" in content_type:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(resp.content)
            tmp.close()
            text, _ = self._fetch_pdf(tmp.name)
            os.unlink(tmp.name)
            return text, str(resp.url)

        return resp.text, str(resp.url)

    def _fetch_pdf(self, path: str) -> tuple[str, str]:
        abs_path = os.path.abspath(path)
        try:
            import fitz  # pymupdf

            doc = fitz.open(abs_path)
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages), abs_path
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pymupdf failed for %s: %s — trying pdfminer", abs_path, exc)

        # pdfminer fallback
        try:
            from pdfminer.high_level import extract_text  # type: ignore[import-untyped]

            text = extract_text(abs_path)
            return text, abs_path
        except ImportError:
            raise RuntimeError(
                "No PDF parser available — install pymupdf or pdfminer.six"
            ) from None

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _find_sent_starts(text: str) -> list[int]:
        """Return sorted character positions where sentences begin."""
        starts = [0]
        for m in re.finditer(r"[.!?](?:\s|$)", text):
            pos = m.end()
            starts.append(pos)
        return starts

    def _chunk(self, text: str) -> list[ChunkData]:
        """Split *text* into overlapping chunks of ~*CHUNK_TOKENS* tokens.

        Snaps boundaries to the nearest sentence start/end to avoid
        mid-sentence splits.  Minimum chunk size: 50 tokens.
        """
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if not tokens:
            return []

        sent_starts = self._find_sent_starts(text)
        text_len = len(text)
        stride = CHUNK_TOKENS - OVERLAP_TOKENS  # 450

        chunks: list[ChunkData] = []
        total = len(tokens)

        start_tok = 0
        while start_tok < total:
            end_tok = min(start_tok + CHUNK_TOKENS, total)

            if end_tok - start_tok < 50:
                break

            # Convert token offsets to character offsets
            start_char = len(enc.decode(tokens[:start_tok]))
            end_char = len(enc.decode(tokens[:end_tok]))

            # Snap to sentence boundaries
            snapped_start = self._snap_sent_start(start_char, sent_starts)
            snapped_end = self._snap_sent_end(end_char, sent_starts, text_len)

            if snapped_end <= snapped_start:
                start_tok += stride
                continue

            chunk_text = text[snapped_start:snapped_end]
            chunk_tokens = enc.encode(chunk_text)
            if len(chunk_tokens) < 50:
                start_tok += stride
                continue

            chunk_id = hashlib.sha256(f"{snapped_start}:{chunk_text[:200]}".encode()).hexdigest()[
                :16
            ]

            chunks.append(
                ChunkData(
                    id=chunk_id,
                    text=chunk_text,
                    char_offset=snapped_start,
                    token_count=len(chunk_tokens),
                )
            )

            start_tok += stride

        return chunks

    @staticmethod
    def _snap_sent_start(pos: int, sent_starts: list[int]) -> int:
        """Snap *pos* backward to the nearest sentence start."""
        for i in range(len(sent_starts) - 1, -1, -1):
            if sent_starts[i] <= pos:
                return sent_starts[i]
        return sent_starts[0]

    @staticmethod
    def _snap_sent_end(pos: int, sent_starts: list[int], text_len: int) -> int:
        """Snap *pos* forward to the nearest sentence end."""
        for i in range(1, len(sent_starts)):
            if sent_starts[i] >= pos:
                return sent_starts[i]
        return text_len

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _get_gliner(self, ontology: list[str]) -> Any:
        key = tuple(sorted(ontology))
        if key not in self._gliner_cache:
            from gliner import GLiNER  # type: ignore[import-untyped]

            self._gliner_cache[key] = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        return self._gliner_cache[key]

    def _get_spacy(self) -> Any:
        if self._spacy_cache is None:
            import spacy  # type: ignore[import-untyped]

            self._spacy_cache = spacy.load("en_core_web_sm")
        return self._spacy_cache

    def _extract_entities(
        self,
        chunk: ChunkData,
        ontology: list[str],
    ) -> list[EntityMention]:
        """Extract named entities from *chunk* using GLiNER2 or spaCy fallback.

        Tries GLiNER2 first (threshold 0.5).  Falls back to spaCy NER if
        GLiNER2 is unavailable.  Returns an empty list if no NER backend
        is installed.
        """
        try:
            model = self._get_gliner(ontology)
            preds = model.predict_entities(chunk.text, ontology, threshold=0.5)
            return [EntityMention(p["text"], p["label"], chunk.id, p["score"]) for p in preds]
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("GLiNER2 inference failed: %s — falling back to spaCy", exc)

        try:
            nlp = self._get_spacy()
            doc = nlp(chunk.text)
            return [
                EntityMention(ent.text, ent.label_, chunk.id, 1.0)
                for ent in doc.ents
                if not ontology or ent.label_ in ontology
            ]
        except ImportError:
            logger.warning("No NER backend (GLiNER2 or spaCy). Returning empty entity list.")
            return []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks: list[ChunkData]) -> list[ChunkData]:
        """Attach dense embeddings to each chunk in batches of 32."""
        if self._embed is None:
            return chunks

        texts = [c.text for c in chunks]
        batch_size = 32
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed.encode(batch)
            all_embeddings.extend(embeddings.tolist())

        for chunk, emb in zip(chunks, all_embeddings):
            chunk.embedding = emb

        return chunks

    # ------------------------------------------------------------------
    # Neo4j write
    # ------------------------------------------------------------------

    def _write_neo4j(
        self,
        doc_id: str,
        title: str,
        url: str,
        source_type: str,
        session_id: str,
        chunks: list[ChunkData],
        mentions: list[EntityMention],
    ) -> None:
        if self._neo4j is None:
            logger.warning("Neo4j driver is None — skipping Neo4j write")
            return

        with self._neo4j.session() as session:
            session.execute_write(
                self._neo4j_write_tx,
                doc_id,
                title,
                url,
                source_type,
                session_id,
                chunks,
                mentions,
            )

    @staticmethod
    def _neo4j_write_tx(
        tx: Any,
        doc_id: str,
        title: str,
        url: str,
        source_type: str,
        session_id: str,
        chunks: list[ChunkData],
        mentions: list[EntityMention],
    ) -> None:
        tx.run(
            "MERGE (d:Document {id: $doc_id}) "
            "SET d.title = $title, d.url = $url, "
            "    d.source_type = $source_type, "
            "    d.ingested_at = datetime(), "
            "    d.session_id = $session_id",
            doc_id=doc_id,
            title=title,
            url=url,
            source_type=source_type,
            session_id=session_id,
        )

        for c in chunks:
            tx.run(
                "MERGE (c:Chunk {id: $chunk_id}) "
                "SET c.text = $text, c.char_offset = $char_offset, "
                "    c.token_count = $token_count, "
                "    c.embedding = $embedding "
                "WITH c "
                "MATCH (d:Document {id: $doc_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                chunk_id=c.id,
                text=c.text,
                char_offset=c.char_offset,
                token_count=c.token_count,
                embedding=c.embedding or None,
                doc_id=doc_id,
            )

        for m in mentions:
            tx.run(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.type = $entity_type "
                "WITH e "
                "MATCH (c:Chunk {id: $chunk_id}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                name=m.name,
                entity_type=m.entity_type,
                chunk_id=m.chunk_id,
            )

    # ------------------------------------------------------------------
    # Memgraph write
    # ------------------------------------------------------------------

    def _write_memgraph(
        self,
        doc_id: str,
        title: str,
        url: str,
        source_type: str,
        session_id: str,
        chunks: list[ChunkData],
        mentions: list[EntityMention],
    ) -> None:
        if self._mg is None:
            return

        try:
            with self._mg.session() as session:
                session.execute_write(
                    self._mg_write_tx,
                    doc_id,
                    title,
                    url,
                    source_type,
                    session_id,
                    chunks,
                    mentions,
                )
        except Exception as exc:
            logger.warning("Memgraph write failed (non-fatal): %s", exc)

    @staticmethod
    def _mg_write_tx(
        tx: Any,
        doc_id: str,
        title: str,
        url: str,
        source_type: str,
        session_id: str,
        chunks: list[ChunkData],
        mentions: list[EntityMention],
    ) -> None:
        """Same Cypher as Neo4j — Memgraph uses Neo4j-compatible dialect."""
        tx.run(
            "MERGE (d:Document {id: $doc_id}) "
            "SET d.title = $title, d.url = $url, "
            "    d.source_type = $source_type, "
            "    d.ingested_at = datetime(), "
            "    d.session_id = $session_id",
            doc_id=doc_id,
            title=title,
            url=url,
            source_type=source_type,
            session_id=session_id,
        )

        for c in chunks:
            tx.run(
                "MERGE (c:Chunk {id: $chunk_id}) "
                "SET c.text = $text, c.char_offset = $char_offset, "
                "    c.token_count = $token_count, "
                "    c.embedding = $embedding "
                "WITH c "
                "MATCH (d:Document {id: $doc_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                chunk_id=c.id,
                text=c.text,
                char_offset=c.char_offset,
                token_count=c.token_count,
                embedding=c.embedding or None,
                doc_id=doc_id,
            )

        for m in mentions:
            tx.run(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.type = $entity_type "
                "WITH e "
                "MATCH (c:Chunk {id: $chunk_id}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                name=m.name,
                entity_type=m.entity_type,
                chunk_id=m.chunk_id,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_source_type(source: str) -> str:
        if source.startswith(("http://", "https://")):
            return "url"
        if source.lower().endswith(".pdf"):
            return "pdf"
        if source == "__synthetic__":
            return "synthetic"
        return "file"

    def ingest(
        self,
        source: str,
        session_id: str,
        ontology: list[str] | None = None,
        title: str | None = None,
    ) -> IngestResult:
        """Ingest a document: fetch -> chunk -> NER -> embed -> dual-write.

        Args:
            source: URL, local file path, or ``__synthetic__`` (test marker).
            session_id: Ephemeral session identifier for Memgraph scoping.
            ontology: List of entity type labels for NER.  Defaults to
                      ``DEFAULT_ONTOLOGY``.
            title: Optional document title.  Falls back to *source*.

        Returns:
            ``IngestResult`` summarising the ingestion.
        """
        errors: list[str] = []
        ontology = ontology or DEFAULT_ONTOLOGY

        # 1. Fetch
        try:
            raw_text, resolved_url = self._fetch(source)
        except Exception as exc:
            logger.exception("Fetch failed for %s", source)
            return IngestResult(
                document_id="",
                title=title or source,
                url=source,
                chunk_count=0,
                entity_count=0,
                mention_count=0,
                errors=[f"Fetch failed: {exc}"],
            )

        if not raw_text.strip():
            return IngestResult(
                document_id="",
                title=title or source,
                url=resolved_url,
                chunk_count=0,
                entity_count=0,
                mention_count=0,
                errors=["Empty document"],
            )

        # 2. Chunk
        chunks = self._chunk(raw_text)
        if not chunks:
            doc_id = hashlib.sha256(resolved_url.encode()).hexdigest()[:16]
            return IngestResult(
                document_id=doc_id,
                title=title or source,
                url=resolved_url,
                chunk_count=0,
                entity_count=0,
                mention_count=0,
                errors=["No chunks produced"],
            )

        # 3. Extract entities
        all_mentions: list[EntityMention] = []
        for chunk in chunks:
            mentions = self._extract_entities(chunk, ontology)
            all_mentions.extend(mentions)

        # 4. Embed chunks (optional)
        if self._embed is not None:
            chunks = self._embed_chunks(chunks)

        # 5. Document ID
        doc_id = hashlib.sha256(resolved_url.encode()).hexdigest()[:16]
        doc_title = title or resolved_url
        source_type = self._infer_source_type(source)

        # 6. Neo4j write
        try:
            self._write_neo4j(
                doc_id,
                doc_title,
                resolved_url,
                source_type,
                session_id,
                chunks,
                all_mentions,
            )
        except Exception as exc:
            errors.append(f"Neo4j write failed: {exc}")
            logger.exception("Neo4j write failed")

        # 7. Memgraph write (optional)
        if self._mg is not None:
            try:
                self._write_memgraph(
                    doc_id,
                    doc_title,
                    resolved_url,
                    source_type,
                    session_id,
                    chunks,
                    all_mentions,
                )
            except Exception as exc:
                errors.append(f"Memgraph write failed (non-fatal): {exc}")
                logger.warning("Memgraph write failed (non-fatal): %s", exc)

        entity_names = {m.name for m in all_mentions}

        return IngestResult(
            document_id=doc_id,
            title=doc_title,
            url=resolved_url,
            chunk_count=len(chunks),
            entity_count=len(entity_names),
            mention_count=len(all_mentions),
            errors=errors,
        )
