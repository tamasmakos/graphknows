"""
Graphknows Archive Engine — Prototype
======================================
Demonstrates the full loop for a corpus-first retrieval system:

  1. INGEST  — load a small EUR-Lex slice from HuggingFace (lex_glue/eurlex)
  2. EMBED   — sentence-transformers (all-MiniLM-L6-v2, fast/light)
  3. STORE   — dual-write:
               • Memgraph: CorpusDocument nodes + CorpusEntity nodes
                 (labelled with :Corpus prefix so they never pollute session KGs)
               • In-memory vector store (numpy, no pgvector dependency for prototype)
  4. RETRIEVE — given a free-text query:
               • BM25 keyword scan over stored texts (rank_bm25)
               • Dense vector ANN (cosine, numpy)
               • RRF fusion of both ranked lists
               • Returns top-K with provenance (source, celex, url, score)
  5. DEMO    — run a handful of test queries and print results

Run:
    cd /root/graphknows
    python3 -B prototype_archive_engine.py

No services need to be running except Memgraph (bolt://172.18.0.1:17687).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

MEMGRAPH_URI  = "bolt://172.18.0.1:17687"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # 22 MB, runs on CPU in <1s per batch
N_DOCS        = 200                   # docs to ingest (keep small for prototype)
TOP_K         = 5                     # results per query
RRF_K         = 60                    # RRF constant (standard = 60)

DEMO_QUERIES = [
    "EU migration policy asylum seekers",
    "climate change emissions trading carbon",
    "digital single market data protection GDPR",
    "agricultural subsidies common agricultural policy",
    "financial regulation banking supervision",
]

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CorpusDoc:
    doc_id:   str
    title:    str
    text:     str
    labels:   list[str]   # EuroVoc concept labels
    source:   str = "eurlex_huggingface"
    url:      str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class RetrievalResult:
    doc_id:  str
    title:   str
    excerpt: str
    labels:  list[str]
    source:  str
    rrf_score: float
    bm25_rank: int | None
    vec_rank:  int | None

# ── 1. INGEST ─────────────────────────────────────────────────────────────────

def load_eurlex_slice(n: int) -> list[CorpusDoc]:
    """Load n docs from the lex_glue/eurlex HuggingFace dataset."""
    print(f"\n[1/4] Loading {n} EUR-Lex docs from HuggingFace …")
    from datasets import load_dataset
    # validation split loads faster than train (smaller)
    ds = load_dataset("lex_glue", "eurlex", split="validation", trust_remote_code=True)

    # EuroVoc label names
    label_names: list[str] = ds.features["labels"].feature.names

    docs: list[CorpusDoc] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        text: str = row["text"]
        title_match = re.match(r"^(.{0,120})", text.strip())
        title = title_match.group(1).strip() if title_match else f"EUR-Lex doc {i}"
        labels = [label_names[j] for j in row["labels"]]
        docs.append(CorpusDoc(
            doc_id = f"eurlex-hf-val-{i:05d}",
            title  = title,
            text   = text[:1000],   # truncate to 1000 chars for prototype
            labels = labels,
            source = "eurlex_huggingface",
        ))

    print(f"    Loaded {len(docs)} docs. Sample labels: {docs[0].labels[:3]}")
    return docs

# ── 2. EMBED ──────────────────────────────────────────────────────────────────

def embed_docs(docs: list[CorpusDoc], model: SentenceTransformer) -> list[CorpusDoc]:
    """Embed all docs in one batched call."""
    print(f"\n[2/4] Embedding {len(docs)} docs with {EMBED_MODEL} …")
    texts = [f"{d.title} {d.text}" for d in docs]
    t0 = time.time()
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    print(f"    Done in {time.time()-t0:.1f}s  shape={embs.shape}")
    for doc, emb in zip(docs, embs):
        doc.embedding = emb
    return docs

# ── 3. STORE ──────────────────────────────────────────────────────────────────

def store_in_memgraph(docs: list[CorpusDoc], driver: Any) -> None:
    """
    Write CorpusDocument nodes + CorpusLabel nodes to Memgraph.
    Uses :Corpus label prefix so these nodes are visually distinct from session KGs.
    MERGE ensures idempotency — re-running the prototype is safe.
    """
    print(f"\n[3/4] Writing to Memgraph (corpus namespace) …")
    t0 = time.time()

    with driver.session() as s:
        # Batch upsert documents
        s.run("""
            UNWIND $docs AS d
            MERGE (n:CorpusDocument {doc_id: d.doc_id})
            SET n.title  = d.title,
                n.text   = d.text,
                n.source = d.source,
                n.labels = d.labels
        """, docs=[
            {"doc_id": d.doc_id, "title": d.title,
             "text": d.text[:500], "source": d.source, "labels": d.labels}
            for d in docs
        ])

        # Upsert EuroVoc label nodes + relationships
        # Collect all unique labels
        label_doc_pairs = []
        for d in docs:
            for lbl in d.labels:
                label_doc_pairs.append({"doc_id": d.doc_id, "label": lbl})

        if label_doc_pairs:
            s.run("""
                UNWIND $pairs AS p
                MERGE (l:CorpusLabel {name: p.label})
                WITH l, p
                MATCH (d:CorpusDocument {doc_id: p.doc_id})
                MERGE (d)-[:TAGGED_WITH]->(l)
            """, pairs=label_doc_pairs)

    elapsed = time.time() - t0
    print(f"    Written {len(docs)} docs + label graph in {elapsed:.1f}s")

    # Verify
    with driver.session() as s:
        cnt = s.run("MATCH (n:CorpusDocument) RETURN count(n) as c").single()["c"]
        lbl = s.run("MATCH (n:CorpusLabel) RETURN count(n) as c").single()["c"]
        print(f"    Memgraph: {cnt} CorpusDocument nodes, {lbl} CorpusLabel nodes")

# ── 4. RETRIEVE ───────────────────────────────────────────────────────────────

class ArchiveRetriever:
    """
    Hybrid retriever: BM25 keyword + dense vector, fused with RRF.
    In-memory for prototype — production would use pgvector + OpenSearch.
    """

    def __init__(self, docs: list[CorpusDoc], model: SentenceTransformer, driver: Any):
        self.docs   = docs
        self.model  = model
        self.driver = driver
        self._build_bm25()
        self._matrix = np.stack([d.embedding for d in docs])  # (N, D)

    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "rank-bm25", "-q"])
            from rank_bm25 import BM25Okapi

        tokenize = lambda t: re.findall(r"\w+", t.lower())
        corpus   = [tokenize(f"{d.title} {d.text}") for d in self.docs]
        self.bm25      = BM25Okapi(corpus)
        self.tokenize  = tokenize

    def _kg_boost(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        """
        Query Memgraph: docs that share CorpusLabel nodes with query keywords
        get a small graph-connectivity boost.
        """
        keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
        if not keywords or not doc_ids:
            return {}
        pattern = "|".join(re.escape(k) for k in keywords)
        with self.driver.session() as s:
            rows = s.run("""
                MATCH (d:CorpusDocument)-[:TAGGED_WITH]->(l:CorpusLabel)
                WHERE d.doc_id IN $ids
                  AND toLower(l.name) =~ $pattern
                RETURN d.doc_id AS doc_id, count(l) AS matches
            """, ids=doc_ids, pattern=f".*({pattern}).*").data()
        return {r["doc_id"]: float(r["matches"]) for r in rows}

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievalResult]:
        q_tokens = self.tokenize(query)
        q_emb    = self.model.encode([query], normalize_embeddings=True)[0]

        # BM25 ranking
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked = np.argsort(bm25_scores)[::-1].tolist()  # high → low

        # Dense vector ranking (cosine, embeddings already normalized)
        cos_scores  = self._matrix @ q_emb
        vec_ranked  = np.argsort(cos_scores)[::-1].tolist()

        # Build rank lookup (1-indexed)
        bm25_rank = {idx: r+1 for r, idx in enumerate(bm25_ranked)}
        vec_rank  = {idx: r+1 for r, idx in enumerate(vec_ranked)}

        # RRF fusion
        all_idx = set(range(len(self.docs)))
        rrf: dict[int, float] = {}
        for idx in all_idx:
            rrf[idx] = (1/(RRF_K + bm25_rank.get(idx, len(self.docs))) +
                        1/(RRF_K + vec_rank.get(idx, len(self.docs))))

        # KG label boost (graph-connected docs score higher)
        top_ids = [self.docs[i].doc_id for i in sorted(rrf, key=lambda x: -rrf[x])[:k*3]]
        boost   = self._kg_boost(query, top_ids)
        for idx, doc in enumerate(self.docs):
            if doc.doc_id in boost:
                rrf[idx] += boost[doc.doc_id] * 0.001  # small, tiebreaker only

        # Final top-k
        top_k = sorted(rrf.items(), key=lambda x: -x[1])[:k]

        return [
            RetrievalResult(
                doc_id    = self.docs[i].doc_id,
                title     = self.docs[i].title,
                excerpt   = self.docs[i].text[:200],
                labels    = self.docs[i].labels[:4],
                source    = self.docs[i].source,
                rrf_score = score,
                bm25_rank = bm25_rank.get(i),
                vec_rank  = vec_rank.get(i),
            )
            for i, score in top_k
        ]

# ── 5. DEMO ───────────────────────────────────────────────────────────────────

def run_demo(retriever: ArchiveRetriever) -> None:
    print("\n" + "="*70)
    print("ARCHIVE ENGINE PROTOTYPE — DEMO QUERIES")
    print("="*70)

    for query in DEMO_QUERIES:
        print(f"\n🔍  Query: \"{query}\"")
        print("-" * 60)
        results = retriever.retrieve(query, k=TOP_K)
        for rank, r in enumerate(results, 1):
            print(f"  [{rank}] rrf={r.rrf_score:.5f}  bm25_rank={r.bm25_rank}  vec_rank={r.vec_rank}")
            print(f"       {r.title[:80]}")
            print(f"       labels: {r.labels}")
            print(f"       excerpt: {r.excerpt[:120].strip()} …")

    print("\n" + "="*70)
    print("PROTOTYPE COMPLETE")
    print("="*70)

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_start = time.time()

    # Load model once
    print(f"Loading embedding model: {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)

    # Memgraph connection
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=None)

    # Pipeline
    docs      = load_eurlex_slice(N_DOCS)
    docs      = embed_docs(docs, model)
    store_in_memgraph(docs, driver)

    # Build retriever
    print(f"\n[4/4] Building retriever …")
    retriever = ArchiveRetriever(docs, model, driver)
    print(f"    BM25 index built over {len(docs)} docs")
    print(f"    Vector matrix: {retriever._matrix.shape}")

    # Run demo
    run_demo(retriever)

    driver.close()
    print(f"\nTotal wall time: {time.time()-total_start:.1f}s")
