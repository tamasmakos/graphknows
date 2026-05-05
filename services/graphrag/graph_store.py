"""
graph_store.py — unified read/write layer for Neo4j + pgvector.

Responsibilities:
  - Neo4j: Chunk, Entity, Topic nodes; MENTIONS, LKIF_REL, IN_COMMUNITY edges
  - pgvector: chunk embeddings + entity embeddings (both 384-dim BGE)
  - Leiden community detection + summary storage
  - All retrieval primitives used by the agent tools
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("graphrag.store")

# ── Embedding model constant — single source of truth ───────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Config ──────────────────────────────────────────────────────────────────

def load_env() -> dict:
    from dotenv import dotenv_values
    repo = Path(__file__).parent.parent.parent
    v = dotenv_values(str(repo / ".env"))
    pwd = v.get("NEO4J_PASSWORD")
    if not pwd:
        raise EnvironmentError("NEO4J_PASSWORD env var is required")
    return {
        "neo4j_uri":      f"bolt://{v.get('NEO4J_HOST','172.18.0.1')}:{v.get('NEO4J_PORT',7687)}",  # Docker bridge IP — override with NEO4J_HOST env var
        "neo4j_user":     v.get("NEO4J_USER", "neo4j"),
        "neo4j_password": pwd,
        "neo4j_database": v.get("NEO4J_DATABASE", "neo4j"),
        "pgvector_dsn": (
            f"postgresql://{v.get('PGVECTOR_USER','postgres')}"
            f":{v.get('PGVECTOR_PASSWORD','pgvectorpass123')}"
            f"@{v.get('PGVECTOR_HOST','172.18.0.1')}"
            f":{v.get('PGVECTOR_PORT','5432')}"
            f"/{v.get('PGVECTOR_DB','vectordb')}"
        ),
        "groq_api_key": v.get("GROQ_API_KEY", ""),
        "embed_model": v.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        "llm_model": v.get("LLM_MODEL", "llama-3.3-70b-versatile"),
    }


# ── Embedding model (singleton) ───────────────────────────────────────────

_embed_model = None

def get_embedder(model_name: str = EMBED_MODEL):
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s ...", model_name)
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    m = get_embedder(model_name)
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── pgvector schema ───────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT,
    text        TEXT,
    embedding   VECTOR(384),
    entity_count INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chunks_emb_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=10);

CREATE TABLE IF NOT EXISTS entities (
    name        TEXT PRIMARY KEY,
    label       TEXT,
    embedding   VECTOR(384),
    chunk_count INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS entities_emb_idx
    ON entities USING ivfflat (embedding vector_cosine_ops) WITH (lists=10);
"""


class PgStore:
    def __init__(self, dsn: str):
        import psycopg2
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

    def ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute(SCHEMA_SQL)
        self.conn.commit()
        log.info("pgvector schema ensured")

    def upsert_chunks(self, chunks: list[dict]):
        cur = self.conn.cursor()
        for c in chunks:
            cur.execute(
                """INSERT INTO chunks (id, doc_id, text, embedding, entity_count)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET
                     text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                     entity_count=EXCLUDED.entity_count""",
                (c["id"], c["doc_id"], c["text"][:4000],
                 c["embedding"], len(c.get("entities", [])))
            )
        self.conn.commit()
        log.info("Upserted %d chunks into pgvector", len(chunks))

    def upsert_entities(self, entities: list[dict]):
        """entities: [{name, label, embedding, chunk_count}]"""
        cur = self.conn.cursor()
        for e in entities:
            cur.execute(
                """INSERT INTO entities (name, label, embedding, chunk_count)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (name) DO UPDATE SET
                     label=EXCLUDED.label, embedding=EXCLUDED.embedding,
                     chunk_count=EXCLUDED.chunk_count, updated_at=NOW()""",
                (e["name"], e["label"], e["embedding"], e.get("chunk_count", 0))
            )
        self.conn.commit()
        log.info("Upserted %d entities into pgvector", len(entities))

    def search_chunks(self, query_emb: list[float], k: int = 5) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT id, doc_id, text, 1-(embedding <=> %s::vector) AS score
               FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
            (query_emb, query_emb, k)
        )
        return [{"chunk_id": r[0], "doc_id": r[1], "text": r[2], "score": float(r[3])}
                for r in cur.fetchall()]

    def search_entities(self, query_emb: list[float], k: int = 8) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT name, label, 1-(embedding <=> %s::vector) AS score
               FROM entities ORDER BY embedding <=> %s::vector LIMIT %s""",
            (query_emb, query_emb, k)
        )
        return [{"name": r[0], "label": r[1], "score": float(r[2])}
                for r in cur.fetchall()]

    def close(self):
        self.conn.close()


# ── Neo4j store ───────────────────────────────────────────────────────────

class NeoStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db = database

    def _run(self, cypher: str, **params):
        with self.driver.session(database=self.db) as s:
            return list(s.run(cypher, **params))

    def clear(self):
        self._run("MATCH (n) DETACH DELETE n")
        log.info("Neo4j cleared")

    def ensure_indexes(self):
        try:
            self._run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            self._run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            self._run("CREATE INDEX topic_id IF NOT EXISTS FOR (t:Topic) ON (t.id)")
        except Exception as e:
            log.debug("Index creation note: %s", e)

    def upsert_chunks(self, chunks: list[dict]):
        for c in chunks:
            self._run(
                "MERGE (ch:Chunk {id:$id}) SET ch.doc_id=$doc_id, ch.text=$text",
                id=c["id"], doc_id=c["doc_id"], text=c["text"][:3000]
            )

    def upsert_entities_and_mentions(self, chunks: list[dict]):
        for c in chunks:
            for ent in c.get("entities", []):
                name = ent["text"].strip().upper()
                if not name or len(name) < 2:
                    continue
                label = ent.get("label", "Entity")
                etype = (
                    ent.get("type")
                    or ent.get("ontology_label")
                    or ent.get("coarse_bucket")
                    or label
                )
                self._run(
                    "MERGE (e:Entity {name:$name}) SET e.label=$label, e.type=$type",
                    name=name, label=label, type=etype,
                )
                self._run(
                    """MATCH (ch:Chunk {id:$cid})
                       MATCH (e:Entity {name:$name})
                       MERGE (ch)-[:MENTIONS]->(e)""",
                    cid=c["id"], name=name
                )

    def upsert_topics(self, topics: list[dict]):
        """topics: [{id, community_id, label, summary, members: [entity_names]}]"""
        for t in topics:
            self._run(
                """MERGE (tp:Topic {id:$id})
                   SET tp.label=$label, tp.summary=$summary, tp.community_id=$community_id""",
                id=t["id"], label=t["label"], summary=t["summary"],
                community_id=t.get("community_id")
            )
            for name in t.get("members", []):
                self._run(
                    """MATCH (e:Entity {name:$name})
                       MATCH (tp:Topic {id:$tid})
                       MERGE (e)-[:IN_COMMUNITY]->(tp)""",
                    name=name, tid=t["id"]
                )

    def get_entity_neighbours(self, entity_name: str, depth: int = 2) -> list[dict]:
        rows = self._run(
            """MATCH (e:Entity {name:$name})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(nbr:Entity)
               WHERE nbr.name <> $name
               RETURN nbr.name AS name, nbr.label AS label, count(c) AS shared_chunks
               ORDER BY shared_chunks DESC LIMIT 20""",
            name=entity_name.upper()
        )
        return [{"name": r["name"], "label": r["label"], "shared_chunks": r["shared_chunks"]}
                for r in rows]

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        rows = self._run(
            """MATCH (e:Entity {name:$name})<-[:MENTIONS]-(c:Chunk)
               RETURN c.id AS id, c.text AS text LIMIT 5""",
            name=entity_name.upper()
        )
        return [{"chunk_id": r["id"], "text": (r["text"] or "")[:400]} for r in rows]

    def get_entity_community(self, entity_name: str) -> Optional[dict]:
        rows = self._run(
            """MATCH (e:Entity {name:$name})-[:IN_COMMUNITY]->(t:Topic)
               RETURN t.id AS id, t.label AS label, t.summary AS summary LIMIT 1""",
            name=entity_name.upper()
        )
        if rows:
            r = rows[0]
            return {"id": r["id"], "label": r["label"], "summary": r["summary"]}
        return None

    def get_all_entity_cooccurrences(self) -> list[tuple]:
        """Return (entity_a, entity_b, weight) for Leiden."""
        rows = self._run(
            """MATCH (a:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(b:Entity)
               WHERE a.name < b.name
               RETURN a.name AS a, b.name AS b, count(c) AS w"""
        )
        return [(r["a"], r["b"], r["w"]) for r in rows]

    def get_all_entity_names(self) -> list[str]:
        rows = self._run("MATCH (e:Entity) RETURN e.name AS name")
        return [r["name"] for r in rows]

    def get_all_topics(self) -> list[dict]:
        rows = self._run(
            """MATCH (t:Topic)
               OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(t)
               RETURN t.id AS id, t.label AS label, t.summary AS summary,
                      collect(e.name) AS members"""
        )
        return [{"id": r["id"], "label": r["label"], "summary": r["summary"],
                 "members": r["members"]} for r in rows]

    def stats(self) -> dict:
        def count(label):
            r = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
            return r[0]["c"] if r else 0
        mentions = self._run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")
        return {
            "chunks":   count("Chunk"),
            "entities": count("Entity"),
            "topics":   count("Topic"),
            "mentions": mentions[0]["c"] if mentions else 0,
        }

    def close(self):
        self.driver.close()
