"""
demo_ingest_upload.py — uploads already-extracted LKIF chunks to Neo4j + pgvector.

This is step 2 of the demo pipeline, after extraction has produced lkif_chunks.json.
It normalises the chunk format, embeds with BGE-small-en, and writes to both stores.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "services" / "graphgen"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo_upload")


def load_config() -> dict:
    from dotenv import dotenv_values
    vals = dotenv_values(str(ROOT / ".env"))
    return {
        "neo4j_host":     vals.get("NEO4J_HOST", "172.18.0.1"),
        "neo4j_port":     int(vals.get("NEO4J_PORT", 7687)),
        "neo4j_user":     vals.get("NEO4J_USER", "neo4j"),
        "neo4j_password": vals.get("NEO4J_PASSWORD", ""),
        "neo4j_database": vals.get("NEO4J_DATABASE", "neo4j"),
        "pgvector_dsn": (
            f"postgresql://{vals.get('PGVECTOR_USER','postgres')}"
            f":{vals.get('PGVECTOR_PASSWORD','')}"
            f"@{vals.get('PGVECTOR_HOST','172.18.0.1')}"
            f":{vals.get('PGVECTOR_PORT','5432')}"
            f"/{vals.get('PGVECTOR_DB','vectordb')}"
        ),
    }


def normalise_chunks(raw: list[dict]) -> list[dict]:
    """Convert lkif_chunks.json format to the ingestion format."""
    out = []
    for i, c in enumerate(raw):
        # lkif format: {id, text, offset, entities: [{text, label, ontology_label, ...}]}
        ents = []
        for e in c.get("entities", []):
            lbl = e.get("ontology_label") or e.get("label") or "Entity"
            txt = e.get("text", "").strip()
            if txt and len(txt) >= 2:
                ents.append({"text": txt, "label": lbl})
        out.append({
            "id":       c.get("id", f"chunk_{i:04d}"),
            "doc_id":   c.get("id", f"chunk_{i:04d}").rsplit("_chunk_", 1)[0] if "_chunk_" in c.get("id","") else "ep_debate_2024_03_13",
            "text":     c["text"],
            "entities": ents,
        })
    return out


def embed_chunks(chunks: list[dict]) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    log.info("Embedding %d chunks with BGE-small-en-v1.5...", len(chunks))
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for chunk, emb in zip(chunks, embs):
        chunk["embedding"] = emb.tolist()
    log.info("Embeddings complete (dim=%d)", len(embs[0]))
    return chunks


def upload_neo4j(chunks: list[dict], cfg: dict) -> dict:
    from neo4j import GraphDatabase
    uri  = f"bolt://{cfg['neo4j_host']}:{cfg['neo4j_port']}"
    auth = (cfg["neo4j_user"], cfg["neo4j_password"])
    db   = cfg["neo4j_database"]
    log.info("Connecting to Neo4j at %s ...", uri)
    driver = GraphDatabase.driver(uri, auth=auth)

    node_ops = rel_ops = 0
    with driver.session(database=db) as s:
        for chunk in chunks:
            s.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.doc_id = $doc_id, c.text = $text, c.entity_count = $ec
                """,
                id=chunk["id"], doc_id=chunk["doc_id"],
                text=chunk["text"][:2000], ec=len(chunk["entities"]),
            )
            node_ops += 1
            for ent in chunk["entities"]:
                name = ent["text"].strip().upper()
                if not name or len(name) < 2:
                    continue
                s.run(
                    "MERGE (e:Entity {name: $name}) SET e.label = $label",
                    name=name, label=ent["label"],
                )
                s.run(
                    """
                    MATCH (c:Chunk {id: $cid}), (e:Entity {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    cid=chunk["id"], name=name,
                )
                node_ops += 1
                rel_ops  += 1

    # Stats
    with driver.session(database=db) as s:
        n_chunks  = s.run("MATCH (c:Chunk)  RETURN count(c) AS n").single()["n"]
        n_ents    = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        n_rels    = s.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n").single()["n"]

    driver.close()
    log.info("Neo4j: %d Chunk nodes, %d Entity nodes, %d MENTIONS edges", n_chunks, n_ents, n_rels)
    return {"chunks": n_chunks, "entities": n_ents, "mentions": n_rels}


def upload_pgvector(chunks: list[dict], dsn: str) -> int:
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(dsn)
    if not pg.connect():
        raise RuntimeError("Cannot connect to pgvector")
    pg.ensure_schema()
    n = pg.upsert_chunks(chunks)
    total = pg.chunk_count()
    pg.close()
    log.info("pgvector: %d chunks inserted (DB total: %d)", n, total)
    return n


def main():
    cfg = load_config()

    chunks_file = ROOT / "output_political_smoke" / "lkif_chunks.json"
    log.info("Loading extracted chunks from %s", chunks_file)
    with open(chunks_file) as f:
        raw = json.load(f)
    log.info("Loaded %d raw chunks", len(raw))

    chunks = normalise_chunks(raw)

    total_ents = sum(len(c["entities"]) for c in chunks)
    unique_labels = {e["label"] for c in chunks for e in c["entities"]}
    log.info("Normalised: %d chunks, %d entities, %d unique LKIF labels", 
             len(chunks), total_ents, len(unique_labels))

    chunks = embed_chunks(chunks)

    neo4j_stats = upload_neo4j(chunks, cfg)
    pgv_n       = upload_pgvector(chunks, cfg["pgvector_dsn"])

    from collections import Counter
    label_counts = Counter(e["label"] for c in chunks for e in c["entities"])

    print("\n" + "="*65)
    print("INGESTION COMPLETE")
    print("="*65)
    print(f"  Source file:      output_political_smoke/lkif_chunks.json")
    print(f"  Corpus:           EP 'This is Europe' debates (LKIF extraction)")
    print(f"  Chunks uploaded:  {len(chunks)}")
    print(f"  Entities:         {total_ents}  ({len(unique_labels)} unique LKIF labels)")
    print()
    print("  Neo4j graph:")
    print(f"    Chunk nodes:    {neo4j_stats['chunks']}")
    print(f"    Entity nodes:   {neo4j_stats['entities']}")
    print(f"    MENTIONS edges: {neo4j_stats['mentions']}")
    print()
    print(f"  pgvector chunks:  {pgv_n}")
    print()
    print("  Top entity types (LKIF ontology labels):")
    for label, cnt in label_counts.most_common(12):
        print(f"    {label:<35} {cnt}")


if __name__ == "__main__":
    main()
