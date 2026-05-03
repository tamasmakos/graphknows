"""
ingest.py — full ingestion pipeline for GraphKnows.

Steps:
  1. Load pre-extracted LKIF chunks (lkif_chunks.json)
  2. Normalise entity format
  3. Embed chunks + entity names with BGE-small-en-v1.5
  4. Upload to Neo4j (Chunk, Entity, MENTIONS)
  5. Upload to pgvector (chunk embeddings + entity embeddings)
  6. Run Leiden community detection on entity co-occurrence graph
  7. LLM-summarize each community -> Topic nodes in Neo4j
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "services" / "graphgen"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graphrag.ingest")


def normalise(raw: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(raw):
        chunk_id = c.get("id", f"chunk_{i:04d}")
        doc_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else "ep_debate"
        ents = []
        for e in c.get("entities", []):
            lbl = e.get("ontology_label") or e.get("label") or "Entity"
            txt = e.get("text", "").strip()
            if txt and len(txt) >= 2:
                ents.append({"text": txt, "label": lbl})
        out.append({"id": chunk_id, "doc_id": doc_id, "text": c["text"], "entities": ents})
    return out


def run_leiden(cooccurrences: list[tuple]) -> dict[str, int]:
    """Return entity_name -> community_id mapping."""
    import igraph as ig
    import leidenalg

    if not cooccurrences:
        return {}

    all_names = list({n for a, b, _ in cooccurrences for n in (a, b)})
    idx = {n: i for i, n in enumerate(all_names)}

    edges = [(idx[a], idx[b]) for a, b, _ in cooccurrences]
    weights = [float(w) for _, _, w in cooccurrences]

    g = ig.Graph()
    g.add_vertices(len(all_names))
    g.add_edges(edges)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition,
        weights="weight", n_iterations=10, seed=42
    )
    log.info("Leiden: %d communities, modularity=%.4f",
             len(partition), partition.modularity)

    return {all_names[i]: partition.membership[i] for i in range(len(all_names))}


def summarize_community(members: list[str], api_key: str, model: str) -> str:
    """LLM-summarize a community of entity names into a topic label + description."""
    from groq import Groq
    client = Groq(api_key=api_key)
    member_str = ", ".join(members[:40])
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                f"These political/legal entities were discussed together in EU Parliament debates:\n"
                f"{member_str}\n\n"
                "In one short sentence (max 20 words), what topic or theme do they share? "
                "Reply with ONLY the topic sentence, no preamble."
            )
        }],
        max_tokens=60,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def ingest(chunks_file: Path, cfg: dict, summarize: bool = True):
    from graph_store import NeoStore, PgStore, embed

    log.info("Loading chunks from %s", chunks_file)
    raw = json.loads(chunks_file.read_text())
    chunks = normalise(raw)
    log.info("Normalised %d chunks", len(chunks))

    # ── NERValidator (optional, non-blocking) ─────────────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from ner_validator import NERValidator
        _validator = NERValidator()
        if _validator.available:
            _total_rate = 0.0
            _validated = 0
            for _c in chunks[:20]:  # sample first 20 chunks
                _ents = [{"text": e["text"], "label": e["label"]} for e in _c.get("entities", [])]
                if _ents:
                    _report = _validator.validate(_c["text"], _ents)
                    _rate = _report.get("agreement_rate", 1.0)
                    _total_rate += _rate
                    _validated += 1
            if _validated > 0:
                _avg_rate = _total_rate / _validated
                if _avg_rate < 0.3:
                    log.warning("NERValidator: low spaCy agreement rate %.2f across %d sampled chunks", _avg_rate, _validated)
                else:
                    log.info("NERValidator: spaCy agreement rate %.2f across %d sampled chunks", _avg_rate, _validated)
    except ImportError:
        pass
    except Exception as _e:
        log.debug("NERValidator skipped: %s", _e)

    # ── Embed chunks ──────────────────────────────────────────────────────
    log.info("Embedding %d chunks ...", len(chunks))
    texts = [c["text"] for c in chunks]
    chunk_embs = embed(texts, cfg["embed_model"])
    for c, emb in zip(chunks, chunk_embs):
        c["embedding"] = emb.tolist()

    # ── Embed entity names ────────────────────────────────────────────────
    # collect unique entities with their LKIF label and chunk count
    ent_info: dict[str, dict] = {}
    for c in chunks:
        for e in c["entities"]:
            name = e["text"].strip().upper()
            if not name or len(name) < 2:
                continue
            if name not in ent_info:
                ent_info[name] = {"label": e["label"], "chunk_count": 0, "canonical": e["text"]}
            ent_info[name]["chunk_count"] += 1

    ent_names = list(ent_info.keys())
    log.info("Embedding %d unique entity names ...", len(ent_names))
    ent_embs = embed(ent_names, cfg["embed_model"])
    entity_rows = []
    for name, emb in zip(ent_names, ent_embs):
        entity_rows.append({
            "name": name,
            "label": ent_info[name]["label"],
            "embedding": emb.tolist(),
            "chunk_count": ent_info[name]["chunk_count"],
        })

    # ── Neo4j ─────────────────────────────────────────────────────────────
    log.info("Uploading to Neo4j ...")
    neo = NeoStore(cfg["neo4j_uri"], cfg["neo4j_user"], cfg["neo4j_password"], cfg["neo4j_database"])
    neo.ensure_indexes()
    neo.upsert_chunks(chunks)
    neo.upsert_entities_and_mentions(chunks)

    # ── pgvector ──────────────────────────────────────────────────────────
    log.info("Uploading to pgvector ...")
    pg = PgStore(cfg["pgvector_dsn"])
    pg.ensure_schema()
    pg.upsert_chunks(chunks)
    pg.upsert_entities(entity_rows)

    # ── Leiden community detection ────────────────────────────────────────
    log.info("Running Leiden community detection ...")
    cooc = neo.get_all_entity_cooccurrences()
    if cooc:
        membership = run_leiden(cooc)

        # Group entities by community
        community_members: dict[int, list[str]] = defaultdict(list)
        for name, cid in membership.items():
            community_members[cid].append(name)

        topics = []
        for cid, members in sorted(community_members.items(), key=lambda x: -len(x[1])):
            if len(members) < 2:
                continue
            summary = f"Community of {len(members)} entities including: " + ", ".join(members[:5])
            if summarize and cfg.get("groq_api_key"):
                try:
                    summary = summarize_community(members, cfg["groq_api_key"], cfg["llm_model"])
                    log.info("Topic %d: %s", cid, summary)
                except Exception as e:
                    log.warning("LLM summary failed for community %d: %s", cid, e)

            topics.append({
                "id": f"topic_{cid}",
                "label": summary[:80],
                "summary": summary,
                "members": members,
            })

        neo.upsert_topics(topics)
        log.info("Stored %d topic nodes in Neo4j", len(topics))
    else:
        log.warning("No co-occurrences found — Leiden skipped")

    stats = neo.stats()
    pg_count = "?"
    try:
        cur = pg.conn.cursor()
        cur.execute("SELECT count(*) FROM chunks")
        pg_count = cur.fetchone()[0]
    except Exception:
        pass

    neo.close()
    pg.close()

    return {
        "chunks": stats.get("chunks", len(chunks)),
        "entities": stats.get("entities", len(entity_rows)),
        "topics": stats.get("topics", 0),
        "mentions": stats.get("mentions", 0),
        "pgvector_chunks": pg_count,
    }


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_store import load_env

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=str(ROOT / "output_political_smoke" / "lkif_chunks.json"))
    parser.add_argument("--no-summarize", action="store_true")
    args = parser.parse_args()

    cfg = load_env()
    result = ingest(Path(args.chunks), cfg, summarize=not args.no_summarize)

    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    for k, v in result.items():
        print(f"  {k:<25} {v}")
