"""
hybrid_agent.py — Hybrid GraphRAG agent backed by live Neo4j + pgvector.

Architecture (EHRAG 2604.17458 / UniAI-GraphRAG 2603.25152):
  1. Dense retrieval  — pgvector cosine search over BGE-small-en chunk embeddings
  2. Entity retrieval — Neo4j substring match on Entity nodes
  3. Graph walk       — Neo4j BFS MENTIONS traversal from matched entities
  4. RRF fusion       — Reciprocal Rank Fusion merges both result lists
  5. ReAct agent      — Groq llama-3.3-70b decides tool sequence, writes grounded answer

All connection parameters are loaded from .env — nothing hardcoded.

Usage:
  python hybrid_agent.py [--query "..."] [--steps 6]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
REPO = ROOT.parent.parent  # services/graphrag -> repo root
sys.path.insert(0, str(REPO / "services" / "graphgen"))
sys.path.insert(0, str(REPO / "services" / "graphrag"))

from graph_store import EMBED_MODEL  # single source of truth for embedding model


# ── Config ────────────────────────────────────────────────────────────────────

def _load_env() -> dict:
    from dotenv import dotenv_values
    env = dotenv_values(str(REPO / ".env"))
    return {
        "neo4j_uri":     f"bolt://{env.get('NEO4J_HOST','172.18.0.1')}:{env.get('NEO4J_PORT',7687)}",
        "neo4j_user":    env.get("NEO4J_USER", "neo4j"),
        "neo4j_password": env.get("NEO4J_PASSWORD", ""),
        "neo4j_database": env.get("NEO4J_DATABASE", "neo4j"),
        "pgvector_dsn": (
            f"postgresql://{env.get('PGVECTOR_USER', 'postgres')}"
            f":{env.get('PGVECTOR_PASSWORD', '')}"
            f"@{env.get('PGVECTOR_HOST', '172.18.0.1')}"
            f":{env.get('PGVECTOR_PORT', 5432)}"
            f"/{env.get('PGVECTOR_DB', 'vectordb')}"
        ),
        "groq_api_key":  env.get("GROQ_API_KEY", ""),
    }


# ── Embedding ─────────────────────────────────────────────────────────────────

_embed_model = None

def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


# ── Tool implementations ──────────────────────────────────────────────────────

def search_chunks(query: str, k: int = 5, *, cfg: dict) -> list[dict]:
    """Dense vector search over debate chunks stored in pgvector."""
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(cfg["pgvector_dsn"])
    if not pg.connect():
        return [{"error": "pgvector not reachable"}]
    qemb = _embed(query)
    results = pg.search_chunks(qemb, k=k)
    pg.close()
    return results


def search_entities(query: str, k: int = 8, *, cfg: dict) -> list[dict]:
    """Search Entity nodes in Neo4j by name substring."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"]))
    results = []
    with driver.session(database=cfg["neo4j_database"]) as s:
        rows = s.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($q)
            RETURN e.name AS name, e.label AS label,
                   size([(e)<-[:MENTIONS]-() | 1]) AS mentions
            ORDER BY mentions DESC
            LIMIT $k
            """,
            q=query, k=k,
        ).data()
        results = [{"name": r["name"], "label": r["label"], "mentions": r["mentions"]} for r in rows]
    driver.close()
    return results


def get_entity_neighbours(entity_name: str, depth: int = 1, *, cfg: dict) -> list[dict]:
    """BFS over co-MENTIONS edges in Neo4j (entities sharing a chunk)."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"]))
    results = []
    with driver.session(database=cfg["neo4j_database"]) as s:
        rows = s.run(
            """
            MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(nbr:Entity)
            WHERE nbr.name <> $name
            RETURN DISTINCT nbr.name AS name, nbr.label AS label,
                   count(c) AS shared_chunks
            ORDER BY shared_chunks DESC
            LIMIT 20
            """,
            name=entity_name.strip().upper(),
        ).data()
        results = [{"name": r["name"], "label": r["label"], "shared_chunks": r["shared_chunks"]} for r in rows]
    driver.close()
    return results


def get_chunks_for_entity(entity_name: str, *, cfg: dict) -> list[dict]:
    """Return debate chunks that MENTION a given entity (from Neo4j + pgvector text)."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"]))
    chunk_ids = []
    with driver.session(database=cfg["neo4j_database"]) as s:
        rows = s.run(
            """
            MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)
            RETURN c.id AS id, c.text AS text
            LIMIT 5
            """,
            name=entity_name.strip().upper(),
        ).data()
        chunk_ids = [{"chunk_id": r["id"], "text": (r["text"] or "")[:400]} for r in rows]
    driver.close()
    return chunk_ids


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(dense: list[dict], graph_chunks: list[dict], k_rrf: int = 60) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    texts:  dict[str, str]   = {}
    for rank, r in enumerate(dense):
        cid = r["chunk_id"]
        scores[cid] += 1.0 / (k_rrf + rank + 1)
        texts[cid]   = r.get("text", "")
    for rank, r in enumerate(graph_chunks):
        cid = r["chunk_id"]
        scores[cid] += 1.0 / (k_rrf + rank + 1)
        texts.setdefault(cid, r.get("text", ""))
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [
        {"chunk_id": cid, "rrf_score": round(sc, 4), "text": texts.get(cid, "")[:400]}
        for cid, sc in ranked[:6]
    ]


# ── ReAct loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a political intelligence analyst. You have access to a knowledge graph built from
EU Parliament 'This is Europe' debates (2022-2024). Answer questions by calling tools to
retrieve grounded evidence, then synthesising a precise answer.

Available tools:
  search_chunks(query, k=5)              — semantic search over debate text passages
  search_entities(query, k=8)            — find named entities (persons, treaties, orgs, acts)
  get_entity_neighbours(entity_name, depth=1) — co-mentioned entities in the graph
  get_chunks_for_entity(entity_name)     — debate passages that mention a specific entity

Strategy:
  1. Start with search_chunks + search_entities
  2. Expand promising entities with get_entity_neighbours
  3. Use get_chunks_for_entity for key entities to get supporting text
  4. Write a grounded Final Answer citing evidence as [chunk_id]

Format:
  Thought: <reasoning>
  Action: <tool_name>(<args>)
  Observation: <tool result>
  ... (up to 6 steps)
  Final Answer: <grounded answer with citations>
"""


def _call_groq(messages: list[dict], stop: list[str] | None, api_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=600,
        stop=stop or None,
    )
    return resp.choices[0].message.content


def _parse_action(text: str) -> tuple[str | None, str | None]:
    m = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _eval_args(args_str: str) -> tuple[list, dict]:
    args_str = args_str.strip()
    if not args_str:
        return [], {}
    pos, kw = [], {}
    for part in re.split(r',\s*(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str):
        part = part.strip()
        if not part:
            continue
        kv = re.match(r'^(\w+)\s*=\s*["\']?([^"\']*)["\']?$', part)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            try:
                kw[key] = int(val)
            except ValueError:
                kw[key] = val
        else:
            pos.append(part.strip('"').strip("'").strip())
    if not pos and kw:
        for param in ("query", "entity_name"):
            if param in kw:
                pos.append(kw.pop(param))
                break
    return pos, kw


def run_agent(query: str, cfg: dict, max_steps: int = 6) -> str:
    tools = {
        "search_chunks":         lambda *a, **kw: search_chunks(*a, **kw, cfg=cfg),
        "search_entities":       lambda *a, **kw: search_entities(*a, **kw, cfg=cfg),
        "get_entity_neighbours": lambda *a, **kw: get_entity_neighbours(*a, **kw, cfg=cfg),
        "get_chunks_for_entity": lambda *a, **kw: get_chunks_for_entity(*a, **kw, cfg=cfg),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")

    trace = ""
    for step in range(max_steps):
        response = _call_groq(messages, stop=["Observation:"], api_key=cfg["groq_api_key"])
        print(f"--- Step {step+1} ---")
        print(response.strip())
        trace += response

        if "Final Answer:" in response:
            break

        tool_name, args_str = _parse_action(response)
        if tool_name and tool_name in tools:
            pos_args, kw_args = _eval_args(args_str or "")
            try:
                result = tools[tool_name](*pos_args, **kw_args)
                obs = json.dumps(result, indent=2)
                if len(obs) > 1400:
                    obs = obs[:1400] + "\n... [truncated]"
            except Exception as exc:
                obs = f"Tool error: {exc}"
            print(f"\nObservation: {obs[:600]}\n")
            messages.append({"role": "assistant", "content": response.strip()})
            messages.append({"role": "user",      "content": f"Observation: {obs}"})
            trace += f"\nObservation: {obs}\n"
        else:
            messages.append({"role": "assistant", "content": response.strip()})
            messages.append({"role": "user",      "content": "Use a tool or write your Final Answer."})

    return trace


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphKnows hybrid GraphRAG agent")
    parser.add_argument("--query", default="What security threats and defence priorities were discussed by EU leaders?")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    cfg = _load_env()
    trace = run_agent(args.query, cfg, max_steps=args.steps)

    print("\n" + "="*70)
    print("FINAL ANSWER")
    print("="*70)
    if "Final Answer:" in trace:
        print(trace.split("Final Answer:")[-1].strip())
    else:
        print("(No final answer reached within step limit)")