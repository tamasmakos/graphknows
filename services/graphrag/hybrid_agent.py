"""
Hybrid GraphRAG Agent — in-memory implementation for testing without Neo4j.

Architecture (informed by EHRAG 2604.17458 and UniAI-GraphRAG 2603.25152):
  1. Dense retrieval — sentence-transformer embeddings on chunks (semantic proximity)
  2. Entity retrieval — fuzzy/exact match on LKIF entity graph
  3. Graph walk — BFS expansion from matched entities to neighbours
  4. Fusion — RRF (Reciprocal Rank Fusion) combines dense + graph-derived chunks
  5. ReAct agent (Groq) — decides which tools to call, in what order, how many hops

Tools exposed to the agent:
  search_chunks(query, k)         — dense vector search over chunks
  search_entities(query, k)       — entity name/type search
  get_entity_neighbours(name, depth) — graph neighbourhood BFS
  get_chunks_for_entity(name)     — all chunks that mention an entity

Usage:
  python hybrid_graphrag_agent.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Load extracted data ───────────────────────────────────────────────────────
DATA_FILE = Path("/root/graphknows/output_political_smoke/lkif_chunks.json")

def _load_data() -> list[dict]:
    with open(DATA_FILE) as f:
        return json.load(f)

# ── Embedding (lazy, cached) ──────────────────────────────────────────────────
_embed_model = None
_chunk_embeddings: dict[str, list[float]] = {}

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model

def _embed(text: str) -> list[float]:
    model = _get_embed_model()
    return model.encode(text, normalize_embeddings=True).tolist()

def _cosine(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(x*x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)

# ── Index building ────────────────────────────────────────────────────────────
def build_index(chunks: list[dict]) -> dict:
    """Build in-memory indexes from extracted chunks."""
    # Entity -> list of chunk ids that mention it
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)
    # Entity -> {type, mentions, neighbours set}
    entity_graph: dict[str, dict] = {}
    # Chunk embeddings
    chunk_embs: dict[str, list[float]] = {}

    print("Building embeddings + entity index...")
    for chunk in chunks:
        cid = chunk["id"]
        chunk_embs[cid] = _embed(chunk["text"])

        # Index entities from this chunk
        for ent in chunk.get("entities", []):
            name = ent["text"].strip().upper()
            lbl = ent["label"]
            if not name or len(name) < 2:
                continue
            entity_to_chunks[name].append(cid)
            if name not in entity_graph:
                entity_graph[name] = {"type": lbl, "mentions": 0, "co_occurs_with": set()}
            entity_graph[name]["mentions"] += 1

        # Build co-occurrence edges (entities in same chunk are related)
        names_in_chunk = [e["text"].strip().upper() for e in chunk.get("entities", [])
                          if len(e["text"].strip()) > 1]
        for i, a in enumerate(names_in_chunk):
            for b in names_in_chunk[i+1:]:
                if a != b and a in entity_graph and b in entity_graph:
                    entity_graph[a]["co_occurs_with"].add(b)
                    entity_graph[b]["co_occurs_with"].add(a)

    # Convert sets to lists for serialisation
    for eg in entity_graph.values():
        eg["co_occurs_with"] = list(eg["co_occurs_with"])[:20]

    print(f"Index ready: {len(chunks)} chunks, {len(entity_graph)} entities")
    return {
        "chunks": {c["id"]: c for c in chunks},
        "chunk_embs": chunk_embs,
        "entity_to_chunks": dict(entity_to_chunks),
        "entity_graph": entity_graph,
    }

# ── Tool implementations ──────────────────────────────────────────────────────
def search_chunks(query: str, k: int = 4, index: dict = None) -> list[dict]:
    """Dense semantic search over chunks. Returns top-k with scores."""
    qemb = _embed(query)
    scored = [
        (cid, _cosine(qemb, emb))
        for cid, emb in index["chunk_embs"].items()
    ]
    scored.sort(key=lambda x: -x[1])
    results = []
    for cid, score in scored[:k]:
        chunk = index["chunks"][cid]
        results.append({
            "chunk_id": cid,
            "score": round(score, 3),
            "text": chunk["text"][:400],
            "entity_count": len(chunk.get("entities", [])),
        })
    return results

def search_entities(query: str, k: int = 8, index: dict = None) -> list[dict]:
    """Search entities by name/type substring match."""
    query_upper = query.upper()
    query_words = set(re.findall(r"[A-Z]{2,}", query_upper))
    results = []
    for name, info in index["entity_graph"].items():
        name_words = set(re.findall(r"[A-Z]{2,}", name))
        overlap = len(query_words & name_words)
        if overlap > 0 or query_upper in name:
            results.append({
                "name": name,
                "type": info["type"],
                "mentions": info["mentions"],
                "neighbours": len(info["co_occurs_with"]),
                "score": overlap + (2 if query_upper in name else 0),
            })
    results.sort(key=lambda x: -x["score"])
    return results[:k]

def get_entity_neighbours(entity_name: str, depth: int = 1, index: dict = None) -> list[dict]:
    """BFS graph walk from entity, return neighbours up to depth hops."""
    entity_name = entity_name.strip().upper()
    if entity_name not in index["entity_graph"]:
        # Try partial match
        matches = [n for n in index["entity_graph"] if entity_name in n]
        if not matches:
            return [{"error": f"Entity '{entity_name}' not found"}]
        entity_name = matches[0]

    visited = {entity_name}
    frontier = {entity_name}
    results = []
    for d in range(depth):
        next_frontier = set()
        for node in frontier:
            for nbr in index["entity_graph"].get(node, {}).get("co_occurs_with", []):
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    info = index["entity_graph"].get(nbr, {})
                    results.append({
                        "name": nbr,
                        "type": info.get("type", "?"),
                        "mentions": info.get("mentions", 0),
                        "hop": d + 1,
                    })
        frontier = next_frontier
    return results[:25]

def get_chunks_for_entity(entity_name: str, index: dict = None) -> list[dict]:
    """Return all chunks that mention a given entity."""
    entity_name = entity_name.strip().upper()
    # Try exact then partial
    chunk_ids = index["entity_to_chunks"].get(entity_name, [])
    if not chunk_ids:
        for name, cids in index["entity_to_chunks"].items():
            if entity_name in name:
                chunk_ids = cids
                entity_name = name
                break
    if not chunk_ids:
        return [{"error": f"No chunks found for entity '{entity_name}'"}]
    results = []
    for cid in chunk_ids[:5]:
        chunk = index["chunks"][cid]
        results.append({
            "chunk_id": cid,
            "text": chunk["text"][:400],
        })
    return results

# ── RRF fusion ────────────────────────────────────────────────────────────────
def rrf_fuse(dense_results: list[dict], graph_chunk_ids: list[str],
             index: dict, k_rrf: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion of dense vector results and graph-derived chunk IDs.
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, r in enumerate(dense_results):
        scores[r["chunk_id"]] += 1.0 / (k_rrf + rank + 1)
    for rank, cid in enumerate(graph_chunk_ids):
        scores[cid] += 1.0 / (k_rrf + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results = []
    for cid, score in ranked[:6]:
        chunk = index["chunks"].get(cid, {})
        results.append({
            "chunk_id": cid,
            "rrf_score": round(score, 4),
            "text": chunk.get("text", "")[:400],
        })
    return results

# ── ReAct agent ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a political intelligence analyst with access to a knowledge graph 
built from EU Parliament debates (April 2026). You answer questions by using tools to retrieve 
relevant information, then synthesizing a grounded answer.

Available tools:
  search_chunks(query, k=4) — semantic search over debate text chunks
  search_entities(query, k=8) — find entities (persons, treaties, orgs, legal acts) by name
  get_entity_neighbours(entity_name, depth=1) — find entities co-occurring with a given entity
  get_chunks_for_entity(entity_name) — retrieve debate passages mentioning an entity

Strategy:
  1. Always start with search_chunks and search_entities for the query
  2. If entities are found, use get_entity_neighbours to find related entities
  3. Use get_chunks_for_entity for the most relevant entities to get supporting text
  4. Fuse all retrieved context to write a grounded answer with citations like [chunk_0]

Respond using this format:
  Thought: <reasoning about what to do next>
  Action: <tool_name>(<args>)
  Observation: <tool result>
  ... (repeat up to 6 times)
  Final Answer: <your grounded answer citing chunk IDs>
"""

def _call_groq(messages: list[dict], stop: list[str] | None = None,
               model: str = "llama-3.3-70b-versatile") -> str:
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        env = Path("/root/graphknows/.env").read_text()
        for line in env.splitlines():
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip()
    client = Groq(api_key=key)
    kwargs = dict(model=model, messages=messages, temperature=0.1, max_tokens=512)
    if stop:
        kwargs["stop"] = stop
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def _parse_action(text: str) -> tuple[str | None, str | None]:
    """Extract (tool_name, args_str) from 'Action: tool(args)' line."""
    m = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
    if m:
        return m.group(1), m.group(2)
    return None, None

def _eval_args(args_str: str) -> tuple[list, dict]:
    """
    Parse tool arguments from model output.
    Handles: "value", kwarg=value, "kwarg"="value", entity_name="value"
    Returns (positional_args, keyword_args) as plain Python types.
    """
    args_str = args_str.strip()
    if not args_str:
        return [], {}

    pos_args: list = []
    kw_args: dict = {}

    # Split on commas not inside quotes
    parts = re.split(r',\s*(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # keyword=value pattern
        kv = re.match(r'^(\w+)\s*=\s*["\']?([^"\']*)["\']?$', part)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            try:
                kw_args[key] = int(val)
            except ValueError:
                kw_args[key] = val
        else:
            # positional — strip quotes
            val = part.strip('"').strip("'").strip()
            if val:
                pos_args.append(val)

    # Hoist common single-arg keyword params to positional
    # so tools can be called as tool(name, depth) or tool(entity_name=name)
    if not pos_args and kw_args:
        for param in ("query", "entity_name", "doc_id"):
            if param in kw_args:
                pos_args.append(kw_args.pop(param))
                break

    return pos_args, kw_args

def run_hybrid_agent(query: str, index: dict, max_steps: int = 6) -> str:
    """Run the ReAct hybrid agent loop with proper stop-token control."""
    tools = {
        "search_chunks": lambda *a, **kw: search_chunks(*a, **kw, index=index),
        "search_entities": lambda *a, **kw: search_entities(*a, **kw, index=index),
        "get_entity_neighbours": lambda *a, **kw: get_entity_neighbours(*a, **kw, index=index),
        "get_chunks_for_entity": lambda *a, **kw: get_chunks_for_entity(*a, **kw, index=index),
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")

    full_trace = ""

    for step in range(max_steps):
        # Stop at "Observation:" — prevent the model from inventing tool results
        response = _call_groq(messages, stop=["Observation:"])
        print(f"--- Step {step+1} ---")
        print(response.strip())
        full_trace += response

        if "Final Answer:" in response:
            break

        tool_name, args_str = _parse_action(response)
        if tool_name and tool_name in tools:
            pos_args, kw_args = _eval_args(args_str or "")
            try:
                result = tools[tool_name](*pos_args, **kw_args)
                obs = json.dumps(result, indent=2)
                if len(obs) > 1200:
                    obs = obs[:1200] + "\n... [truncated]"
            except Exception as e:
                obs = f"Tool error: {e}"
            print(f"\nObservation: {obs[:500]}\n")
            messages.append({"role": "assistant", "content": response.strip()})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            full_trace += f"\nObservation: {obs}\n"
        else:
            messages.append({"role": "assistant", "content": response.strip()})
            messages.append({"role": "user",
                             "content": "Please use one of the tools now, or write your Final Answer."})

    return full_trace

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = _load_data()
    index = build_index(chunks)

    queries = [
        "What legal acts or treaties were referenced in the EU Parliament debate?",
        "What was discussed about energy and strategic independence?",
    ]

    for q in queries[:1]:
        answer = run_hybrid_agent(q, index)
        print("\n" + "="*70)
        print("FINAL ANSWER EXTRACTED")
        print("="*70)
        if "Final Answer:" in answer:
            print(answer.split("Final Answer:")[-1].strip())
