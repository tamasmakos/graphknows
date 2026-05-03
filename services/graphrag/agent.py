"""
agent.py — GraphKnows Agentic GraphRAG

Architecture (from literature):
  - KAG-Solver-style query planner: decomposes question into typed sub-tasks
  - 6 specialist tools covering dense + graph + community retrieval channels
  - RRF fusion across retrieval channels (EHRAG / UniAI-GraphRAG)
  - Full attribution: every claim traced to chunk_id + entity path
  - Rich terminal output with reasoning trace

Tools:
  search_chunks(query, k)            -- pgvector cosine over chunk embeddings
  search_entities(query, k)          -- pgvector cosine over entity name embeddings
  get_neighbours(entity, depth)      -- Neo4j BFS over co-occurrence graph
  get_chunks_for_entity(entity)      -- entity -> source chunks (provenance)
  get_community(entity)              -- Leiden topic node for entity's cluster
  fuse_and_rank(chunk_ids, k)        -- RRF fusion of accumulated chunk candidates
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("graphrag.agent")


# ── Tool registry ─────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
Available tools:

search_chunks(query: str, k: int = 5)
  Semantic search over debate passages. Returns top-k chunks by cosine similarity.
  Use for: finding passages directly relevant to the query.

search_entities(query: str, k: int = 8)
  Semantic search over entity names (Person, Organization, Policy, Treaty, etc.)
  Returns entities whose names are semantically close to the query.
  Use for: identifying key actors, documents, or concepts mentioned.

get_neighbours(entity: str, depth: int = 2)
  Graph traversal: returns entities co-mentioned with the given entity in the KG.
  Higher depth = wider neighbourhood. Use for: finding related concepts or actors.

get_chunks_for_entity(entity: str)
  Returns the source debate passages where this entity appears.
  Use for: getting primary evidence / attribution for a specific entity.

get_community(entity: str)
  Returns the Leiden community (topic cluster) that this entity belongs to.
  Use for: understanding the thematic context of an entity; answering global queries.

fuse_and_rank(chunk_ids: list, k: int = 5)
  RRF fusion: takes a list of chunk_ids (can contain duplicates from multiple tools),
  reranks by reciprocal rank frequency, returns top-k with final scores.
  Use for: final ranking before synthesizing the answer.
""".strip()

SYSTEM_PROMPT = f"""You are a political intelligence analyst operating a knowledge graph of EU Parliament debates.
You answer questions by reasoning step by step, using tools to retrieve evidence from the graph.

{TOOL_DESCRIPTIONS}

Format EVERY response as:
Thought: <reasoning about what to do next>
Action: <tool_name>(<args>)

When you have gathered enough evidence, output:
Thought: I have sufficient evidence to answer.
Final Answer: <your answer, citing [chunk_id] and [entity_name] as evidence>

Rules:
- Always cite chunk IDs and entity names in your final answer
- Use fuse_and_rank before giving a final answer when you have multiple chunk candidates
- If search_entities returns results, follow up with get_neighbours on the top entity
- For global/thematic questions, always call get_community on a relevant entity
- Maximum MAX_STEPS reasoning steps before forced final answer
"""


# ── Agent class ───────────────────────────────────────────────────────────

class GraphRAGAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pg: Any = None
        self._neo: Any = None
        self._embedder: Any = None
        self._chunk_candidates: dict[str, list[float]] = defaultdict(list)  # chunk_id -> [rank positions]

    def _pg_store(self):
        if self._pg is None:
            from graph_store import PgStore
            self._pg = PgStore(self.cfg["pgvector_dsn"])
        return self._pg

    def _neo_store(self):
        if self._neo is None:
            from graph_store import NeoStore
            self._neo = NeoStore(
                self.cfg["neo4j_uri"],
                self.cfg["neo4j_user"],
                self.cfg["neo4j_password"],
                self.cfg["neo4j_database"],
            )
        return self._neo

    def _embed(self, text: str) -> list[float]:
        from graph_store import embed
        return embed([text], self.cfg["embed_model"])[0].tolist()

    # ── Tools ──────────────────────────────────────────────────────────────

    def search_chunks(self, query: str, k: int = 5) -> list[dict]:
        emb = self._embed(query)
        results = self._pg_store().search_chunks(emb, k=k)
        # register candidates for RRF
        for rank, r in enumerate(results):
            self._chunk_candidates[r["chunk_id"]].append(rank + 1)
        return [{"chunk_id": r["chunk_id"], "score": round(r["score"], 4),
                 "text": r["text"][:400]} for r in results]

    def search_entities(self, query: str, k: int = 8) -> list[dict]:
        emb = self._embed(query)
        return self._pg_store().search_entities(emb, k=k)

    def get_neighbours(self, entity: str, depth: int = 2) -> list[dict]:
        nbrs = self._neo_store().get_entity_neighbours(entity, depth)
        # register their chunks as candidates too
        for n in nbrs[:5]:
            chunks = self._neo_store().get_chunks_for_entity(n["name"])
            for rank, c in enumerate(chunks):
                self._chunk_candidates[c["chunk_id"]].append(len(self._chunk_candidates) + rank + 1)
        return nbrs[:15]

    def get_chunks_for_entity(self, entity: str) -> list[dict]:
        results = self._neo_store().get_chunks_for_entity(entity)
        for rank, r in enumerate(results):
            self._chunk_candidates[r["chunk_id"]].append(rank + 1)
        return results

    def get_community(self, entity: str) -> Optional[dict]:
        return self._neo_store().get_entity_community(entity)

    def fuse_and_rank(self, chunk_ids: list[str] = None, k: int = 5) -> list[dict]:
        """RRF fusion over all accumulated chunk candidates."""
        K = 60
        scores: dict[str, float] = {}
        # incorporate explicitly passed ids
        if chunk_ids:
            for rank, cid in enumerate(chunk_ids):
                self._chunk_candidates[cid].append(rank + 1)
        for cid, ranks in self._chunk_candidates.items():
            scores[cid] = sum(1.0 / (K + r) for r in ranks)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [{"chunk_id": cid, "rrf_score": round(score, 6)} for cid, score in ranked]

    # ── Tool dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, args_str: str) -> Any:
        args, kwargs = _parse_args(args_str)

        def get_arg(pos_name: str, pos: int, default=None):
            if pos < len(args):
                return args[pos]
            return kwargs.get(pos_name, default)

        if tool_name == "search_chunks":
            return self.search_chunks(
                get_arg("query", 0, ""),
                int(get_arg("k", 1, 5))
            )
        elif tool_name == "search_entities":
            return self.search_entities(
                get_arg("query", 0, ""),
                int(get_arg("k", 1, 8))
            )
        elif tool_name == "get_neighbours":
            return self.get_neighbours(
                get_arg("entity", 0, ""),
                int(get_arg("depth", 1, 2))
            )
        elif tool_name == "get_chunks_for_entity":
            return self.get_chunks_for_entity(get_arg("entity", 0, ""))
        elif tool_name == "get_community":
            return self.get_community(get_arg("entity", 0, ""))
        elif tool_name == "fuse_and_rank":
            raw = get_arg("chunk_ids", 0, None)
            if isinstance(raw, str):
                raw = [x.strip().strip('"\'') for x in raw.strip("[]").split(",") if x.strip()]
            return self.fuse_and_rank(raw or [], int(get_arg("k", 1, 5)))
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        from groq import Groq
        client = Groq(api_key=self.cfg["groq_api_key"])
        resp = client.chat.completions.create(
            model=self.cfg["llm_model"],
            messages=messages,
            stop=["Observation:"],
            max_tokens=512,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    # ── Main run loop ──────────────────────────────────────────────────────

    def run(self, query: str, max_steps: int = 8) -> dict:
        system = SYSTEM_PROMPT.replace("MAX_STEPS", str(max_steps))
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Question: {query}"},
        ]

        trace = []
        final_answer = None

        for step in range(1, max_steps + 1):
            response = self._call_llm(messages)

            # Parse thought + action
            thought = _extract("Thought", response)
            action_raw = _extract("Action", response)
            final_raw = _extract("Final Answer", response)

            if final_raw:
                final_answer = final_raw
                trace.append({"step": step, "thought": thought,
                               "action": None, "observation": None,
                               "final": True})
                break

            if not action_raw:
                # Forced final answer if no valid action
                final_answer = response.strip()
                break

            # Parse tool name + args
            m = re.match(r"(\w+)\((.*)\)$", action_raw.strip(), re.DOTALL)
            if not m:
                obs = {"error": f"Cannot parse action: {action_raw}"}
                tool_name = "?"
            else:
                tool_name, args_str = m.group(1), m.group(2)
                try:
                    obs = self._dispatch(tool_name, args_str)
                except Exception as e:
                    obs = {"error": str(e)}

            obs_str = json.dumps(obs, ensure_ascii=False)
            trace.append({
                "step": step,
                "thought": thought,
                "action": f"{tool_name}({args_str[:60]})",
                "observation": obs,
            })

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {obs_str[:3000]}"})

        if not final_answer:
            # Force answer from last LLM response
            final_answer = _extract("Final Answer", response) or response.strip()

        # Build attribution: map chunk_ids mentioned in final_answer
        cited_chunks = list(set(re.findall(r"chunk_\w+", final_answer)))
        citations = []
        for cid in cited_chunks:
            chunks = self._neo_store().get_chunks_for_entity("") if False else []
            # pull text directly from pgvector
            try:
                cur = self._pg_store().conn.cursor()
                cur.execute("SELECT doc_id, text FROM chunks WHERE id=%s", (cid,))
                row = cur.fetchone()
                if row:
                    citations.append({
                        "chunk_id": cid,
                        "doc_id": row[0],
                        "excerpt": row[1][:200],
                    })
            except Exception:
                pass

        return {
            "query": query,
            "answer": final_answer,
            "trace": trace,
            "citations": citations,
            "steps": len(trace),
        }

    def close(self):
        if self._pg:
            self._pg.close()
        if self._neo:
            self._neo.close()


# ── Parsing helpers ───────────────────────────────────────────────────────

def _extract(label: str, text: str) -> Optional[str]:
    """Extract content after 'Label:' up to next label or end."""
    pattern = rf"^{label}:\s*(.+?)(?=\n(?:Thought|Action|Observation|Final Answer):|$)"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_args(args_str: str) -> tuple[list, dict]:
    """Parse 'arg1, key=val, ...' into positional list and keyword dict."""
    args_str = args_str.strip()
    if not args_str:
        return [], {}

    pos = []
    kw = {}
    # Simple tokenizer: split on comma, respecting brackets
    depth = 0
    current = ""
    tokens = []
    for ch in args_str + ",":
        if ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            tokens.append(current.strip())
            current = ""
        else:
            current += ch

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok and not tok.startswith('"') and not tok.startswith("'"):
            k, _, v = tok.partition("=")
            k = k.strip()
            v = v.strip().strip('"\'')
            kw[k] = v
        else:
            pos.append(tok.strip('"\''))

    # hoist common first-arg param names to positional
    if not pos and kw:
        for param in ("query", "entity", "entity_name", "chunk_ids", "k"):
            if param in kw:
                pos.append(kw.pop(param))
                break

    return pos, kw
