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

get_triplets(entity: str, limit: int = 10)
  Returns CO_OCCURS triplets for an entity: (entity_a, CO_OCCURS, entity_b, weight, via_chunks).
  Use for: discovering direct knowledge-graph relationships after get_neighbours.

get_source_document(chunk_id: str)
  Returns full provenance for a chunk: id, doc_id, text, institution, url, date.
  Use for: resolving attribution and verifying source metadata for a cited chunk.

get_topic_entities(topic_id: str)
  Returns all entities belonging to a Leiden topic/community cluster.
  Use for: exploring the full membership of a thematic cluster.""".strip()

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
- After get_neighbours, call get_triplets on the top entity to surface direct KG relationships
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

    def get_triplets(self, entity: str, limit: int = 10) -> list[dict]:
        """Return CO_OCCURS triplets for an entity from the Neo4j knowledge graph.

        Args:
            entity: The entity name to look up.
            limit: Maximum number of triplets to return.

        Returns:
            List of dicts with keys entity_a, relation, entity_b, weight, via_chunks.
        """
        rows = self._neo_store()._run(
            """
            MATCH (a:Entity {name: $name})-[r:CO_OCCURS]->(b:Entity)
            RETURN a.name AS entity_a, 'CO_OCCURS' AS relation,
                   b.name AS entity_b, r.weight AS weight, r.via_chunks AS via_chunks
            ORDER BY r.weight DESC
            LIMIT $limit
            """,
            name=entity,
            limit=limit,
        )
        return [
            {
                "entity_a": row["entity_a"],
                "relation": row["relation"],
                "entity_b": row["entity_b"],
                "weight": row["weight"],
                "via_chunks": row["via_chunks"] or [],
            }
            for row in rows
        ]

    def get_source_document(self, chunk_id: str) -> dict:
        """Return full provenance metadata for a chunk from the Postgres store.

        Args:
            chunk_id: The chunk identifier to look up.

        Returns:
            Dict with keys id, doc_id, text, institution, url, date.
        """
        pg = self._pg_store()
        try:
            pg.conn.rollback()
        except Exception:
            pass
        cur = pg.conn.cursor()
        cur.execute("SELECT id, doc_id, text FROM chunks WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return {"error": f"Chunk not found: {chunk_id}"}
        chunk_id_val: str = row[0]
        doc_id: str = row[1] or ""
        text: str = row[2] or ""
        # Infer institution/url/date from doc_id
        if doc_id.startswith("ep_debate_"):
            institution = "European Parliament"
            url = "https://europarl.europa.eu/plenary/en/debates.html"
            # Extract date portion: ep_debate_YYYY_MM_DD
            parts = doc_id.split("_")
            date = "-".join(parts[2:5]) if len(parts) >= 5 else ""
        else:
            institution = ""
            url = ""
            date = ""
        return {
            "id": chunk_id_val,
            "doc_id": doc_id,
            "text": text[:600],
            "institution": institution,
            "url": url,
            "date": date,
        }

    def get_topic_entities(self, topic_id: str) -> list[dict]:
        """Return all entities belonging to a Leiden topic/community cluster.

        Args:
            topic_id: The topic/community identifier (string or numeric id).

        Returns:
            List of dicts with keys name, label, community_id.
        """
        rows = self._neo_store()._run(
            """
            MATCH (t:Topic)-[:IN_COMMUNITY]-(e:Entity)
            WHERE t.id = $tid OR t.community_id = $tid
            RETURN e.name AS name, e.label AS label, t.id AS community_id
            ORDER BY e.name
            """,
            tid=topic_id,
        )
        return [
            {
                "name": row["name"],
                "label": row["label"] or "Entity",
                "community_id": row["community_id"],
            }
            for row in rows
        ]

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
        elif tool_name == "get_triplets":
            return self.get_triplets(
                get_arg("entity", 0, ""),
                int(get_arg("limit", 1, 10)),
            )
        elif tool_name == "get_source_document":
            return self.get_source_document(get_arg("chunk_id", 0, ""))
        elif tool_name == "get_topic_entities":
            return self.get_topic_entities(get_arg("topic_id", 0, ""))
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
            try:
                pg = self._pg_store()
                # Use a fresh cursor; reset transaction state if needed
                if pg.conn.closed:
                    from graph_store import PgStore
                    self._pg = PgStore(self.cfg["pgvector_dsn"])
                    pg = self._pg
                try:
                    pg.conn.rollback()  # clear any aborted transaction
                except Exception:
                    pass
                cur = pg.conn.cursor()
                cur.execute("SELECT doc_id, text FROM chunks WHERE id = %s", (cid,))
                row = cur.fetchone()
                cur.close()
                if row:
                    doc_id: str = row[0]
                    excerpt: str = row[1][:300]
                    # Infer URL from doc_id
                    if doc_id.startswith("ep_debate_"):
                        url = "https://europarl.europa.eu/plenary/en/debates.html"
                    else:
                        url = ""
                    citations.append({
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "excerpt": excerpt,
                        "url": url,
                    })
                    log.debug("Citation resolved: %s → doc_id=%s", cid, doc_id)
                else:
                    log.warning("Citation chunk not found in pgvector: %s", cid)
            except Exception as exc:
                log.warning("Citation lookup failed for %s: %s", cid, exc)

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
