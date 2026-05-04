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
import xml.etree.ElementTree as ET
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
  Use for: exploring the full membership of a thematic cluster.

wikipedia_search(query: str, sentences: int = 5)
  Search Wikipedia for background context on a topic. Free, no API key.
  Use when: entity/topic not found in KG, need definitional background, need historical context.
  Output: [{title, url, summary}]

web_search(query: str, max_results: int = 5)
  Search the web via DuckDuckGo for public records, news, and official documents.
  Free, no API key. Prioritise with 'site:europa.eu OR site:parliament.uk' suffix for EU/UK docs.
  Use when: KG has fewer than 3 relevant results, need recent events, need source URLs for citations.
  Output: [{title, url, body}]""".strip()

SYSTEM_PROMPT = f"""You are a political intelligence analyst operating a knowledge graph of EU Parliament debates.
You answer questions by reasoning step by step, using tools to retrieve evidence from the graph.

{TOOL_DESCRIPTIONS}

Format EVERY response as:
Thought: <reasoning about what to do next>
Action: <tool_name>(<args>)

When you have gathered enough evidence, output:
Thought: I have sufficient evidence to answer.
Final Answer: <your answer>
Cite sources as [Source: Institution, Date, URL] — NOT as [chunk_id].
For web results: [Source: Title, URL].
For Wikipedia: [Wikipedia: Title, URL].
Never expose internal chunk IDs or entity variable names to the user.

Rules:
- Always cite human-readable source metadata in your final answer, never internal chunk IDs
- Use fuse_and_rank before giving a final answer when you have multiple chunk candidates
- If search_entities returns results, follow up with get_neighbours on the top entity
- REQUIRED STEP ORDER:
  1. search_entities or search_chunks first
  2. get_neighbours on top entity
  3. get_triplets on top entity (REQUIRED after get_neighbours)
  4. IF top chunk score from search_chunks < 0.65: IMMEDIATELY call web_search
     with query + 'site:europa.eu OR site:parliament.uk', then wikipedia_search
  5. get_community on a relevant entity
  6. fuse_and_rank on all collected chunk_ids
  7. Final Answer
- web_search and wikipedia_search MUST be called before fuse_and_rank when scores are weak
- NEVER skip web_search when the KG does not have direct evidence for the question
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

    # ── Metadata cache helpers ─────────────────────────────────────────────

    def _fetch_chunk_meta(self, chunk_id: str, meta_cache: dict) -> dict:
        """Fetch and cache Postgres metadata for a single chunk_id.

        Args:
            chunk_id: The chunk identifier to look up.
            meta_cache: Mutable dict mapping chunk_id -> metadata dict.

        Returns:
            The metadata dict (also stored in meta_cache under chunk_id).
        """
        if chunk_id in meta_cache:
            return meta_cache[chunk_id]
        try:
            pg = self._pg_store()
            try:
                pg.conn.rollback()
            except Exception:
                pass
            cur = pg.conn.cursor()
            cur.execute("SELECT doc_id FROM chunks WHERE id = %s", (chunk_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                doc_id: str = row[0] or ""
                if doc_id.startswith("ep_debate_"):
                    parts = doc_id.split("_")
                    meta_cache[chunk_id] = {
                        "institution": "European Parliament",
                        "url": "https://europarl.europa.eu/plenary/en/debates.html",
                        "date": "-".join(parts[2:5]) if len(parts) >= 5 else "",
                        "doc_id": doc_id,
                    }
                else:
                    meta_cache[chunk_id] = {
                        "institution": "", "url": "", "date": "", "doc_id": doc_id,
                    }
            else:
                meta_cache[chunk_id] = {"institution": "Unknown", "url": "", "date": "", "doc_id": ""}
        except Exception as exc:
            log.warning("meta fetch failed for %s: %s", chunk_id, exc)
            meta_cache[chunk_id] = {"institution": "Unknown", "url": "", "date": "", "doc_id": ""}
        return meta_cache[chunk_id]

    # ── Observation XML formatter ──────────────────────────────────────────

    def _format_obs_xml(self, tool_name: str, obs: Any, meta_cache: dict) -> str:
        """Format a tool observation as XML with full source metadata.

        Populates meta_cache as a side-effect for any new chunk_ids seen.

        Args:
            tool_name: The tool that produced this observation.
            obs: The raw observation dict or list.
            meta_cache: Cache of chunk_id -> {doc_id, institution, date, url}.

        Returns:
            XML string for LLM context.
        """
        root = ET.Element("observation", tool=tool_name)
        if tool_name in ("search_chunks", "fuse_and_rank", "get_chunks_for_entity"):
            for item in (obs if isinstance(obs, list) else [obs]):
                if not isinstance(item, dict):
                    continue
                cid = item.get("chunk_id", "")
                if cid:
                    self._fetch_chunk_meta(cid, meta_cache)
                meta = meta_cache.get(cid, {})
                chunk_el = ET.SubElement(
                    root, "chunk",
                    id=cid,
                    institution=meta.get("institution", "Unknown"),
                    date=meta.get("date", ""),
                    url=meta.get("url", ""),
                    score=str(item.get("score", item.get("rrf_score", ""))),
                )
                ET.SubElement(chunk_el, "text").text = item.get("text", "")[:400]
        elif tool_name in ("search_entities", "get_neighbours"):
            for item in (obs if isinstance(obs, list) else [obs]):
                if not isinstance(item, dict):
                    continue
                ET.SubElement(
                    root, "entity",
                    name=item.get("name", ""),
                    label=item.get("label", ""),
                    score=str(item.get("score", item.get("shared_chunks", ""))),
                )
        elif tool_name == "web_search":
            for item in (obs if isinstance(obs, list) else [obs]):
                if not isinstance(item, dict):
                    continue
                src = ET.SubElement(root, "result",
                                    title=item.get("title", ""), url=item.get("url", ""))
                ET.SubElement(src, "body").text = item.get("body", "")[:300]
        elif tool_name == "wikipedia_search":
            for item in (obs if isinstance(obs, list) else [obs]):
                if not isinstance(item, dict):
                    continue
                src = ET.SubElement(root, "article",
                                    title=item.get("title", ""), url=item.get("url", ""))
                ET.SubElement(src, "summary").text = item.get("summary", "")[:400]
        else:
            ET.SubElement(root, "data").text = json.dumps(obs)[:500]
        return ET.tostring(root, encoding="unicode")

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
    def wikipedia_search(self, query: str, sentences: int = 5) -> list[dict]:
        """Search Wikipedia for background context on a topic.

        Free tool — no API key required. Uses the wikipedia-api package.
        Returns article summaries to supplement KG knowledge.

        Args:
            query: Search query string.
            sentences: Number of summary sentences to return per article.

        Returns:
            List of {title, url, summary} dicts (up to 3 results).
        """
        try:
            import wikipediaapi  # type: ignore[import]
            wiki = wikipediaapi.Wikipedia(
                language="en",
                user_agent="graphknows-agent/1.0 (political-intelligence-research)",
            )
            page = wiki.page(query)
            if page.exists():
                summary = " ".join(page.summary.split(".")[:sentences]) + "."
                return [{"title": page.title, "url": page.fullurl, "summary": summary}]
            # Try search if exact page not found
            import wikipedia  # type: ignore[import]
            results = wikipedia.search(query, results=3)
            out = []
            for title in results[:3]:
                try:
                    p = wiki.page(title)
                    if p.exists():
                        s = " ".join(p.summary.split(".")[:sentences]) + "."
                        out.append({"title": p.title, "url": p.fullurl, "summary": s})
                except Exception:
                    continue
            return out
        except Exception as exc:
            log.warning("wikipedia_search failed for %r: %s", query, exc)
            return [{"error": str(exc)}]

    def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web via DuckDuckGo for public records and news.

        Free tool — no API key required. Uses the duckduckgo-search package.
        Prioritises .europa.eu, .parliament.uk, .un.org, .gov domains.

        Args:
            query: Search query string. Append 'site:europa.eu OR site:parliament.uk'
                   for EU/UK public record focus.
            max_results: Maximum number of results to return.

        Returns:
            List of {title, url, body} dicts.
        """
        try:
            from duckduckgo_search import DDGS  # type: ignore[import]
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": r.get("body", "")[:400],
                    })
            return results
        except Exception as exc:
            log.warning("web_search failed for %r: %s", query, exc)
            return [{"error": str(exc)}]

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
        elif tool_name == "wikipedia_search":
            return self.wikipedia_search(
                get_arg("query", 0, ""),
                int(get_arg("sentences", 1, 5)),
            )
        elif tool_name == "web_search":
            return self.web_search(
                get_arg("query", 0, ""),
                int(get_arg("max_results", 1, 5)),
            )
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

    def run(self, query: str, max_steps: int = 12) -> dict:
        system = SYSTEM_PROMPT.replace("MAX_STEPS", str(max_steps))
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Question: {query}"},
        ]

        trace = []
        final_answer = None
        meta_cache: dict[str, dict] = {}
        hint_injected = False

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
            xml_obs = self._format_obs_xml(tool_name, obs, meta_cache)

            # Detect weak KG scores after search_chunks and inject a mandatory hint
            if tool_name == "search_chunks" and isinstance(obs, list) and obs and not hint_injected:
                top_score = obs[0].get("score", 1.0) if isinstance(obs[0], dict) else 1.0
                if top_score < 0.65:
                    hint_injected = True
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation:\n{xml_obs}\n"
                            f"[SYSTEM HINT: Top chunk score is {top_score:.2f} which is below "
                            f"threshold 0.65. You MUST now call web_search to supplement KG evidence.]"
                        ),
                    })
                    continue

            messages.append({"role": "user", "content": f"Observation:\n{xml_obs}"})

        if not final_answer:
            # Force answer from last LLM response
            final_answer = _extract("Final Answer", response) or response.strip()

        final_answer = _clean_citations(final_answer, meta_cache)

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

def _clean_citations(answer: str, meta_cache: dict) -> str:
    """Replace internal [chunk_N] references with human-readable source citations.

    Args:
        answer: The raw final answer string from the LLM.
        meta_cache: Cache of chunk_id -> {institution, date, url} metadata.

    Returns:
        Answer string with chunk IDs replaced by formatted citation links.
    """
    def replace_chunk(m: re.Match) -> str:
        cid = m.group(0)  # e.g. 'chunk_7'
        meta = meta_cache.get(cid, {})
        inst = meta.get("institution", "Source") or "Source"
        date = meta.get("date", "")
        url = meta.get("url", "")
        if url:
            label = f"{inst}, {date}" if date else inst
            return f"[{label}]({url})"
        return f"[{inst}, {date}]" if date else f"[{inst}]"

    return re.sub(r"chunk_\w+", replace_chunk, answer)


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
