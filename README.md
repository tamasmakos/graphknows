# GraphKnows

Agentic memory for LLM applications: it reads a conversation or document stream,
builds a **knowledge graph** of entities, relations and temporal facts over
[ArcadeDB](https://arcadedb.com/), and serves **hybrid retrieval** (vector ANN +
graph traversal + BM25, fused with RRF) back to your agent.

Two extraction modes:

- **`llm_free`** — deterministic extraction (GLiNER2 entities + spaCy relations) and
  deterministic retrieval. No LLM in the ingest or retrieval path; local embeddings by
  default, so it runs with **zero API keys**.
- **`llm_assisted`** — the same extraction from **one structured LLM call per chunk**,
  over the same injectable ontology, so the graph stays domain-specific. Needs only an
  API key: `Memory(mode="llm_assisted")`, no extra to install.

Use it two ways from the same core: as a **Python library**, or as an **MCP stdio
server** (Claude Desktop / Claude Code).

## Install

```bash
pip install graphknows              # both modes + MCP server
pip install "graphknows[ontology]"  # + load your own RDF/OWL ontology file
python -m spacy download en_core_web_lg   # required: not on PyPI, so pip cannot pull it
```

spaCy models aren't PyPI packages, so this one can't be declared as a dependency or
an extra — skip it and the first spaCy-dependent call fails with an error naming this
same command.

| Extra | Adds |
| --- | --- |
| `ontology` | Loading your own RDF/OWL ontology file (rdflib + networkx) |
| `langgraph` | LangGraph `BaseStore` adapter |

`pip install graphknows` is batteries-included: GLiNER/GLiNER2 extraction, local
embeddings, BM25, community detection, document parsers, the MCP server **and the
LLM decoder** are all base dependencies — both modes are the supported surface, so
neither sits behind an extra. The extras above are the RDF loader and framework
adapters.

The `graphknows.integrations.client` SDK (drive the server out-of-process) needs no extras.

`pip install graphknows` gives you the library plus the `graphknows-mcp`
console script — no repo checkout needed.

It is not standalone-runnable, though: GraphKnows stores its graph in
[ArcadeDB](https://arcadedb.com/), and every ingest or recall call needs a
reachable server. Point the library at one with `GRAPHKNOWS_ARCADEDB_URL`
(default `http://localhost:2480`) and `GRAPHKNOWS_ARCADEDB_USER` /
`GRAPHKNOWS_ARCADEDB_PASSWORD` (default `root` / `changeme` — refused in
production). Either an ArcadeDB you already have running is fine, or spin
one up from a checkout of this repo:

```bash
docker compose up -d arcadedb
```

> A `pip` install downloads its models on first use (the relex extractor, the
> embedder and the reranker — a few GB). The two NLTK corpora (FrameNet,
> WordNet) are the exception: they are never fetched at runtime, so provision
> them ahead of time — a missing FrameNet corpus raises, a missing WordNet
> corpus only disables sense-anchored expansion.
> To warm everything ahead of time from a repo clone/checkout: `python scripts/bake_models.py`
> (the wheel ships only the `graphknows` package, so this script isn't
> available to a plain `pip install`).
> The dev image downloads the relex extractor, embedder and reranker on first
> use into a compose named volume, so a rebuild never re-downloads gigabytes of
> weights; the NLTK corpora still need baking ahead of time as above.
> `python scripts/bake_models.py` warms the volume. CPU by default; for a GPU host:
> `docker compose build --build-arg TORCH_BACKEND=cu126`.
> See [.env.example](https://github.com/tamasmakos/graphknows/blob/main/.env.example) and [docs/configuration.md](https://github.com/tamasmakos/graphknows/blob/main/docs/configuration.md).

## Quickstart

`Memory` is the single entrypoint — in-process, transport-neutral:

```python
import asyncio
from graphknows import Memory

async def main():
    async with Memory() as mem:
        await mem.ingest_memory("Alex joined Acme as CTO in March.", session_id="s1")
        hits = await mem.recall_memory("Where does Alex work?", session_id="s1")
        print(hits)

asyncio.run(main())
```

Out-of-process (no ML deps in your process), drive the MCP server directly:

```python
from graphknows.integrations.client import GraphKnowsMCPClient

async with GraphKnowsMCPClient(command="graphknows-mcp") as client:
    await client.call_tool(
        "memory_ingest",
        {"text": "Alex joined Acme as CTO in March.", "session_id": "s1"},
    )
    hits = await client.call_tool(
        "memory_query", {"query": "Where does Alex work?", "session_id": "s1"}
    )
```

Configuration is via `GRAPHKNOWS_*` environment variables (or a `.env` file) — see
[.env.example](https://github.com/tamasmakos/graphknows/blob/main/.env.example) and [docs/configuration.md](https://github.com/tamasmakos/graphknows/blob/main/docs/configuration.md).

## Running the server

```bash
docker compose up -d            # arcadedb + mcp
```

> Runs from a repo clone/checkout: the wheel ships only the `graphknows` package, so
> `docker-compose.yaml` isn't available to a plain `pip install`.

**MCP** (Claude Desktop / Claude Code) — register the stdio server:

```json
{
  "mcpServers": {
    "graphknows": {
      "command": "graphknows-mcp",
      "env": { "GRAPHKNOWS_ARCADEDB_URL": "http://localhost:2480" }
    }
  }
}
```

## Integrating into an agent framework

Wire your framework's memory hooks to the `Memory` verbs. The
LangGraph adapter is shipped as the reference; the generic recipe (write →
`ingest_memory`, read → `recall_memory`, thread id → `session_id`, end of
episode → `flush_memory`) covers any framework — see
[docs/integrations.md](https://github.com/tamasmakos/graphknows/blob/main/docs/integrations.md) and
[examples/langgraph_agent.py](https://github.com/tamasmakos/graphknows/blob/main/examples/langgraph_agent.py).

Domain vocabulary is data, not code: a **domain pack** supplies the concepts,
predicates, labels and identity rules of one domain, and `load_packs` merges
them all-or-nothing — see
[docs/packs.md](https://github.com/tamasmakos/graphknows/blob/main/docs/packs.md).
For Claude Code, the shipped hook block maps each event to one verb of `python -m
graphknows.integrations.claude_code`.


### LangGraph

`GraphKnowsMemory` binds a session to ready-to-use LangGraph nodes — three
`add_node` calls instead of hand-wiring `recall`/`remember` with
`functools.partial`. Turns go to the STM buffer as you converse; `flush()` at
end of session ingests them into the knowledge graph so a later session can
recall them.

```python
from graphknows.integrations.langgraph import GraphKnowsMemory

async with GraphKnowsMemory("s1") as mem:
    graph.add_node("recall", mem.recall)
    graph.add_node("remember", mem.remember)  # before respond: both hooks read
    graph.add_node("respond", respond)        # the LAST message, and respond
    ...                                       # appends the assistant's reply
    await app.ainvoke({"messages": [...]})
    await mem.flush()
```

```bash
python examples/langgraph_agent.py --session s1   # teach it something
python examples/langgraph_agent.py --session s2   # recall it in a new session
```

Full runnable example: [examples/langgraph_agent.py](https://github.com/tamasmakos/graphknows/blob/main/examples/langgraph_agent.py).

## Benchmarks

Full [LoCoMo](https://arxiv.org/abs/2402.17753) run — 1540 questions across all 10
conversations, `llm_free` mode (deterministic, zero API keys), ontology injection +
extractive topics on, deterministic retrieval:

| metric | value |
|---|---|
| accuracy (LLM judge) | **0.814** |
| — single_hop (n=841) | 0.860 |
| — knowledge_synthesis (n=282) | 0.798 |
| — temporal (n=321) | 0.741 |
| — open_ended (n=96) | 0.698 |
| evidence recall @ probe | 0.983 |
| evidence recall in generator context | 0.921 |
| gold context coverage | 0.843 |
| cost per question | $0.0026 |

Retrieval delivers the evidence on ~98% of questions; the residual gap is
generation, not retrieval. See
[evaluation/README.md](https://github.com/tamasmakos/graphknows/blob/main/evaluation/README.md)
for how to reproduce.

## Documentation

Full index: **[docs/](https://github.com/tamasmakos/graphknows/blob/main/docs/README.md)**.

- [docs/services.md](https://github.com/tamasmakos/graphknows/blob/main/docs/services.md) — the main services and how they stack
- [docs/architecture.md](https://github.com/tamasmakos/graphknows/blob/main/docs/architecture.md) — lifecycle states, consolidation, RRF retrieval
- [docs/modes.md](https://github.com/tamasmakos/graphknows/blob/main/docs/modes.md) — llm_free vs llm_assisted
- [docs/ontology.md](https://github.com/tamasmakos/graphknows/blob/main/docs/ontology.md) — the bundled CCO ontology, and injecting your own
- [docs/packs.md](https://github.com/tamasmakos/graphknows/blob/main/docs/packs.md) — domain packs: the shipped code and agent vocabularies
- [docs/configuration.md](https://github.com/tamasmakos/graphknows/blob/main/docs/configuration.md) — full settings reference
- [docs/integrations.md](https://github.com/tamasmakos/graphknows/blob/main/docs/integrations.md) — framework integration recipe
- [docs/multi-tenancy.md](https://github.com/tamasmakos/graphknows/blob/main/docs/multi-tenancy.md) — namespaces, isolation and scaling
- [docs/versioning.md](https://github.com/tamasmakos/graphknows/blob/main/docs/versioning.md) — semver & deprecation policy

The `evaluation/` directory is internal LoCoMo/BEAM/LongMemEval benchmarking — not part
of the installed package.

## License

[Apache-2.0](https://github.com/tamasmakos/graphknows/blob/main/LICENSE).
