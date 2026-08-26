# GraphKnows

Agentic memory for LLM applications: it reads a conversation or document stream,
builds a **knowledge graph** of entities, relations and temporal facts over
[ArcadeDB](https://arcadedb.com/), and serves **hybrid retrieval** (vector ANN +
graph traversal + BM25, fused with RRF) back to your agent.

Two extraction modes:

- **`llm_free`** — deterministic extraction (GLiNER2 entities + spaCy relations) and
  deterministic retrieval. No LLM in the ingest or retrieval path; local embeddings by
  default, so it runs with **zero API keys**.
- **`llm_assisted`** — LLM-driven extraction with an **injectable RDF/OWL ontology**, so
  the graph stays domain-specific. Adds `pip install "graphknows[assisted]"`.

Use it two ways from the same core: as a **Python library**, or as an **MCP stdio
server** (Claude Desktop / Claude Code).

## Install

```bash
pip install graphknows              # full llm_free stack + MCP server
pip install "graphknows[assisted]"  # + LLM extraction & injectable ontology
```

| Extra | Adds |
| --- | --- |
| `assisted` | DSPy LLM extraction + injectable RDF/OWL ontology (`llm_assisted` mode) |
| `langgraph` | LangGraph `BaseStore` adapter |

`pip install graphknows` is batteries-included: GLiNER/GLiNER2 extraction, local
embeddings, BM25, community detection, document parsers, and the MCP server are
all base dependencies. The extras above are only framework adapters and the
LLM-assisted mode.

The `graphknows.integrations.client` SDK (drive the server out-of-process) needs no extras.

`pip install graphknows` gives you the library plus two console scripts,
`graphknows-mcp` and `graphknows-topics` — no repo checkout needed for either.

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
> embedder, the reranker, the spaCy pipeline and two NLTK corpora — a few GB).
> To warm them ahead of time from a repo clone/checkout: `python scripts/bake_models.py`
> (the wheel ships only the `graphknows` package, so this script isn't
> available to a plain `pip install`).
> The **Docker images bake all of them at build time**, so a container serves
> requests with no per-call download and works with the network cut.
> See [.env.example](.env.example) and [docs/configuration.md](docs/configuration.md).

## Deploy

```bash
cp .env.example .env          # then set GRAPHKNOWS_ARCADEDB_PASSWORD
docker compose -f docker-compose.prod.yaml up -d
```

`docker-compose.prod.yaml` is the deployment stack: the image is the artifact,
with no repo bind-mount. (`docker-compose.yaml` is the *development* stack — it
mounts the working tree over the image and puts it first on `PYTHONPATH`, which
is convenient while editing and means the container runs live source against
image dependencies.)

CPU by default. For a GPU host:

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
             -t graphknows:cuda .
```

To verify an image is complete — every dependency present, every model baked,
nothing resolved over the network — run `scripts/docker-smoke.sh`. The same
check runs as the container healthcheck (`scripts/bake_models.py --check`).

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
[.env.example](.env.example) and [docs/configuration.md](docs/configuration.md).

## Running the server

```bash
docker compose up -d            # arcadedb + mcp
```

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
[docs/integrations.md](docs/integrations.md) and
[examples/langgraph_agent.py](examples/langgraph_agent.py).


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

Full runnable example: [examples/langgraph_agent.py](examples/langgraph_agent.py).

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
generation, not retrieval. Full per-case report:
[evaluation/results/locomo](evaluation/results/locomo).

## Documentation

Full index: **[docs/](docs/README.md)**.

- [docs/services.md](docs/services.md) — the main services and how they stack
- [docs/architecture.md](docs/architecture.md) — lifecycle states, consolidation, RRF retrieval
- [docs/modes.md](docs/modes.md) — llm_free vs llm_assisted
- [docs/ontology.md](docs/ontology.md) — the bundled CCO ontology, and injecting your own
- [docs/topics.md](docs/topics.md) — the re-runnable topic stage
- [docs/configuration.md](docs/configuration.md) — full settings reference
- [docs/integrations.md](docs/integrations.md) — framework integration recipe
- [docs/multi-tenancy.md](docs/multi-tenancy.md) — namespaces, isolation and scaling
- [docs/versioning.md](docs/versioning.md) — semver & deprecation policy

The `evaluation/` directory is internal LoCoMo/BEAM/LongMemEval benchmarking — not part
of the installed package.

## License

[Apache-2.0](LICENSE).
