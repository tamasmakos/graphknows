# Contributing to GraphKnows

Thank you for your interest in contributing! This guide covers the extension points designed for contributors.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Adding a Document Parser](#adding-a-document-parser)
3. [Adding a Graph Plugin](#adding-a-graph-plugin)
4. [Adding an Agent Tool](#adding-an-agent-tool)
5. [Code Style](#code-style)

---

## Development Setup

**Prerequisites**: Docker, Docker Compose v2, Node.js ≥18, pnpm ≥9.

```bash
# 1. Clone
git clone https://github.com/your-org/graphknows.git
cd graphknows

# 2. Environment
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY or OPENAI_API_KEY

# 3. Start
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

Services:
| Service | URL |
|---------|-----|
| Web UI | http://localhost:3000 |
| GraphGen API | http://localhost:8020 |
| GraphRAG API | http://localhost:8010 |
| Neo4j Browser | http://localhost:7474 |

---

## Adding a Document Parser

There is one parser — `PlainTextParser` in
`graphknows/ingestion/parsers/text.py` — and it handles plain text and Markdown,
which is what the ingest path feeds it. It is a plain class, not a registered
plugin: there is no `BaseParser` ABC, no extension registry and no auto-discovery,
because one implementation never needed them.

To support another format, add a function to that module that turns the source
into `TextSegment`s and call it from `parse()`. Reach for a second class only
once a second format actually exists.

---

## Adding a Graph Plugin

Plugins run after the main graph extraction step and can annotate or enrich nodes.

1. Create `services/graphgen/src/kg/plugins/<name>.py`
2. Subclass `GraphPlugin`:

```python
from kg.plugins import GraphPlugin
import networkx as nx

class MyPlugin(GraphPlugin):
    name = "my_plugin"   # used in config to enable/disable

    async def run(self, graph: nx.DiGraph) -> nx.DiGraph:
        for node, data in graph.nodes(data=True):
            if data.get("node_type") == "ENTITY":
                data["my_annotation"] = "..."
        return graph
```

3. The plugin is auto-discovered and available via `get_plugin("my_plugin")`.

---

## Adding an Agent Tool

Tools extend what the RAG agent can do in `services/graphrag/src/agent/tools.py`.

1. Write an `async def my_tool(driver, query, ...) -> list[dict]` function.
2. Wrap it as a `FunctionTool` in `workflow.py`'s `_build_tools()`:

```python
FunctionTool.from_defaults(
    fn=_sync(my_tool, driver=driver, database=database),
    name="my_tool",
    description="What this tool does and when to use it. Args: ...",
),
```

3. Add any new dependencies to `services/graphrag/pyproject.toml`.

---

## Code Style

- **Python**: `ruff` + `ruff format`. Run `uv run ruff check . && uv run ruff format .`
- **TypeScript**: `eslint` + `prettier`. Run `pnpm lint`
- **Cypher**: Always use parameterised queries — never f-string interpolation in Cypher.
- **Async**: Use `async with driver.session()` for all Neo4j queries.
- **Lifespan**: Use `@asynccontextmanager lifespan` — never `@app.on_event`.
- **Tests**: Add a test for every new parser/plugin/tool under `tests/`.
