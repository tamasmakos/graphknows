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

1. Create `services/graphgen/src/kg/parser/<format>.py`
2. Subclass `BaseParser` and declare the file extensions:

```python
from kg.parser import BaseParser, ParsedDocument
from pathlib import Path

class MyFormatParser(BaseParser):
    extensions = ("xyz",)          # auto-registers on import

    def parse(self, path: Path) -> ParsedDocument:
        doc = ParsedDocument.from_path(path, "application/x-myformat")
        raw = path.read_text()
        # ... split into chunks ...
        doc.chunks = list(self._chunk(doc.doc_id, raw))
        return doc
```

3. Register the parser by adding it to `kg/parser/registry.py`:

```python
from kg.parser.myformat import MyFormatParser  # noqa: F401
```

4. Add any new dependencies to `services/graphgen/pyproject.toml`.

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

- **Python**: `ruff` + `black`. Run `uv run ruff check . && uv run black .`
- **TypeScript**: `eslint` + `prettier`. Run `pnpm lint`
- **Cypher**: Always use parameterised queries — never f-string interpolation in Cypher.
- **Async**: Use `async with driver.session()` for all Neo4j queries.
- **Lifespan**: Use `@asynccontextmanager lifespan` — never `@app.on_event`.
- **Tests**: Add a test for every new parser/plugin/tool under `tests/`.
