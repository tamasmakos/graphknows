# GRAPHGEN-002 — Expose corpus_ingest as MCP Tool (Phase D)

## Context
Working directory: `/root/graphknows`. Confirm this before doing anything.

GraphKnows has a batch corpus pipeline in `graphknows/graphgen/`. The entry point
for agents to trigger it is `graphknows/graphgen/ingest.py::run_ingestion()`:

```python
async def run_ingestion(
    documents: list[dict],          # [{text, title, id, source}, ...]
    config: dict | None = None,
    taxonomy_file: str | None = None,
) -> dict[str, Any]:
```

This runs the full KnowledgePipeline (NER+RE → graph → Neo4j) on a list of documents.
It is NOT currently exposed as an MCP tool — agents cannot trigger batch graph building.

## Task: Add `corpus_ingest` MCP tool to stdio_server.py

**File:** `graphknows/mcp/stdio_server.py`

Add ONE new MCP tool `corpus_ingest` after the existing tools (around line 690, before `main()`).

### Tool signature
```python
@app.tool()
async def corpus_ingest(
    documents: list[dict],
    taxonomy_file: str | None = None,
) -> dict:
    """Ingest a corpus of documents into the long-term knowledge graph.

    Runs the full graphgen pipeline: NER → relation extraction → graph cleaning
    → community detection → Neo4j write-back. Use for batch knowledge ingestion.
    GDS enrichment (Node2Vec + Louvain) runs automatically after write.

    Args:
        documents: List of document dicts. Each must have:
            - text (str): document content
            - title (str): human-readable title
            - id (str): unique document identifier
            - source (str, optional): provenance URL or label
        taxonomy_file: Optional path or name of taxonomy YAML to use.
            Defaults to taxonomy.generic.yaml. Pass "taxonomy.eu-politics.yaml"
            for EU political domain.

    Returns:
        Dict with pipeline statistics: documents_processed, entities_extracted,
        relations_extracted, nodes_written, edges_written, errors (list).
    """
```

### Implementation
```python
    if not documents:
        return {"documents_processed": 0, "error": "no documents provided"}
    try:
        from graphknows.graphgen.ingest import run_ingestion
        config: dict = {}
        if taxonomy_file:
            config["taxonomy_file"] = taxonomy_file
        stats = await run_ingestion(documents=documents, config=config or None)
        return stats
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "documents_processed": 0}
```

### Important: lazy import
The `run_ingestion` import MUST be inside the function body (lazy), NOT at the module top level.
`graphgen` imports heavy optional deps (leidenalg, igraph, node2vec) that are not installed
in the MCP server environment and must never be imported at startup.

## Test to add
In `tests/test_mcp_stdio_server.py`, add a test that:
1. Mocks `graphknows.graphgen.ingest.run_ingestion` to return a dict
2. Calls `corpus_ingest(documents=[{"text": "test", "title": "t", "id": "1"}])`
3. Asserts the mock was called with `documents=[...]`
4. Asserts the result matches the mock return value

Also add a test for the empty documents case (no mock needed — just assert error key present).

## What NOT to touch
- `graphknows/graphgen/ingest.py` — do not modify
- Any other MCP tools in `stdio_server.py`
- Any other test files

## Acceptance Criteria
1. `corpus_ingest` tool is registered in `stdio_server.py` with lazy import.
2. Tool docstring is clear and complete.
3. Two unit tests added to `test_mcp_stdio_server.py`.
4. `uv run pytest tests/ -x -q -m "not integration"` passes — no new failures.
5. `ruff format graphknows/ && ruff check --no-fix graphknows/` — zero new errors.
6. `python3 -B -m py_compile graphknows/mcp/stdio_server.py` — clean.

## Coding Standards
Read before starting:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

## Scope
Only modify `graphknows/mcp/stdio_server.py` and `tests/test_mcp_stdio_server.py`.

## End-of-task summary required
```
Files changed: <list with one-line description>
Files NOT touched: <confirm>
Follow-up needed: <list or "none">
```
