# GAA-003: Deterministic tool/terminal summarizer — Brief

## Context
Working directory: `/root/graphknows`
GAA-001 is complete. `graphknows/memory/adapter_types.py` is in place with `ToolCallRecord` and `MemoryClass`.

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
cat /root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md
cat /root/graphknows/graphknows/memory/adapter_types.py
```

## Objective
Create a deterministic tool/terminal execution summarizer. No LLM calls. No raw stdout passthrough as default summary. This produces structured, memory-worthy summaries from tool call data.

## Files to create
- `graphknows/memory/tool_summarizer.py`
- `tests/test_tool_summarizer.py`

## Files to modify
- `graphknows/memory/__init__.py` — export `ToolSummarizer`, `summarize_tool_call`

## Core design

### `ToolCategory` enum
```python
class ToolCategory(str, Enum):
    shell = "shell"           # terminal, bash, exec
    file_read = "file_read"   # read_file, cat
    file_write = "file_write" # write_file, patch
    web = "web"               # web_search, fetch, browser
    quality = "quality"       # pytest, mypy, ruff, bandit
    graph = "graph"           # neo4j, memgraph, cypher
    memory = "memory"         # recall, ingest, flush
    other = "other"
```

### `ToolSummary` dataclass (frozen)
```python
@dataclass(frozen=True)
class ToolSummary:
    tool_name: str
    category: ToolCategory
    outcome: str          # "success" | "error" | "warning" | "partial"
    signal: str           # First meaningful signal line (not raw stdout)
    artifact_paths: list[str]   # Any file paths touched/produced
    error_class: str | None     # e.g. "ImportError", "AssertionError", "CalledProcessError"
    is_durable_fact: bool       # True if the summary describes a stable fact worth promoting
    memory_class: MemoryClass   # Always raw_evidence for v1
```

### `ToolSummarizer` class
```python
class ToolSummarizer:
    """Deterministic tool/terminal execution summarizer.

    Converts raw tool call data into structured, memory-worthy evidence
    records. No LLM calls. No raw stdout passthrough.
    """

    _QUALITY_TOOLS: frozenset[str] = frozenset(["pytest", "mypy", "ruff", "bandit", "coverage"])
    _SHELL_TOOLS: frozenset[str] = frozenset(["terminal", "bash", "shell", "exec", "run"])
    _FILE_READ_TOOLS: frozenset[str] = frozenset(["read_file", "cat", "view"])
    _FILE_WRITE_TOOLS: frozenset[str] = frozenset(["write_file", "patch", "apply_patch"])
    _WEB_TOOLS: frozenset[str] = frozenset(["web_search", "fetch", "browser_navigate", "webfetch"])
    _GRAPH_TOOLS: frozenset[str] = frozenset(["neo4j_query", "memgraph_query", "cypher"])
    _MEMORY_TOOLS: frozenset[str] = frozenset(["recall", "ingest", "flush", "graphknows_recall"])

    def categorize(self, tool_name: str) -> ToolCategory:
        """Return the ToolCategory for a given tool name."""
        ...

    def summarize(
        self,
        tool_name: str,
        input_text: str,
        output_text: str,
        exit_code: int | None = None,
        error_text: str | None = None,
    ) -> ToolSummary:
        """Produce a structured ToolSummary from raw tool call data.

        Args:
            tool_name: Name of the tool that was called.
            input_text: Human-readable input summary (not full args dump).
            output_text: Raw output text (will be summarized, not passed through).
            exit_code: Shell exit code for shell tools; None otherwise.
            error_text: First error line or exception type; None on success.

        Returns:
            ToolSummary with structured fields.
        """
        ...
```

### `summarize_tool_call(record: ToolCallRecord) -> ToolSummary` (module-level function)
Convenience wrapper: `ToolSummarizer().summarize(record.tool_name, record.input_summary, record.output_summary, record.exit_code, record.error)`.

## Summarization logic (deterministic rules)

### Outcome
- `exit_code is not None and exit_code != 0` → `"error"`
- `error_text is not None` → `"error"`
- `"warning" in output_text.lower()[:500]` → `"warning"`
- else → `"success"`

### Signal extraction (first meaningful signal)
- Quality tools (pytest/mypy/ruff/bandit): extract the LAST summary line — e.g. `"27 passed in 0.17s"`, `"Found 3 errors"`, `"All checks passed"`
- Shell tools: first non-empty non-path line, max 120 chars
- File read: `f"Read {path}" ` where path is extracted from input_text
- File write: `f"Wrote {path}"` where path is extracted from input_text
- Web: first sentence of output, max 120 chars
- Other: first 120 chars of output_text stripped of whitespace

### Artifact paths
- Extract any strings matching `r'[/~]\S+\.\w+'` from input_text + output_text
- Deduplicate, keep max 10

### Error class
- If error_text is not None: extract first word that ends in "Error" or "Exception", e.g. `"ImportError"`, `"AssertionError"`
- If not found but exit_code != 0: `"NonZeroExit"`
- Else: None

### is_durable_fact
- True if: quality tool AND outcome == "success"
- True if: file write AND outcome == "success"
- True if: graph tool AND outcome == "success" AND output_text contains "nodes" or "relationships"
- False in all other cases

## Tests to write (tests/test_tool_summarizer.py)
Required test cases:
- `test_categorize_pytest` — `categorize("pytest")` returns `ToolCategory.quality`
- `test_categorize_terminal` — `categorize("terminal")` returns `ToolCategory.shell`
- `test_categorize_read_file` — `categorize("read_file")` returns `ToolCategory.file_read`
- `test_categorize_unknown` — `categorize("my_custom_tool")` returns `ToolCategory.other`
- `test_summarize_pytest_success` — exit_code=0, output includes "27 passed in 0.17s" → outcome="success", signal contains "27 passed", is_durable_fact=True
- `test_summarize_pytest_failure` — exit_code=1, output includes "3 failed in 0.5s" → outcome="error", signal contains "3 failed", is_durable_fact=False
- `test_summarize_terminal_success` — shell tool, exit_code=0 → outcome="success"
- `test_summarize_terminal_error` — exit_code=1, error_text="CalledProcessError" → outcome="error", error_class="CalledProcessError"
- `test_summarize_file_write_success` — write_file, input includes "/tmp/test.py" → artifact_paths contains "/tmp/test.py", is_durable_fact=True
- `test_summarize_traceback` — error_text includes "ImportError: No module named X" → error_class="ImportError"
- `test_summarize_nonzero_exit_no_error_text` — exit_code=2, no error_text → error_class="NonZeroExit"
- `test_summarize_tool_call_convenience_fn` — call `summarize_tool_call(ToolCallRecord(...))` and check it returns a `ToolSummary`
- `test_summary_is_frozen` — verify `ToolSummary` raises `FrozenInstanceError` on mutation
- `test_no_raw_stdout_in_signal` — signal field is max 120 chars (not full stdout passthrough)

## Acceptance criteria
- `ToolSummarizer` and `summarize_tool_call` importable from `graphknows.memory`
- `uv run pytest tests/test_tool_summarizer.py -v` — all tests pass
- `ruff format graphknows/ tests/ && ruff check --no-fix graphknows/ tests/` — no new errors in new files
- No LLM calls anywhere in the module
- Signal field is always ≤ 120 chars

## End-of-task report required
```
Files changed: <list>
What was modified: <one line per file>
Files not touched: <list>
Test result: <pytest summary line>
Follow-up: <any gaps>
```

## Do NOT
- Add LLM calls of any kind
- Modify any existing memory layer files except `__init__.py`
- Import from `hermes`, `langchain`, `openai`, or any host runtime
- Pass raw stdout through as the summary signal
- Commit or push anything
