# Contract: Installable Extras

The public installation surface. Enforced by `tests/test_packaging.py`, which asserts that
`MissingExtraError.KNOWN_EXTRAS` and the extras declared in `pyproject.toml` agree, and that every
`MissingExtraError` call site names a declared extra.

## After this change

| Extra | Contents | Purpose |
|---|---|---|
| *(base)* | GLiNER/GLiNER2, spaCy, sentence-transformers, torch, BM25, nltk, mcp | Working `llm_free` memory service and MCP server, zero API keys |
| `ontology` | `rdflib>=7.0.0`, `networkx>=3.0` | **NEW.** Import and project an RDF/OWL ontology. No LLM stack. |
| `assisted` | `dspy>=2.5.0`, `rdflib>=7.0.0`, `networkx>=3.0` | LLM-driven relation extraction. Reaches the network. |
| `langgraph` | `langgraph>=0.2.0` | Framework adapter. Unsupported. |
| `srl` | `frame-semantic-transformer>=1.1.0` | Transformer frame identification. Unsupported. |
| `observability` | `langfuse>=3.0`, `langchain>=1.0` | Tracing for the example. Unsupported. |

The split is **additive**: `ontology` is a subset of `assisted`, not a slice cut out of it, so
`assisted` keeps `rdflib` and `networkx` and no existing install, image or CI leg loses a dependency
it has today.

An earlier draft of this contract reduced `assisted` to `dspy` alone. That was a preference, not a
requirement — every requirement here (FR-017, SC-006, and guarantee 2 below) is satisfied either
way, and the additive form removes nothing from anyone. Implementation chose it deliberately and
recorded the reason in `pyproject.toml`; this contract follows.

## Guarantees

1. `pip install "graphknows[ontology]"` yields a working RDF import path and installs **no** `dspy`.
2. `pip install "graphknows[assisted]"` still drives LLM-assisted extraction exactly as before. The
   one rdflib import on that path already documents itself as falling back to the manifest's raw
   context string when rdflib is absent, so `assisted` alone stays functional — this is the fact
   that makes a clean split possible rather than requiring `assisted` to depend on `ontology`.
3. A `MissingExtraError` raised from an ontology path names `ontology`. One raised from the
   LLM-extraction path names `assisted`.

## Call sites that change

Five, all currently naming `"assisted"`, all needing `"ontology"`:

| File | Line | Feature named in the error |
|---|---|---|
| `graphknows/symbolic/ontology/rdf_io.py` | 109 | Parsing RDF |
| `graphknows/symbolic/ontology/rdf/taxonomy.py` | 274 | Extracting an RDF taxonomy |
| `graphknows/symbolic/ontology/rdf/projection.py` | 531 | Projecting RDF onto a property graph |
| `graphknows/symbolic/ontology/rdf/prefixes.py` | 170 | Reading vann prefix declarations |
| `graphknows/symbolic/ontology/digest.py` | 324 | Building an ontology digest |

**Unchanged**: `graphknows/ingestion/extraction/relations/llm_assisted/dspy.py:34` keeps
`"assisted"`. It is the real dspy path.

## Must move in one commit

`pyproject.toml` and `graphknows/exceptions.py:36` (`KNOWN_EXTRAS`) are asserted equal by
`tests/test_packaging.py::test_known_extras_constant_matches_pyproject`. Splitting them across
commits leaves the tree red in between.

## Acceptance

A **new node** in the existing file — the file itself already passes, so naming it proves nothing
(research.md R6):

```bash
pytest tests/test_packaging.py::test_ontology_extra_excludes_dspy -q
```

Asserts the **declaration only**: `ontology` is declared, it contains `rdflib` and `networkx`, and
it does not contain `dspy`.

The five call sites are pinned by a *separate* node, `test_ontology_call_sites_name_the_ontology_extra`,
which lands with the call-site change itself. Folding them into one node would make its Verify
unpassable until both edits land, which breaks the one-task-one-runnable-check rule.

## Correction to ship alongside

`pyproject.toml:119` refers to `tests/test_extras.py`, which does not exist — the enforcement lives
in `tests/test_packaging.py`. The stale comment misled three earlier design attempts and should be
corrected in this ticket, since it is the comment describing the very constant being edited.
