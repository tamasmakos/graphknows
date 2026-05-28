# Fix: test_ontology.py import path after repo split

## Context
The graphknows repo was split from a monorepo (2026-05-22). `services/agent/` was moved
to a separate private repo (`tamasmakos/civicalgraph`). The graphknows package is now at
the repo root: `/root/graphknows/graphknows/`.

## Problem
`tests/test_ontology.py` still has stale imports:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "agent"))
from ontology.loader import OntologyLoader, OntologyManifest
from ontology.router import OntologyRouter
```
This fails with `ModuleNotFoundError: No module named 'ontology'` because `services/agent/` no longer exists.

The ontology package now lives at: `graphknows/ontology/` (loader.py, router.py, __init__.py, rel_extractor.py, assets/).

There is also a reference to:
```python
REGISTRY_ROOT = Path(__file__).parent.parent / "services" / "agent" / "ontology" / "assets"
```
The assets are now at: `/root/graphknows/graphknows/ontology/assets/`

## Fix required
1. Remove the `sys.path.insert` line
2. Change imports to:
   ```python
   from graphknows.ontology.loader import OntologyLoader, OntologyManifest
   from graphknows.ontology.router import OntologyRouter
   ```
3. Fix the `REGISTRY_ROOT` path:
   ```python
   REGISTRY_ROOT = Path(__file__).parent.parent / "graphknows" / "ontology" / "assets"
   ```
4. Verify the assets directory actually contains the expected files (ls it first)
5. Run the tests to confirm they pass

## Working directory
`/root/graphknows`

## Files to touch
- `tests/test_ontology.py` — ONLY this file. No other changes.

## Acceptance criteria
```bash
cd /root/graphknows
uv run pytest tests/test_ontology.py -q --tb=short
```
All tests in this file pass (or are skipped with a valid reason). Zero collection errors.

Also confirm no regressions:
```bash
uv run pytest tests/ -q -m 'not integration' --tb=no 2>&1 | tail -5
```
Still 209+ passed.

## Coding standards
- Read `.hermes/instructions/python-style.instructions.md` before making changes
- Surgical scope: only touch `tests/test_ontology.py`
- Do NOT commit anything
