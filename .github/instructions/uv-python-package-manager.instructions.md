---
description: "Use when installing packages, managing dependencies, running Python scripts, setting up virtual environments, or working with pyproject.toml and requirements files. Covers uv commands, workspace management, and migration from pip."
applyTo: "pyproject.toml, **/pyproject.toml, requirements*.txt, **/requirements*.txt"
---
# Python Package Management with uv

This project uses [uv](https://docs.astral.sh/uv/) exclusively. Never suggest or run `pip`, `pip install`, `python -m pip`, `virtualenv`, or `poetry` commands.

## Command Substitutions

| Instead of | Use |
|---|---|
| `pip install <pkg>` | `uv add <pkg>` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip uninstall <pkg>` | `uv remove <pkg>` |
| `python -m venv .venv` | `uv venv` |
| `python script.py` | `uv run script.py` |
| `pip install -e .` | `uv sync` (editable install is the default) |
| `pip install --upgrade <pkg>` | `uv add <pkg> --upgrade-package <pkg>` |

## Workspace Structure

The root `pyproject.toml` defines a `uv` workspace with two members:

```
/workspace/              ← workspace root (dev tools: ruff, pytest)
  services/graphgen/     ← graphgen member
  services/graphrag/     ← graphrag member
```

- Run workspace-wide commands from `/workspace/`
- Run service-specific commands from the service directory, or use `--package <name>`

## Common Workflows

**Sync all workspace dependencies:**
```bash
uv sync
```

**Add a dependency to a specific service:**
```bash
uv add <pkg> --package graphgen
# or from inside the service directory:
cd services/graphgen && uv add <pkg>
```

**Add a dev dependency (workspace root):**
```bash
uv add --dev <pkg>
```

**Run a script or module:**
```bash
uv run python -m pytest
uv run uvicorn src.main:app --reload
```

**Run a one-off tool without installing:**
```bash
uvx ruff check .
uvx mypy .
```

**Lock and export:**
```bash
uv lock                          # update uv.lock
uv export --format requirements-txt > requirements.txt
```

## Rules

- Always use `uv add` to add packages so `pyproject.toml` and `uv.lock` stay in sync.
- Never manually edit `uv.lock`; it is auto-managed.
- The virtual environment lives at `.venv/` in the workspace root — do not create additional venvs.
- When suggesting shell commands in responses, use `uv run <cmd>` rather than activating the venv manually.
