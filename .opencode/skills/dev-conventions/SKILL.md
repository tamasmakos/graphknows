---
name: dev-conventions
description: Graphknows project layout and development conventions. Load for every coding task in this repo.
---

# Graphknows Dev Conventions

## Repository Layout

```
/root/graphknows/
  services/        <- all production code
  tests/           <- committed, CI-run tests only
  research/        <- GITIGNORED: prototypes, demos, scratch scripts — never committed
  .hermes/instructions/  <- authoritative style + TDD specs
  .opencode/skills/      <- auto-loaded opencode skills
```

## The One Rule

| What | Where |
|---|---|
| Prototype / demo / one-off script | `research/` only — gitignored |
| Production code | `services/<service>/` |
| Committed tests | `tests/` or `services/<service>/tests/` |

**Never create `.py` files at the repo root.** If you find one, move it to `research/` and delete it from git.

## Venvs

- graphknows CLI/package: `/root/graphknows/.venv`
- agent (LangGraph): `/root/graphknows/services/agent/.venv`

Use the correct venv for whichever service you are working in.

## Before Writing Any Code

1. Confirm cwd is `/root/graphknows`
2. Read the python-style and tdd skills (auto-loaded here)
3. Write the failing test first, then the implementation
4. Validate with the correct venv's pytest before declaring done
