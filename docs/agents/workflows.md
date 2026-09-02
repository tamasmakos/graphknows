# The line: from request to main

Three borrowed shapes, one line. Pre-code is Spec Kit as installed. Coding is the
Agentless loop. Post-merge is the existing CI. Every human gate is named; nothing else
stops to ask.

| Step | Who | Does |
|---|---|---|
| `/speckit-constitution` | human authors | Once. Today `.specify/memory/constitution.md` is the unfilled template; seed it from `CLAUDE.md`. |
| `/speckit-specify` | automatic | `specs/NNN-slug/spec.md` and the feature branch. |
| `/speckit-clarify` | human answers | The cheapest correction point in the whole line. |
| `/speckit-checklist` | human reviews | Optional. Unit tests for the English. |
| `/speckit-plan` | human approves | Approve the architecture before it becomes tasks. |
| `/ready-tasks` | automatic; human reads the report | `/speckit-tasks` plus one rule: every task carries a `Verify:` command. Each is run on the tip and **must fail**; one that passes is rejected. Tasks that edit the harness are gated. Ends with `/speckit-analyze`. |
| `/implement-tasks` | automatic; human samples the diff | Per task, serial: probe (Verify must still fail) → N sampled patches in isolated worktrees → validate (Verify, then regression, lint, types) → commit the winner. Then a draft PR. |
| merge | human | Or `/implement-tasks {merge: true}`: merges only when CI is green **and** no harness file is in the branch diff. Both checks live in the script, not in a prompt. One PR at a time. |
| `ci.yml`, `release.yml` | automatic | Commit, acceptance and release stages. Unchanged. |
| `/speckit-converge` | human runs | Appends unbuilt work as tasks; then `/ready-tasks` and `/implement-tasks` again. |

## The Verify line

```markdown
- [ ] T012 [P] [US1] Create User model in graphknows/models/user.py
  - Verify: `pytest -q tests/models/test_user.py`
```

One backticked command per task, single line. It fails before and passes after; if it
names a test that does not exist, writing the test is part of the task. Python commands
are written as they run at `/app`; the scripts wrap them in
`docker exec graphknows-workspace`. The sub-bullet is invisible to the other speckit
skills, which only read task lines.

## Rules held in code

- **Harness**: `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
  `scripts/gate.sh`, `.pre-commit-config.yaml`. A task or patch that names one is never
  implemented or merged by a script. An agent that can edit its own verification has none.
- **Dependencies**: a patch touching `pyproject.toml` or `uv.lock` is dropped for a person.
- **Merge**: `args.merge === true` and CI green and harness untouched. Never an agent's opinion.
- **Retries**: CI is watched once. No re-run loops.

## Running

`/implement-tasks` needs the checkout on the feature branch with a clean tree; it commits
per task. It is not resumable mid-task: rerun it, and finished (`[X]`) tasks are skipped.
Run it per story with `{ tasks: ['T012', 'T013'] }` to keep a run under ~50 agents;
`{ samples: 1 }` for a cheap first pass. Candidate patches land in
`docs/agents/runs/<feature>/` (gitignored).

The previous generation of scripts is in `.claude/workflows/deprecated/` for reference.
