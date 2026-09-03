# The line: from request to main

Three borrowed shapes, one line. Pre-code is Spec Kit's skills, run as one workflow. Coding is
the Agentless loop. Post-merge is the existing CI. Every human gate is named; nothing else
stops to ask.

Everything the line runs on lives under `.claude/`, and none of it is tracked in git -- the
whole folder is gitignored. Portability is by copying it into another repository, not by
cloning: the speckit skills with their templates beside them, the three workflow scripts, and
`.claude/constitution.md`. No Spec Kit CLI, no PowerShell helpers, no hook file. Specs are
written to `.claude/specs/<NNN-slug>/`, and stay untracked until `/specify {github: true}`
force-adds that one feature directory.

`/implement-tasks` samples in a `git worktree add` checkout of HEAD, which only contains
committed content, so it reads spec.md and plan.md from the main checkout by path rather than
expecting its own worktree to have them.

| Step | Who | Does |
|---|---|---|
| `/triage-issues` | automatic; human reads the report | Optional, for an existing backlog. One screen per open issue against the current tree; labels decided in code and applied only with `{apply: true}`; `ready-for-agent` is never applied by a script. Returns candidate specs (`/specify {issues: [n, m]}`), single-task fixes, and issues to close by hand. |
| `/speckit-constitution` | human authors | Once. `.claude/constitution.md` holds the ratified principles every stage is measured against. |
| `/specify <description>` | automatic; human reads spec.md | One run: spec → clarify → plan → tasks → probe → analyze. Creates `.claude/specs/NNN-slug/` and the feature branch. Clarify cannot ask you, so it answers each question with the option it would recommend and records it under `## Clarifications` marked `auto` — read those and overrule any you disagree with. Every task carries a `Verify:` command that is run on the tip and **must fail**; one that passes is rejected. Tasks that edit the harness are gated. Also `{issues: [n, m]}` to spec GitHub issues, and `{feature: "NNN-slug"}` to re-enter at Tasks after `/speckit-converge`. |
| `/specify {github: true, feature: "NNN-slug"}` | human opts in | Publishes: commits the spec directory, pushes the branch, files one GitHub issue per surviving task in a milestone named after the feature directory, the Verify in the body. Without it nothing leaves the machine. |
| `/speckit-clarify`, `/speckit-checklist` | human runs | Optional, by hand. Interactive clarification when the auto answers are not good enough; unit tests for the English. |
| `/implement-tasks` | automatic; human samples the diff | Roster: open tasks in tasks.md. If the feature milestone has issues, only tasks with an **open issue** run: close the issue and the task is skipped. If it has none (a local spec), tasks.md alone decides. Per task, serial: probe (Verify must still fail) → N sampled patches in isolated worktrees, each asking the knowledge graph where the task lands first → validate (Verify, then regression, lint, types) → commit the winner. Then `graphify --update` refreshes the graph from the commits, and a draft PR is opened whose body closes every finished task's issue. |
| merge | human | Or `/implement-tasks {merge: true}`: merges only when CI is green **and** no harness file is in the branch diff. Both checks live in the script, not in a prompt. One PR at a time. Merging is what closes the task issues. |
| `ci.yml`, `release.yml` | automatic | Commit, acceptance and release stages. Unchanged. |
| `/speckit-converge` | human runs | Appends unbuilt work as tasks; then `/specify {feature}` and `/implement-tasks` again. |

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

- **Feature directory**: `.claude/specs/<NNN-slug>/`, where the basename is also the branch and
  the GitHub milestone. Every skill finds it by the current branch name; there is no pointer file.
- **Tracker**: GitHub is where a published task is opened, deferred or killed; tasks.md holds
  the Verify and the per-branch `[X]`. `/implement-tasks` only works a task whose issue is
  still open once issues exist, so pulling a task is closing its issue, with nothing to edit.
  `/speckit-converge` appends tasks to the file; the next `/specify {feature}` probes the new
  ones and, with `{github: true}`, files them, deduplicated by task id.
- **Harness**: `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
  `scripts/gate.sh`, `.pre-commit-config.yaml`. A task or patch that names one is never
  implemented or merged by a script. An agent that can edit its own verification has none.
- **Dependencies**: a patch touching `pyproject.toml` or `uv.lock` is dropped for a person.
- **Publishing**: `{github: true}` is the only thing that commits a spec or writes an issue.
- **Merge**: `args.merge === true` and CI green and harness untouched. Never an agent's opinion.
- **Retries**: CI is watched once. No re-run loops.

## Running

`/implement-tasks` needs the checkout on the feature branch with a clean tree; it commits
per task. It is not resumable mid-task: rerun it, and finished (`[X]`) tasks are skipped.
Run it per story with `{ tasks: ['T012', 'T013'] }` to keep a run under ~50 agents;
`{ samples: 1 }` for a cheap first pass. Candidate patches land in
`docs/agents/runs/<feature>/` (gitignored). The graph stages are skipped when
`graphify-out/graph.json` is absent; `/graphify .` builds it once.

There is no second way to implement a task, triage an issue or write a spec; a command
that is not in the table above was removed because it duplicated one that is.
