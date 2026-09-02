---
name: the-line
description: "Route a piece of work onto this repo's line of speckit skills and workflows. Use when the user asks which command to run next, how to start a feature, ship an issue, triage the backlog, or fix a stage that rejected their work; when they mention tasks.md, a Verify command, ready-tasks, implement-tasks or triage-issues; and when another skill needs to hand work back to the line."
---

# The line

Work reaches main by one path. Every station is either a **gate** a person owns or a
machine step that runs unattended; nothing in between. `docs/agents/workflows.md` is the
station list. This skill decides which station the work in front of you enters at, and what
to do when one rejects it.

## Where does this work enter?

Match the work, then start at that station. Starting too early is the common waste: eight
commands for a one-line fix.

| The work is | Enter at |
|---|---|
| A backlog nobody has read lately | `/triage-issues`, which hands you groups |
| A group of issues that are one feature | `/speckit-specify`, pasting the group's paragraph |
| A feature you can describe but not yet design | `/speckit-specify` |
| A design already agreed, needing slicing | `/speckit-plan`, then `/ready-tasks` |
| One small defect with an obvious check | Write the task and its Verify by hand, then `/implement-tasks` |
| A change to CI, the gate script, or pre-commit | Your own hands. No script may touch the harness |

Between `/speckit-specify` and `/ready-tasks` sit `/speckit-clarify` and `/speckit-plan`,
which need your answers and your approval. Run them in conversation. A workflow cannot stop
to ask, which is why they are not in one.

## The Verify contract

One command per task, on an indented sub-bullet, that **fails now and passes when the task
is done**.

```markdown
- [ ] T012 [P] [US1] Create User model in graphknows/models/user.py
  - Verify: `pytest -q tests/models/test_user.py`
```

Everything downstream is built on it. `/ready-tasks` runs each one and rejects any that
already passes, because a check that passes describes finished work. `/implement-tasks`
runs it again before starting, uses it to rank candidate patches, and treats it as the
definition of done. Write it as the command would run at `/app`; the scripts add the
`docker exec`.

A Verify naming a test that does not exist yet is correct. Writing that test is part of the
task.

## When a stage rejects your work

Each rejection names a different defect. Fix that one, not the check.

- **"already satisfied"** — the tree does this already. Delete the task, or replace the
  Verify with one that observes the part still missing. A check too weak to see the missing
  work is the usual cause.
- **"not runnable"** — the Verify is prose, malformed, or the workspace container is down.
  `make up` first, then re-read the command as a shell would.
- **"edits the harness"** — the change rewrites what verifies it, so its own green checks
  prove nothing. Land it by hand, on its own branch.
- **"every candidate edits a dependency file"** — a new third-party package is a
  supply-chain decision. Approve it yourself, or find the change that does not need one.
- **"winner passed but was not committed"** — the run stopped there deliberately. A dirty
  tree poisons every later task, since each patch is sampled against HEAD. Clean it, rerun;
  finished tasks are skipped.

## Cost

`/implement-tasks` spends a few agents per task and defaults to three candidate patches
each. Run one story at a time with `{"tasks": ["T012","T013"]}`, and take the first pass at
`{"samples": 1}` to learn whether the tasks are well-formed before paying for candidates.

## What stays yours

Three writes are opt-in in code rather than asked of an agent: applying tracker labels
(`{"apply": true}`), merging (`{"merge": true}`, and only with CI green and the harness
untouched), and the `ready-for-agent` label, which no script ever applies. Reviewing the
draft PR is the fourth, and it is sampled by risk rather than read line by line.
