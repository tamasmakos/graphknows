---
name: merge-train
description: Launch and recover a merge train — the human-controlled front end of the run-merge-train workflow, which drains one spec issue's approved tickets onto main serially.
argument-hint: "[spec issue number]"
disable-model-invocation: true
---

The drain itself is a workflow now: `.claude/workflows/run-merge-train.js` rosters a spec's sub-issue tickets, admits only the ones a human approved, orders them by their `## Blocked by` edges, and lands them one at a time — each car cut from `main` as it stands after the previous merge, so sibling conflicts cannot happen. This skill is the part that must stay human: deciding to launch, tuning the order when taste matters, and cleaning up after a park or a halt.

## Launch

The train needs a spec issue number (prepare-tickets files one per request, with the tickets attached as sub-issues) and the explicit word that authorizes writing to main:

```
Workflow({ name: 'run-merge-train', args: { spec: <n>, merge: true, runLabel: '<label>' } })
```

Before launching, confirm the approvals are real: every ticket that should board carries `ready-for-agent` and not `needs-triage`. The workflow enforces both — a ticket without explicit approval is refused, silence is not consent — so an unlabeled backlog produces an empty train, not a surprise. `force` is refused outright; to override the gate on one issue, a human runs `process-issues` on it directly.

**Two lanes.** Cars worth a few lines of work (N ≤ 30, at most two files, nothing blocked by another car) ride the **express lane** first: one branch, one commit per ticket, one gate run, one PR, one CI run, one rebase-merge. Everything else drains one car at a time as before.

The roster **sizes every candidate**. It takes an `Estimated diff: ~N lines` line when the body has one, and otherwise reads the files the ticket names and counts, reporting the symbols it read as evidence. A ticket it cannot size either way takes the long road, so sizing fails closed. Wrong-but-measured is cheap: the lane ejects anything over 50 real lines before it commits anything. The run log now names every car that missed the lane and why — that is how you catch a whole train taking the long road because nothing carried an estimate.

That exists because the normal lane's cost is almost entirely per-PR waiting, none of which shrinks with the diff. Measured 2026-08-28: ~4.5 min of pre-push gate, ~6.5 min of PR CI, ~11 of post-merge main watch. Those are the only numbers here worth trusting, and only until the suite changes — a stale "3-4 minutes" in the push budget is what parked two cars of the #135 train after they had done all the work. What the lane buys with that is one review of the batch instead of one per ticket, which is only a fair trade while the changes really are tiny — so a ticket that turns out bigger than its estimate is **ejected mid-batch, commits nothing, and is re-queued as a full car**. Watch for that in the run log; it is the sign an estimate was wrong, not that anything broke.

`{ express: false }` sends everything the long way, which is what to reach for if a batch has just landed something you did not expect.

**Ordering by taste.** The workflow orders by blocking edges, smallest first within a tier. It does not know that one car moves a seam the others build on, or that two cars touching the same area should ride adjacently (the second inherits the first's merged code). When that matters, don't run the train — run `process-issues` per issue in the order you chose, `{ issues: [<n>], merge: true }` per car, and consult `graph-metrics` on the files each ticket names to find the shape-changer.

**Mid-train control.** There is no pause button; the brakes are labels. Re-applying `needs-triage` to a ticket parks it at its dispatch check — the workflow re-reads labels before every car.

## Recover

The workflow parks, halts, and reports; it never improvises. Each terminal state and whose move it is:

| State | Whose move | What to do |
|---|---|---|
| `landed` | nobody | Done. `release_notes` in the return value carries the changelog lines. |
| `parked` (CI red, needs_human, refused merge) | human | Read the comment the run left on the issue and the car's run report under `docs/agents/runs/` (gitignored; it is there in your tree, not in the repository). The PR stays open and the worktree stays in place; fix and merge by hand, or re-triage the ticket. A later train skips it as work-in-flight — it will not be re-dispatched behind your back. |
| `parked` after a failed recovery | human | The workflow already spent its one rebase-and-resolve attempt. A second conflict on the same car means the branch and the tip disagree about something real — read the conflict, don't re-run it. |
| `parked-by-dependency` | human | Nothing wrong with the car itself; clear its blocker's derail and relaunch. |
| `halted-main-red` | **human, before anything else** | Main is poisoned and every undispatched car was refused. Revert-first is the default (`git revert <merge_commit>` — the run report names the commit); roll forward only if the fix is obvious and quick. Nothing lands until main is green. |
| `not-dispatched` | human | Relaunch `{ spec: <n>, merge: true }` once the halt or `maxCars` cap that stopped the train is dealt with. Position rebuilds from GitHub state alone — the per-car ledger lives as comments on the spec issue. |

After every train: the runs may have filed new issues under `agent-found` + `needs-triage`. Triaging those is a human's job (`/triage`), and `ready-for-agent` is a human's label to apply.
