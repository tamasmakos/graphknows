---
name: deep-code-review
description: Interactive refactoring interview — the user brings an architectural pain, the agent digs with graphify metrics and code reading between questions, and together they converge on behavior-preserving pattern moves.
disable-model-invocation: true
---

# Deep Code Review

The user arrives sensing a problem — a module that fights every change, a class nobody wants to touch. Interview them about it the way `/grilling` does: one question at a time, walking the decision tree until you reach a **shared understanding** of what's wrong and which refactor cures it. The twist: **between questions, you dig** — graphify metrics, code reading, git history — so every question you ask is grounded in evidence, and every recommendation carries a number.

The division of labor is `/grilling`'s: **facts are yours** (never ask what you can look up); **decisions are the user's** (what counts as pain, which move to take, what's deliberately ugly and staying). The end state is a refactor list you both believe in — small, behavior-preserving catalog moves, never a rewrite. Where `/improve-codebase-architecture` redesigns module boundaries, this skill works inside them: Pythonic OOP hygiene.

## Opening

Ask what hurts — where they feel the friction and what they were trying to do when they felt it. If they already named a target, skip the question and start digging there. If they genuinely have no target, run the scoreboard (below) and offer the ranked hotspots as candidates: the itinerary is theirs to pick.

Then name the **destination** in one line and get their nod — what "fixed" looks like for *this* pain ("adding a new retrieval channel means touching one file, not five"). Every later question steers toward it; work beyond it is out of scope for this session.

If `graphify-out/graph.json` is missing or predates recent work (check `built_at_commit` against `git log`), refresh it via the `/graphify` skill (`--update` when a graph exists).

## Speak developer

Every fact, finding, and question is written for a developer skimming at speed — plain words, precise claims:

- **Translate every metric into its consequence.** The number is evidence, never the headline. Not "settings.py has Ca=45 and an SDP violation of severity 0.34" but "45 files import `settings.py`, so any change to it can break 45 places — and it currently imports `memory.py`, which changes far more often than it does, so it inherits that churn."
- **Every claim carries its receipt**: the file:line it happened at, the exact number, and the concrete scenario where it bites ("rename a field here and these three call sites in `retrieval/` break silently").
- **Jargon only after it's earned.** Name a smell or pattern once you've shown the thing in their code; from then on the name is shorthand you both share, not a label doing the arguing.
- **A question states its stakes.** Not "should we extract a class?" but "these 12 methods only ever touch the vector-index fields — split them out, and `graph_store.py` stops changing every time indexing does. Split, or is there a reason they live together?"

## The loop

Repeat until shared understanding:

**1. Dig.** Gather evidence for or against the current hypothesis before saying anything:

- Run the scoreboard (`python .claude/skills/deep-code-review/metrics.py`) or re-read the relevant section of its output; use `graphify query` / `path` / `explain` for the story behind a number.
- **Read the suspect code in full** — the number says where, the code says what. Name the problem as a smell from [CODE-SMELLS.md](CODE-SMELLS.md), with file:line — or conclude the number is innocent (a high-degree hub that is a stable, deep interface is healthy, not a god node) and say so.
- Check git history when intent matters: is this hot because it's broken, or because it's where the work is?

**2. Ask.** Put **one decision** to the user — the sharpest question the evidence currently supports — with your recommended answer, and wait. Never bundle questions, and never ask about fog you can't yet phrase precisely: dig until the question is sharp, or park it aloud ("once we settle the ingest split, there's a follow-up about the schema module — not sharp enough to ask yet"). The decisions, in rough dependency order:

- Is this the real pain, or a symptom of something upstream? (e.g. "the graph says `stm/ingest.py` co-changes with `graph_store.py` 15× with no import edge — is the pain really in ingest, or in the implicit contract between them?")
- Which smell names it — do they recognize the diagnosis?
- Which move cures it — technique or pattern, and which variant?
- What's the after-shape — predicted number shift ("god node, degree 154 → three classes around 50"), what the interface becomes, what tests survive?

**3. Fold in the answer.** What the user tells you — why the code is shaped this way, what's about to change anyway, what's off-limits — is context no graph can see. It reshapes every later dig and verdict. A rejection with a reason is a finding too: record it so you don't re-propose the same thing.

Every few decisions, and whenever the direction shifts, post a one-breath **decisions so far** — each agreed move in a line, by plain name ("split the vector-index methods out of GraphStore") — so the user always sees the map they're navigating and how far the destination is.

## Reading the scoreboard

`metrics.py` prints five sections (feasibility audit with recipes and thresholds: `docs/research/2026-08-10-graphify-code-metrics.md`):

| Section | Signal | Reading |
|---|---|---|
| 0 Coverage | relation counts, INFERRED share, cross-file call % | the trust envelope — quote it with any verdict; it also names what this graph **cannot** measure (CBO, feature envy, LCOM4, cyclomatic — don't claim them) |
| 1 Imports | high Ca file | ripple risk — every change fans out; deserves a stable, small interface |
| 1 Imports | SDP violation (depends on something less stable) | Shotgun Surgery risk → invert the dependency or extract the stable part |
| 2 Cycles | non-trivial SCC | cyclic dependency — break with Extract Class or dependency inversion |
| 2 Cycles | high blast radius × churn | changes here reach everything |
| 3 Communities | tangled file (members in 3+ communities) | Divergent Change — file changes for many reasons → split along community lines |
| 3 Communities | low community-vs-folder NMI | package layout lies about the real structure — communities are the reorg proposal |
| 4 Hubs | god node (top degree/PageRank) | Large Class / god module → Extract Class/Method, Facade |
| 4 Hubs | RFC > 50 | testability burden → Extract Class along the seams you find by reading |
| 5 Evolution | hotspot (churn × Ca) | where refactoring pays — rank candidates by this |
| 5 Evolution | hidden coupling (co-changed, no import edge) | Inappropriate Intimacy the static graph can't see — usually a shared implicit contract |

## Choosing the move

- Start with a technique from [REFACTORING-TECHNIQUES.md](REFACTORING-TECHNIQUES.md) — most smells die to Extract Method/Class, Move Method, or guard clauses.
- Reach for a pattern from [PYTHON-DESIGN-PATTERNS.md](PYTHON-DESIGN-PATTERNS.md) only when it **deletes code**: a Strategy that retires an if/elif ladder, an Observer that retires hand-rolled callbacks. Prefer the Pythonic form — plain functions as strategies, `@property`, dataclasses, protocols, context managers — over ceremony class hierarchies.
- Every move is behavior-preserving: no new features, no API changes beyond the refactor itself, existing tests pass unchanged (tests that poke privates may move with the code).

## Close

When the destination is reachable and the user has no unanswered pain left, recap the agreed moves — each with smell, number, move, and predicted shift — ranked by (hotspot score × ease), and name the one you'd make first. Confirm the recap matches their understanding. Apply nothing — implementation is a separate ask, and only starts once they confirm.
