---
title: Frontend Reconciliation — 2026-05-06
created: 2026-05-06
updated: 2026-05-06
type: concept
tags: [frontend, qa, architecture, operations]
sources: []
confidence: high
---

# Frontend Reconciliation — 2026-05-06

## Summary
The latest richer same-day frontend work was not lost. It had diverged into a richer integration worktree variant while the canonical/public surface was temporarily serving an older promoted checkpoint. Reconciliation is now complete: the canonical served SPA has been replaced with that richer same-day variant.

## Canonical source of truth
- Canonical served SPA: `apps/web/index.html`
- Reconciled source promoted into canonical: the richer same-day frontend worktree variant
- Previous older canonical snapshot was preserved before reconciliation

## Hash verification
- Current canonical file SHA256 recorded during reconciliation: `cc4f0f931302830e83c387289d0542cc823ec79890fea7411d048d4e76ad1c16`
- Local served HTML matched the same SHA256 during verification
- Publicly exposed HTML matched the same SHA256 during verification
- Previous older canonical checkpoint SHA256: `982964839415d0046418a70fd2223169263a84a5918800fc63f7bbf0ae821d45`

## Features restored into canonical/public build
- richer onboarding copy and suggested-query chips
- `Knowledge / Lexical` graph toggle
- mobile answer/graph view switcher
- graph focus chip with clear action
- expanded graph toolbar and mobile graph controls
- richer source chip/footer behavior
- mobile graph focus/zoom workflow

## QA outcome
Dogfood QA confirmed:
- no desktop JS errors on load
- local and public HTML parity
- real mobile viewport behavior (`390x844`) for empty, result, graph, and focused-node states
- graph view gets usable space on mobile (`366x690` in probe run)
- focus chip becomes visible after node selection

## Important nuance
Local/public HTML parity is now correct, but live answer/source payloads can still vary across sessions. In probe runs, local and public returned slightly different source-chip counts for the same prompt, which looks like backend/runtime variability rather than frontend version drift.

## Operational rule
Before publishing or verifying a public frontend link:
1. identify the intended canonical frontend file
2. compare it against recent worktree candidates
3. hash-check local canonical vs public URL
4. run at least one mobile viewport pass
