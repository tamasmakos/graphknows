---
title: Canonical Frontend Promotion — 2026-05-06
created: 2026-05-06
updated: 2026-05-06
type: concept
tags: [frontend, architecture, admin]
sources: []
confidence: high
---

# Canonical Frontend Promotion — 2026-05-06

## Why this note exists

A user-facing trust break occurred: the canonical repo-root SPA served publicly did not match the intended immersive integration build that the user expected.

Related pages:
- [[immersive-answer-graph-redesign-2026-05-06]]
- [[integration-qa-immersive-answer-graph-2026-05-06]]
- [[frontend-v2-design]]

## Root cause

The repository had a canonical served SPA at `apps/web/index.html`, but meaningful redesign work had also been happening in isolated integration worktrees before promotion into the canonical entrypoint.

This created frontend drift between:
1. the canonical surface served by `apps/web/server.py`
2. the reviewed but unpromoted immersive integration frontend

## Decision

Promote the immersive dark-blue integration SPA into the canonical served frontend.

## Action taken

- promoted the immersive integration SPA into `apps/web/index.html`
- verified that the local served HTML matched the intended canonical file
- verified that the publicly exposed HTML matched the same canonical file before sharing it

## Result

At that checkpoint, the canonical public frontend matched the immersive integration build that had just been promoted, including:
- a darker, simpler shell
- persistent inspector-led graph interaction
- removal of the earlier public-shell mismatch

This page is a historical checkpoint. The canonical frontend continued evolving afterward; see [[frontend-reconciliation-2026-05-06]] for the later richer same-day promotion.

## Operational rule going forward

Do not publish a public frontend link unless the exact served file is verified against the intended canonical source by SHA or direct diff.
