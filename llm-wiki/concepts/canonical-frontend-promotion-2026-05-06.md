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

A user-facing trust break occurred: the public Cloudflare link exposed the canonical repo-root SPA on `:8080`, while the user expected the darker immersive integration build that had only existed in a temporary worktree.

Related pages:
- [[immersive-answer-graph-redesign-2026-05-06]]
- [[integration-qa-immersive-answer-graph-2026-05-06]]
- [[frontend-v2-design]]

## Root cause

The repository had a canonical served SPA at:
- `~/graphknows/apps/web/index.html`

But meaningful redesign work had been done in temporary worktrees, especially:
- `/tmp/iaa-integration/apps/web/index.html`
- `/tmp/gk-ux-integration/apps/web/index.html`

This created functional frontend drift between:
1. the public/canonical surface served by `apps/web/server.py` on `:8080`
2. the reviewed but unpromoted immersive integration frontend

## Decision

Promote the immersive dark-blue integration SPA into the canonical served frontend.

## Action taken

- copied `/tmp/iaa-integration/apps/web/index.html`
- to `~/graphknows/apps/web/index.html`

Verification:
- canonical local file SHA256 = `982964839415d0046418a70fd2223169263a84a5918800fc63f7bbf0ae821d45`
- integration file SHA256 = `982964839415d0046418a70fd2223169263a84a5918800fc63f7bbf0ae821d45`
- local `http://127.0.0.1:8080/` returned the same SHA256
- public quick tunnel `https://gifts-eight-december-east.trycloudflare.com` returned the same SHA256

## Result

The canonical public frontend now matches the dark-blue immersive integration build, including:
- simpler shell
- single `⚡ Force ▼` control
- persistent inspector presence
- removal of the earlier hero-style light/alternate shell from the public entrypoint

## Operational rule going forward

Do not publish a public frontend link unless the exact served file is verified against the intended canonical source by SHA or direct diff.
