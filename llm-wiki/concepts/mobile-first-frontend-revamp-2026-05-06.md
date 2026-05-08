---
title: Mobile-First Frontend Revamp — apps/web/index.html
created: 2026-05-06
updated: 2026-05-06
type: concept
tags: [frontend, design, ux, mobile, apps-web]
sources: []
confidence: medium
---

# Mobile-First Frontend Revamp — `apps/web/index.html`

## Why this page exists

The first integrated UX polish improved the desktop feel of the live SPA, but real iPhone-sized QA showed the product still failed badly on mobile:
- graph canvas effectively collapsed in answered state
- legend consumed too much vertical space
- answer + reasoning + sources + graph created a long-scroll trap
- controls were too dense for touch
- investigations chrome wasted vertical space

This page records the follow-up **mobile-first restructuring** done directly in the live SPA at `apps/web/index.html` inside the clean integration worktree.

## What changed

### 1. Answer-first mobile shell
- Added a mobile-only `Answer | Graph` view switcher in the header.
- Default mobile mode is **answer view**.
- Graph now lives in a separate mobile view instead of competing with the answer in one long column.
- Investigations sidebar is moved behind a hamburger-triggered drawer/backdrop.

### 2. Mobile onboarding / empty state
- Added a dedicated empty state card in the chat panel.
- Added suggestion chips that inject example questions into the input.
- Chat empty state is now synced in JS so it hides as soon as user/agent messages appear and returns on reset.

### 3. Mobile answer card restructuring
- Reasoning trace is collapsed inline inside each answer card instead of dominating the page.
- Added visible disclosure affordance to the trace summary.
- Added source-card section with numbered chips and moved export action into a dedicated footer row.
- Answer cleaning now strips or normalizes raw chunk/topic identifiers that leaked into prose.

### 4. Mobile graph separation and control simplification
- Added mobile graph toolbar actions for `Legend` and `Tools` so controls are hidden until needed.
- Graph controls collapse under a mobile toggle instead of staying always expanded.
- Legend is now a dismissible mobile overlay rather than a permanently competing block.
- Session row is hidden on mobile to reduce chrome.

### 5. Mobile graph layout mitigation
- Added mobile-specific graph re-centering and redraw when switching into graph view.
- Added a mobile-only grid-ish pre-placement before force layout to avoid the previous top-left collapse.
- Suppressed edge labels on mobile.
- Hid raw chunk/UUID-like node labels on mobile.
- Reduced mobile force charge/collision intensity to make the graph less chaotic.

## Verification

Local mobile verification was run against a dedicated integration server on `http://127.0.0.1:8082` using Playwright with iPhone 13 sizing.

Verified behaviors:
- mobile answer view shows only chat / input
- mobile graph view shows a separate graph canvas
- reasoning trace collapsed by default
- source chips render in answer footer
- session row hidden on mobile
- edge labels suppressed on mobile graph
- graph canvas height materially improved vs earlier collapsed state

Temporary public tunnel for this revamp:
- `https://consider-gathering-alone-believed.trycloudflare.com`

The tunnel HTML was verified to match local `:8082` exactly by SHA256.

## Remaining issues after revamp

The mobile revamp materially improved structure, but it is **not done**:
- home screen still has too much dead vertical space
- input area still needs safer bottom spacing / tighter composition
- source/export footer still needs finer layout polish
- graph remains too dense on phone for true analytical use
- node labels still truncate heavily
- graph needs a more principled mobile interaction model (progressive disclosure, tap-to-focus, probably fewer default labels)

## Key lesson

For this product, mobile UX cannot be achieved by polishing a desktop split-pane layout. The correct direction is:
- answer-first mobile flow
- graph as a secondary dedicated mode
- hidden investigations / tools / legend
- stronger information hierarchy
- lower-density graph defaults on phone

## Links
- [[frontend-v2-design]]
- [[graph-visualisation-diagnosis]]
- [[architecture-reference]]
