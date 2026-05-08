# Mobile graph usability pass (2026-05-06)

## Context
This note captures the follow-up graph-specific pass after the broader mobile-first SPA revamp in `apps/web/index.html`.

The earlier mobile refactor fixed the structural problem by separating **Answer** and **Graph** views on phones, but the graph itself was still only partially usable. The main remaining issues were:
- too many labels or the wrong labels shown by default
- graph framed too low-value for mobile reading
- focused-node mode existed only weakly
- interaction hints were vague
- graph chrome consumed too much vertical space
- the selected neighborhood was still not clear enough on phones

## File / branch
- worktree: `/tmp/gk-ux-integration`
- branch: `feat/ux-integration`
- file: `/tmp/gk-ux-integration/apps/web/index.html`

## What changed

### 1. Stronger mobile graph framing
Adjusted the mobile graph shell so more vertical space is reserved for the actual canvas.

Changes included:
- compressed mobile graph header spacing
- reduced graph-toolbar footprint in graph view
- reduced mobile switcher height in graph view
- increased graph canvas height from the previous mobile calculation to:
  - `min-height: calc(100dvh - 154px - env(safe-area-inset-bottom, 0px))`

Result:
- graph view now uses materially more of the phone screen
- graph toolbar height in QA measured at about `48px`
- graph canvas height in QA measured at about `510px`

### 2. Focus chip and tap-to-focus mode
Added a clearer focused-node mode and made the focused state visually explicit.

Implemented:
- persistent mobile focus chip overlay inside `#graph-canvas`
- focused node title + metadata + clear action
- helper copy: `Tap again to clear focus`
- selected node gets a stronger highlight ring/stroke
- clear button resets focus and zoom state

Behavior:
- tapping a node sets `selectedNode`
- focus chip updates with node label, neighbor count, and relation-type count
- focused node zooms in with a mobile-specific scale and offset
- tapping again clears the focus state

### 3. Better mobile zoom/focus behavior
Improved mobile node zooming so focused nodes land in a more usable part of the viewport instead of feeling vaguely centered or clipped.

Implemented in `zoomToNode(...)`:
- mobile scale increased to `1.85`
- added downward offset compensation so the node is not hidden under the focus chip
- zoom state persists through graph-view transitions with `lastZoomTransform`

Result:
- focused state is now visibly different and meaningfully more usable
- selected node is easier to inspect than in the prior pass

### 4. Progressive disclosure for labels
Reduced default graph noise while keeping some at-a-glance meaning.

Implemented:
- mobile graph still limits rendered nodes to a top-ranked subset (`maxNodes = 20`)
- introduced `mobileLabelIds` so only a curated subset of nodes receives visible labels
- default mobile graph now prefers a few higher-value labels rather than attempting to label everything
- focused mode keeps labels for the selected node and prioritized nearby nodes
- chunk-style / hashy / low-value node labels remain suppressed
- default visible labels are now chosen using mobile-specific ranking that considers:
  - selected node
  - selected-node adjacency
  - node type importance
  - local degree within visible edges

Observed outcome in QA:
- default mobile graphs typically showed about 6 visible labels out of 20 rendered nodes
- focused mode retained clearer context while still suppressing most long-tail clutter

### 5. Stronger node prioritization for mobile
Adjusted the mobile ranking heuristic so the default graph is less random and more policy-relevant.

Priority now favors:
- selected node first
- directly connected nodes when focused
- organizations / orgs
- laws / topics / persons
- then concepts / geography
- chunks stay low priority

This makes the default mobile subgraph more likely to foreground politically meaningful entities rather than noisy tail nodes.

### 6. Stronger selected-neighborhood styling
Improved contrast when a node is focused.

Implemented in `highlightConnected()`:
- focused node gets larger radius and bright ring
- direct neighborhood remains visible
- non-neighborhood nodes fade more aggressively
- non-prioritized labels stay hidden in mobile focus mode

Result:
- the focused node is now clearly identifiable
- neighborhood inspection is more understandable than in the earlier mitigation-only pass

## QA / verification

### Runtime verification
Verified locally:
- `http://127.0.0.1:8082`
- `http://127.0.0.1:8082/api/health`

Verified public quick tunnel still matches local exactly:
- `https://consider-gathering-alone-believed.trycloudflare.com`
- SHA256 for local and public HTML:
  - `4779d759a53eb98ae6ee3cda5ea5a666bb6a30ae612bd27d051c1af9d9d683b2`

### Mobile QA method
Repeated Playwright QA at iPhone-13-sized viewport against local `8082`, using real app execution via `sendMessage(...)` so graph rendering and selection behavior were exercised in the actual frontend code path.

Screenshot batches from this pass:
- `/tmp/mobile-graph-pass-3/`
- `/tmp/mobile-graph-pass-4/`
- `/tmp/mobile-graph-pass-5/`
- `/tmp/mobile-graph-pass-6/`
- `/tmp/mobile-graph-pass-7/`
- `/tmp/mobile-graph-pass-8/`

Representative final measurements:
- graph toolbar height: ~`48px`
- graph canvas rect: ~`366x510`
- rendered nodes in mobile graph view: `20`
- total graph nodes in tested session: ~`51`
- visible labels in final default mobile graph QA: `6`
- hidden labels in final default mobile graph QA: `14`
- edge labels shown in mobile mode: `0`

Representative final focus-state QA:
- `focusVisible: true`
- `focusTitle: Eu`
- `focusMeta: Organization • 14 nearby nodes • 3 relation types`

## Honest assessment
This pass materially improved the graph experience on phones.

What is now clearly better:
- graph uses more of the screen
- focus mode is real and visually clear
- selected node is obvious
- neighborhood inspection is more coherent
- default graph is less label-chaotic
- graph view feels more intentional rather than a shrunk desktop panel

What is still not solved:
- label legibility is still the main weakness
- too many nodes remain unlabeled in the default graph
- focused neighborhoods still do not surface enough neighbor labels
- relationship semantics are still mostly hidden because mobile edge labels remain suppressed
- graph understanding still depends too much on tapping around

## Recommended next graph pass
If continuing, the highest-value next move is **label strategy**, not more generic polish.

Specifically:
1. show short labels for all immediate neighbors in focused mode
2. add a tiny relation legend / relation chips for the focused neighborhood
3. consider one-step “focus summary” below the chip (top connected neighbors / relation mix)
4. consider a small “show more labels” toggle in mobile graph tools
5. optionally expose relation type filtering before revisiting full edge-label display

## Status
- implemented and verified in `/tmp/gk-ux-integration/apps/web/index.html`
- not yet committed
- current branch state remains modified (`M apps/web/index.html`)
