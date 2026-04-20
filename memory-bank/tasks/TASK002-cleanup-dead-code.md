# TASK002 — Phase 1b: Resolve Merge Conflicts & Delete Dead Code

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Fix the 3 files with unresolved `<<<<<<< HEAD` merge conflict markers, and delete all code and directories that are explicitly out of scope for the refactor.

## Thought Process
This is a housekeeping task that must happen before any other implementation, otherwise downstream edits will hit conflict markers. It's also the natural moment to remove the dead code that the clean-break decision eliminates — doing it early keeps diffs clean.

Files with confirmed merge conflicts:
- `services/graphgen/src/main.py`
- `services/graphrag/src/main.py`
- `frontend/static/script.js` (deleted anyway as part of frontend removal)

Dead code to delete:
- `frontend/` entire directory (replaced by `apps/web/`)
- `services/graphgen/src/simulation/` (life-log specific)
- `services/graphgen/src/kg/graph/parsers/life.py`
- `services/graphgen/src/kg/graph/parsers/base.py` (life-log specific base, replaced by Phase 2)
- `services/graphgen/src/kg/graph/parsing.py` (`SegmentData` model)
- `example_graphgen.py`, `example_graphrag.py`, `smoke_test.py` (replaced by scripts/)
- `requirements.txt` at root (was for old frontend BFF only)
- `input/synthetic_gen.py`, `input/synthetic_test_data.csv`

Files to resolve (keep the right side of the conflict):
- `services/graphgen/src/main.py` — accept the cleaner version
- `services/graphrag/src/main.py` — accept the cleaner version

## Implementation Plan
- [ ] Resolve merge conflicts in `services/graphgen/src/main.py`
- [ ] Resolve merge conflicts in `services/graphrag/src/main.py`
- [ ] Delete `frontend/` directory
- [ ] Delete `services/graphgen/src/simulation/` directory
- [ ] Delete life-log parser files
- [ ] Delete root example scripts and requirements.txt
- [ ] Delete `input/synthetic_*` files
- [ ] Move `input/GOAL.MD` and `input/GOAL copy.MD` to `docs/`

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 2.1 | Resolve graphgen/main.py conflicts | Not Started | 2026-04-20 | |
| 2.2 | Resolve graphrag/main.py conflicts | Not Started | 2026-04-20 | |
| 2.3 | Delete frontend/ directory | Not Started | 2026-04-20 | Confirm with user before rm -rf |
| 2.4 | Delete simulation/ directory | Not Started | 2026-04-20 | |
| 2.5 | Delete life-log parser files | Not Started | 2026-04-20 | |
| 2.6 | Delete root scripts + requirements.txt | Not Started | 2026-04-20 | |
| 2.7 | Move GOAL docs to docs/ | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.
