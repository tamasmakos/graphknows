# Wave 0 — Architecture Baseline Plan

**Issues:** #44 (package map), #45 (import rules), #46 (legacy audit), #47 (tach baseline)
**Scope:** Documentation + tach.toml update only. Zero production code changes.
**CI contract:** `tach check` must exit 0 after the tach.toml update.
**Status:** ✅ Delivered on branch `wave0-docs-tach`

---

## Issue #44 — ARCH-32.1: Target Package Map

### Target Package Map

| Target Package | Responsibility | Owner Tier | Allowed Consumers |
|---|---|---|---|
| `graphknows.domain` | Canonical entity labels, type constants, taxonomy schema. Pure data — no I/O. | Tier 0 | All packages |
| `graphknows.memory.models` | Shared DTOs: IngestRequest, RecallResult, PromotionRecord, Evidence. No I/O. | Tier 0 | memory.stm, memory.ltm, memory.semantic, builders, transports |
| `graphknows.config` | Settings, env resolution, health checks. Pure config — no business logic. | Tier 0 | All packages (leaf dependency) |
| `graphknows.adapters` | Thin infra wrappers: Neo4j, Memgraph, pgvector, embedder. No business logic. | Tier 1 | memory.*, builders, graphgen |
| `graphknows.ingestion` | Document parsers only. No graph I/O. | Tier 2 | graphgen, memory.stm |
| `graphknows.memory.stm` | Short-term ingest, STM recall, STM schema management. | Tier 2 | builders, transports |
| `graphknows.memory.ltm` | LTM promotion, enrichment (GDS/Node2Vec/Louvain), LTM recall, forget. | Tier 2 | builders, transports |
| `graphknows.memory.semantic` | Topic summaries, embedding storage, semantic search. | Tier 2 | builders, transports |
| `graphknows.graphgen` | Batch corpus pipeline (NER → relations → graph cleaning → community). Standalone subsystem. | Tier 2 | builders, transports |
| `graphknows.builders` | Composition root: wires adapters + services into usable handles. | Tier 3 | transports only |
| `graphknows.transports.mcp` | Thin MCP tool modules delegating to builders. | Tier 4 | Nothing (top of stack) |
| `graphknows.transports.cli` | Thin CLI commands delegating to builders. | Tier 4 | Nothing (top of stack) |

### Ownership Tiers (top-to-bottom dependency order)

```
Tier 0 (pure data):   domain, memory.models, config
Tier 1 (infra):       adapters
Tier 2 (services):    memory.stm, memory.ltm, memory.semantic, graphgen, ingestion
Tier 3 (composition): builders
Tier 4 (transports):  transports.mcp, transports.cli
```

**Key principle:** Dependencies only flow downward (higher tier → lower tier). No back-edges permitted.

### Notes

- `domain` currently contains only `schema.py` (label constants). Team must agree that
  taxonomy YAML loading also lives here (not in graphgen/config) before Wave 1 begins.
- `memory.models` vs `graphknows.models` naming collision with the existing stale
  `graphknows.models` package — deletion must be coordinated in Wave 2.
- `graphknows.migrations` is **excluded from tach tracking** (Wave 0 decision). It is an
  infrastructure-level package with a dependency profile that differs from the application
  graph. Add to tach tracking when its `depends_on` is known and stable.

---

## Issue #45 — ARCH-32.2: Allowed Import Directions & Banned Cross-Layer Dependencies

### Allowed Import Directions (downward only)

```
transports.mcp    → builders, domain, config
transports.cli    → builders, domain, config
builders          → memory.stm, memory.ltm, memory.semantic, graphgen, adapters, domain, config
memory.stm        → adapters, domain, memory.models, config, ingestion
memory.ltm        → adapters, domain, memory.models, config
memory.semantic   → adapters, domain, memory.models, config
memory.models     → domain, config
graphgen          → adapters, domain, config, ingestion
ingestion         → domain, config
adapters          → config, domain
config            → (nothing internal)
domain            → (nothing internal)
```

### Banned Back-Edges (must never appear)

| From | To | Why Banned |
|---|---|---|
| `adapters` | `memory.*` | Infra must not depend on services |
| `adapters` | `graphgen` | Infra must not depend on pipeline |
| `memory.stm` | `memory.ltm` | STM and LTM are peers; promotion flows via builders |
| `memory.ltm` | `memory.stm` | Symmetric peer ban |
| `memory.*` | `builders` | Services must not depend on composition root |
| `memory.*` | `transports.*` | Services must not depend on transports |
| `graphgen` | `memory.*` | Batch pipeline must not call live memory services |
| `graphgen` | `builders` | Same reason — graphgen is a standalone subsystem |
| `domain` | anything | Domain is a leaf; must have zero internal imports |
| `config` | anything | Config is a leaf; must have zero internal imports |
| `ingestion` | `adapters` | Parsers must not do graph I/O |
| `ingestion` | `memory.*` | Parsers must not call memory services |

### Current Known Violations (legacy smells — not yet banned in tach)

These violations exist in the current (legacy) codebase. They are documented here as
future CI enforcement targets; Wave 5 will encode them as `tach` failures. Until then,
they appear as `# SMELL` comments in `tach.toml`.

| Violation | Rule Violated | Resolution Wave |
|---|---|---|
| `graphknows.retrieval ↔ graphknows.graphgen` — bidirectional cycle | Banned: `graphgen → memory.*` equivalent | W2 |
| `graphknows.retrieval ↔ graphknows.memory` — bidirectional cycle | Banned: `memory.* → adapters` peer | W2 |
| `graphknows.ontology → graphknows.retrieval` — ontology calling infra | Ontology is not a defined target package; splits in W1/W3 | W3 |
| `graphknows.memory → graphknows.graphgen` — memory calling batch pipeline | Banned: `memory.* → graphgen` | W2 |

### Risks

- The `retrieval` package spans two target roles (adapters + memory services).
  Any rule that bans `memory → retrieval` breaks CI today. Rules must remain
  aspirational in Wave 0 tach and only be activated per-wave.
- If `graphgen` is kept as a standalone subsystem but still embeds its own
  `vector_embedder` and `pgvector_adapter` utilities, the adapter boundary is
  blurred. Team must decide: migrate graphgen's internal adapters to
  `graphknows.adapters` in Wave 3, or keep graphgen self-contained.

---

## Issue #46 — ARCH-32.3: Legacy Module Audit & Disposition

### Disposition Key

- **Migrate** — move to a target package; original path deleted after migration
- **Split** — contents go to multiple target packages; original deleted after all parts move
- **Delete** — no migration; callers updated or dead code removed
- **Keep** — already well-bounded; path unchanged in target architecture
- **Keep + Expand** — keep path, add content in future waves

### Legacy Module Dispositions

| Module | Current Path | Description | Disposition | Target Owner | Wave |
|---|---|---|---|---|---|
| `__init__.py` | `graphknows/__init__.py` | Package root; currently re-exports some symbols. | **Migrate** | Slim down to version metadata only. Re-exports move to builders. | W3 |
| `config.py` | `graphknows/config.py` | Runtime config (env vars, settings). | **Migrate** | `graphknows.config` package (becomes a proper package, not a module). | W1 |
| `container.py` | `graphknows/container.py` | Stale composition root. Wires adapters + services together. | **Migrate → Delete** | Logic moves to `graphknows.builders`. File deleted after Wave 3. | W3 |
| `feature_engineering.py` | `graphknows/feature_engineering.py` | LTM enrichment logic (Node2Vec, Louvain calls). Misplaced at root. | **Migrate** | `graphknows.memory.ltm.enrichment` | W2 |
| `forget.py` | `graphknows/forget.py` | Stale. Uses old label conventions (pre-domain/schema.py). | **Delete** | Reimplemented in `graphknows.memory.ltm.forget` using `domain.schema` labels. | W2 |
| `mcp_ltm.py` | `graphknows/mcp_ltm.py` | Re-export shim for `mcp.stdio_server`. Exists for backward compat. | **Delete** | Callers updated to import from `graphknows.transports.mcp`. | W4 |
| `recall.py` | `graphknows/recall.py` | Live recall orchestration. Misplaced at root. | **Migrate** | `graphknows.memory.ltm.recall` (LTM path) and `graphknows.memory.stm.recall` (STM path) | W2 |
| `schema.py` | `graphknows/schema.py` | DB constraints / DDL helpers. Separate from `domain/schema.py`. | **Migrate** | DDL helpers → `graphknows.adapters` (per-store constraint initialisation). | W3 |
| `models/` | `graphknows/models/` | Pipeline-era DTOs: session, entity, pipeline, constants, tools, base. | **Split** | Shared DTOs → `graphknows.memory.models`. Pipeline-only DTOs → `graphknows.graphgen`. Stale/dead models → deleted. | W2 |
| `retrieval/` | `graphknows/retrieval/` | Spans infra (stores, embedder) and services (recall_stm, recall_ltm, retriever, context). | **Split** | Infra → `graphknows.adapters`. Recall services → `graphknows.memory.stm/ltm`. | W2 |
| `ontology/` | `graphknows/ontology/` | `rel_extractor.py` (relation extraction) + `loader.py` (taxonomy YAML). | **Split** | `rel_extractor` → `graphknows.graphgen.pipeline.entity_relation`. `loader` → `graphknows.domain` (taxonomy). | W1/W3 |
| `memory/` | `graphknows/memory/` | Mixed: ingestor, extractor, dialog, flush, adapter, adapter_types. | **Restructure** | Each file maps to `memory.stm`, `memory.ltm`, or `memory.models` sub-packages. | W2 |
| `mcp/` | `graphknows/mcp/` | `stdio_server.py`: 797-line monolith containing all MCP tools. | **Split** | Each MCP tool group → thin module in `graphknows.transports.mcp`. | W4 |
| `graphgen/` | `graphknows/graphgen/` | Batch corpus pipeline. Well-bounded subsystem. | **Keep** | Remains as `graphknows.graphgen`. Internal utils (graphdb, vector_embedder) should eventually move to `graphknows.adapters` — deferred to W3. | — |
| `ingestion/` | `graphknows/ingestion/` | Document parsers. Well-bounded. | **Keep** | Remains as `graphknows.ingestion`. | — |
| `domain/` | `graphknows/domain/` | `schema.py` with canonical label constants. Created in Phase 0. | **Keep + Expand** | Taxonomy loader moves here in W1. Becomes the authoritative domain package. | W1 |
| `cli/` | `graphknows/cli/` | `memory.py`, `main.py` CLI entrypoints. | **Migrate** | Becomes `graphknows.transports.cli`. | W4 |
| `migrations/` | `graphknows/migrations/` | DB schema migrations. Infrastructure-level; not part of application graph. | **Excluded from tach** | Not tracked in tach.toml (Wave 0 decision). Revisit when `depends_on` is stable. | — |

### Key Deletion Candidates (require no migration work)

- `forget.py` — reimplement from scratch; old label conventions are incompatible
- `mcp_ltm.py` — pure shim; callers can update import directly
- `container.py` — superseded by `builders` package

### Validation

Every package/file in `graphknows/` at time of Wave 0 appears in exactly one row of
the table above. Verify with:

```bash
find graphknows/ -name "*.py" | grep -v __pycache__ | sort
```

No row has disposition "TBD".

### Risks

- `recall.py` imports from `graphknows.retrieval` which is being split. The migration
  order matters: adapters must exist before recall.py moves.
- `models/` split requires careful DTO archaeology — some models may be referenced
  by both graphgen and memory. A dead-import scan (e.g. `vulture`) should run before
  Wave 2 deletes anything.
- `schema.py` (root) and `domain/schema.py` have overlapping names. Engineers must
  not confuse them during Wave 3.

---

## Issue #47 — ARCH-32.4: tach Baseline Update

See `tach.toml` for the concrete changes. Summary of Wave 0 tach changes:

1. **Added `graphknows.domain`** as a zero-dependency tracked module (it already exists
   and is imported by root-level `schema.py`, but was previously untracked).
2. **Added `# TARGET ARCHITECTURE` comment block** near the top documenting the
   aspirational `depends_on` for each future package (human-readable; not parsed by tach).
3. **Preserved all existing `[[modules]]` entries unchanged** so `tach check` keeps
   passing on the current codebase.
4. **Annotated each existing module** with inline `# TARGET:` and `# SMELL:` comments
   for known violations, citing banned back-edge rules from §#45.
5. **Did NOT add** `graphknows.memory.stm`, `graphknows.transports.*`, etc. — those
   packages don't exist yet; adding them would cause tach to error.
6. **`graphknows.migrations` excluded** — not added to tach; noted as untracked.

**Validation:** `tach check --no-cache` exits 0.

---

## Cross-Cutting Risks

1. **tach syntax drift** — tach TOML syntax has changed between versions. The
   `source_roots = ["."]` (no `[tool.tach]` wrapper) format must be verified
   against the installed tach version before editing.

2. **`graphknows.domain` import graph** — `domain/schema.py` is imported by root-level
   `graphknows/schema.py` (untracked in tach) but by no tracked modules directly.
   Adding `domain` as a tracked zero-dependency module is safe under the current layout.

3. **`graphknows.migrations` is untracked** — explicitly excluded in Wave 0. Revisit
   before any `tach sync` in later waves to avoid auto-add with unexpected dependencies.

4. **Documentation staleness** — the package map document lives in `.hermes/plans/`
   which is not normally code-reviewed. Consider symlinking or copying to `docs/`
   so it stays in the PR review path.

5. **`graphknows.models` name collision** — the existing stale `models/` package
   shares a prefix with the target `memory.models`. IDEs and importers may
   auto-suggest the wrong one during Wave 2 migration.

---

## Unresolved Questions (for PM to resolve before engineering starts)

1. **Taxonomy loader ownership**: Does `graphknows.domain` own the taxonomy YAML
   loader (currently in `graphknows.graphgen.config.taxonomy` and
   `graphknows.ontology.loader`)? Or does graphgen keep its own copy? This
   determines whether `domain` stays zero-dependency or gains a YAML file-I/O dep.

2. **graphgen adapter extraction**: Should `graphgen/utils/graphdb/` and
   `graphgen/utils/vector_embedder/` migrate to `graphknows.adapters` in Wave 3,
   or is graphgen kept fully self-contained?

3. **`graphknows.migrations` disposition**: Wave 0 decision is "excluded from tach".
   What `depends_on` is correct if added in a future wave?

4. **`graphknows.schema.py` (root) naming**: Should root-level `schema.py` be
   immediately renamed to `graphknows.db_schema` or `graphknows.ddl` to eliminate
   the name collision with `graphknows.domain.schema` before Wave 1 moves anything?

5. **Wave 5 enforcement timeline**: Will banned back-edges (§#45) be enforced
   incrementally per wave, or all at once in Wave 5?

6. **`vulture` / dead-code scan**: Should a dead-import scan run as part of Wave 0
   (documentation only, no fixes) to inform the legacy audit?

---

## Recommended Sequencing

### Can #44–#46 be done in one file? **Yes.**
Issues #44, #45, and #46 are all documentation. They belong in a single markdown
file (`wave0-package-map.md`) with clear section headers.

### Should #47 be a separate commit? **Yes.**
The tach.toml change is the only file that has a mechanical validation gate
(`tach check`). Keeping it in a separate commit makes it easy to bisect if
`tach check` fails unexpectedly.

```
Commit 1 (docs):  .hermes/plans/wave0-package-map.md  (#44, #45, #46)
Commit 2 (tach):  tach.toml                           (#47)
```

### Minimum viable Wave 0 done-criteria

- [x] `wave0-package-map.md` exists and covers all three documentation issues
- [x] Every current `graphknows/` package/file has a row in the legacy audit table
- [x] `tach.toml` includes `graphknows.domain` as a tracked module
- [x] `tach check` exits 0
- [x] No production `.py` file has been created, moved, or deleted
