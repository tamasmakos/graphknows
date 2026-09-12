# Contract — MCP tools

**Surface**: `graphknows-mcp`, served from `graphknows/server/mcp/`. This is what the agent
integration and any hosted client call. The stdlib-only client in
`graphknows/integrations/client/` speaks it.

---

## `memory_ingest`

| In | Type | Notes |
|---|---|---|
| `uri` | `str` | Resolvable location. Selects the parser by its MIME. |
| `namespace` | `str` | Optional; defaults to the server's namespace. |
| `meta` | `object` | Optional parser hints (session id, `cwd`, `gitBranch`). |

Returns `IngestReport` as JSON: `source_id`, `segments_written`, `facts_written`,
`entities_touched`, and the full ingest counter set.

- Deduplicates on content hash before any model runs (FR-007). A repeat ingest returns
  `sources_deduplicated: 1` and writes nothing (SC-008).
- Serialised per namespace; a queued call reports `ingest_queue_wait_ms` rather than failing
  (FR-047, SC-015).
- A drifted or malformed record is counted, never fatal (FR-027, SC-014).

## `memory_recall`

| In | Type | Notes |
|---|---|---|
| `query` | `str` | |
| `budget` | `object` | Optional. Explicit bound on the returned slice (FR-010). |
| `namespace` | `str` | Optional. |

Returns `RecallResult` as JSON: `facts` (each with its evidence — `source_uri`,
`byte_range`, `text`), `no_evidence`, `budget`, `truncated_by`, `counters`.

- Results are facts with evidence, never segments (FR-009, SC-002).
- An empty or below-floor pool returns `no_evidence: true`, **not** an empty `facts` array
  presented as success (FR-015). A caller can always tell "nothing is known" from "the
  lookup failed".
- When the budget cuts the slice, `truncated_by` names the ordering that did the cutting.
  Truncation is visible (edge case 6).

## `memory_forget`  *(new in this slice)*

| In | Type | Notes |
|---|---|---|
| `fact_id` | `str` | Exactly one of `fact_id` / `entity_id`. |
| `entity_id` | `str` | |
| `namespace` | `str` | Optional. |

Tombstones the record and everything whose only evidence it was. Returns the counters,
including `facts_excluded_tombstoned`.

- Never deletes. Merge logs and provenance stay replayable (FR-037, SC-012).
- Not behind an admin flag — per-fact forget is a user operation, unlike the existing
  namespace purge (audit C5).

---

## Removed

`topics_*` tools go with `graphknows.topics` (FR-046). They were the only readers of a plane
that is no longer written.

---

## Client discipline

`graphknows/integrations/client/` stays stdlib-only, enforced by the
`client-stdlib-only` contract in `.importlinter`. The in-loop hook process reaches memory
**only** through this client, against the running service, and loads no machine-learning
dependency (FR-036, SC-010).

Suggested `.mcp.json` for an agent workspace:

```json
{"mcpServers": {"graphknows": {"command": "graphknows-mcp",
  "env": {"GRAPHKNOWS_MODE": "llm_free", "GRAPHKNOWS_NAMESPACE": "repo"}}}}
```

## Verify

```bash
pytest tests/mcp_server tests/client -q
lint-imports        # client and hooks stay stdlib-only
```
