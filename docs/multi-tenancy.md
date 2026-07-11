# Multi-tenancy & scaling

graphknows isolates tenants with **namespaces**. A namespace maps to its own
physical ArcadeDB database pair — `stm_<ns>` (working memory) and `ltm_<ns>`
(long-term memory) — so a query in one namespace **physically cannot** read
another namespace's data. There is no shared row-level tenant column to get
wrong: isolation is by construction.

The empty namespace (`""`) resolves to the original `stm`/`ltm` databases, so a
single-tenant deployment needs no changes and existing data is untouched.

## The two levels of scope

| Scope | What it is | Isolation |
|---|---|---|
| **`namespace`** | Physical database pair `stm_<ns>`/`ltm_<ns>` | **Hard** — separate databases, no cross-talk |
| **`session_id`** | Partition *within* a namespace (encodes `user_id`/`agent_id`/`run_id`) | STM: hard per session. LTM: consolidated within the namespace |

Pick the namespace granularity to match your isolation requirement:

- **Per tenant/customer (recommended default).** One namespace per customer or
  deployment. Users within a customer share that customer's consolidated
  long-term knowledge graph; different customers are physically separate. Best
  fit for most B2B deployments, and it keeps the number of databases moderate.
- **Per end-user.** Set `namespace = user_id` when each user must be a hard
  island (a personal-assistant product). Strongest isolation, but see *Scaling*
  below — this can mean a very large number of databases.

### Why LTM consolidates *within* a namespace

On flush-to-LTM, entities are merged by name and the topic / PageRank /
community layer is computed over the whole namespace. This is a **feature**: all
of a tenant's sessions consolidate into one knowledge graph (two sessions that
both mention "Acme" resolve to one entity). It is scoped to the namespace, so it
never crosses tenants. The corollary: if two *users* must never share an entity
graph, give them **separate namespaces** (`namespace = user_id`) rather than
separate `session_id`s within one namespace.

## Using namespaces

Every MCP memory tool takes an optional `namespace` argument; when omitted it
falls back to the server's `GRAPHKNOWS_NAMESPACE`. One server process serves many
tenants concurrently — each namespace gets its own cached runtime and its own
databases, so concurrent tenant calls never interleave data.

```jsonc
// ingest for one tenant
memory_ingest { "text": "...", "session_id": "u:alice", "namespace": "acme" }
// query only sees acme's data
memory_query  { "query": "...", "namespace": "acme" }
// tear a tenant down in O(1) — drops both databases
memory_drop_namespace { "namespace": "acme" }
```

A namespace's databases are created automatically on first ingest; no
provisioning step is required. `memory_drop_namespace` is the fast reset — it
drops the two databases outright instead of walking and deleting every node.
(It refuses the default `""` namespace, whose shared `stm`/`ltm` must be cleared
with `memory_purge` instead.)

## Scaling guidance

- **Database count.** Each namespace is two ArcadeDB databases, each with its own
  files, schema, and vector indexes, and carries memory overhead while open.
  Per-tenant granularity keeps this moderate. Per-user granularity at tens of
  thousands of users means tens of thousands of databases — size the ArcadeDB
  server accordingly, or keep a namespace per *tenant* and use `session_id` for
  per-user partitioning within it.
- **Open-runtime cap.** The MCP server keeps at most `GRAPHKNOWS_MAX_OPEN_NAMESPACES`
  (default 64) namespace runtimes open at once, evicting (closing) the
  least-recently-used, so unbounded tenant cardinality can't leak connection
  pools. Raise it to match your active-tenant working set and server sizing.
- **Concurrency.** Because tenants map to distinct databases, concurrent
  multi-tenant traffic needs no cross-tenant locking. Within a single namespace,
  keep ingestion serialized (`EVAL_INGEST_CONCURRENCY`/write path) — concurrent
  writes to the same graph race the entity MERGE.
- **If you truly need massive lightweight tenancy** (hundreds of thousands of
  tiny tenants), a shared-database row-scope model is the alternative to revisit;
  it trades hard physical isolation for a single database at the cost of adding a
  tenant predicate to every read. The current design deliberately favours
  correctness and hard isolation for the per-tenant/per-user cases.

## Evaluations

The benchmark harness uses namespaces so each mode runs in its own database
pair without contaminating the other (`eval_locomo_llm_free`,
`eval_locomo_llm_assisted`). This also enables the fast tuning loop:

```bash
python -m evaluation --limit 20                   # ingest once into eval_locomo_llm_free
python -m evaluation --limit 20 --no-ingest       # tweak retrieval, reuse the graph
python -m evaluation --reset                      # drop this mode's namespace, clean re-ingest
```

See [`evaluation/README.md`](../evaluation/README.md).
