# Multi-tenancy & scaling

graphknows isolates tenants with **namespaces**. A namespace maps to its own
physical ArcadeDB database — `mem_<ns>` — so a query in one namespace
**physically cannot** read another namespace's data. There is no shared
row-level tenant column to get wrong: isolation is by construction.

One server process serves exactly **one** namespace, bound at startup from
`GRAPHKNOWS_NAMESPACE`. No caller can name a namespace, so no caller can reach
another tenant's database. Serving several tenants means running
one process per tenant.

The empty namespace (`""`) resolves to the shared `mem` database, so a
single-tenant deployment needs no changes and existing data is untouched.

Short-term and long-term memory are **not** separate databases. They are a
`state` lifecycle (`raw` → `consolidated`) on one graph, flipped in place by
flush. See [architecture.md](architecture.md); the resolution rules live in
`graphknows/storage/namespace.py`.

## The two levels of scope

| Scope | What it is | Isolation |
|---|---|---|
| **`namespace`** | Physical database `mem_<ns>`, fixed per process | **Hard** — separate databases and separate processes, no cross-talk |
| **`session_id`** | Partition *within* a namespace (encodes `user_id`/`agent_id`/`run_id`) | Soft — a filter on one graph. Consolidated entities are shared namespace-wide. |

Pick the namespace granularity to match your isolation requirement:

- **Per tenant/customer (recommended default).** One namespace — and therefore
  one process — per customer or deployment. Users within a customer share that
  customer's consolidated long-term knowledge graph; different customers are
  physically separate. Best fit for most B2B deployments, and it keeps the
  number of databases and processes moderate.
- **Per end-user.** Set `GRAPHKNOWS_NAMESPACE` to the user id when each user
  must be a hard island (a personal-assistant product). Strongest isolation,
  but see *Scaling* below — this means one database *and one process* per user.

### Why consolidation happens *within* a namespace

On flush, entities are merged by name and the topic / PageRank /
community layer is computed over the whole namespace. This is a **feature**: all
of a tenant's sessions consolidate into one knowledge graph (two sessions that
both mention "Acme" resolve to one entity). It is scoped to the namespace, so it
never crosses tenants. The corollary: if two *users* must never share an entity
graph, give them **separate namespaces** — separate processes — rather than
separate `session_id`s within one namespace.

## Running a tenant

Set the namespace in the environment of the process and start it. Every MCP
memory tool then operates on that namespace; none of them takes a `namespace`
argument.

```bash
GRAPHKNOWS_NAMESPACE=acme python -m graphknows.server.mcp   # serves acme, only acme
```

```jsonc
// ingest and query hit the process's own namespace
memory_ingest { "text": "...", "session_id": "u:alice" }
memory_query  { "query": "..." }
// tear this tenant down in O(1) — drops the process's database
memory_drop_namespace {}
```

The namespace's database is created automatically on first ingest; no
provisioning step is required. `memory_drop_namespace` is the fast reset — it
drops the database outright instead of walking and deleting every node.
(It refuses the default `""` namespace, whose shared `mem` must be cleared
with `memory_purge` instead. Both are destructive tools: they are only
advertised when `GRAPHKNOWS_ENABLE_ADMIN_TOOLS` is set.)

## Scaling guidance

- **Database and process count.** Each namespace is one ArcadeDB database with
  its own files, schema, and vector indexes, fronted by its own server process.
  Per-tenant granularity keeps both moderate. Per-user granularity at tens of
  thousands of users means tens of thousands of databases and processes — size
  the deployment accordingly, or keep a namespace per *tenant* and use
  `session_id` for per-user partitioning within it.
- **Concurrency.** Because tenants map to distinct databases in distinct
  processes, multi-tenant traffic needs no cross-tenant locking. Within a single
  namespace, keep ingestion serialized (`EVAL_INGEST_CONCURRENCY`/write path) —
  concurrent writes to the same graph race the entity MERGE.
- **If you truly need massive lightweight tenancy** (hundreds of thousands of
  tiny tenants), a shared-database row-scope model is the alternative to revisit;
  it trades hard physical isolation for a single database at the cost of adding a
  tenant predicate to every read. The current design deliberately favours
  correctness and hard isolation for the per-tenant/per-user cases.

## Evaluations

The benchmark harness gives each mode its own namespace, so runs do not
contaminate each other (`eval_locomo_llm_free`, `eval_locomo_llm_assisted`).
The harness sets `GRAPHKNOWS_NAMESPACE` for the server it drives, which is what
enables the fast tuning loop:

```bash
python -m evaluation --limit 20                   # ingest once into eval_locomo_llm_free
python -m evaluation --limit 20 --no-ingest       # tweak retrieval, reuse the graph
python -m evaluation --reset                      # drop this mode's namespace, clean re-ingest
```

See [`evaluation/README.md`](../evaluation/README.md).
