# Memgraph Setup for Graphknows

## Start Memgraph (run on your host machine)

```bash
docker run -d \
  --name graphknows-memgraph \
  -p 17687:7687 \
  -p 17444:3000 \
  memgraph/memgraph-mage:latest
```

## Verify it's running

```bash
docker ps | grep memgraph
curl http://localhost:17444  # Memgraph Lab UI
```

## Connect from hermes container

The agent probes these URIs in order (2 s timeout each):

- `bolt://172.17.0.1:17687` — default Docker bridge gateway (host machine from inside container)
- `bolt://172.18.0.1:17687` — graphknows Docker network gateway

If neither is reachable the agent sets `_memgraph_ok = False` and falls back to a
lightweight **in-memory session store** — all reads/writes silently use that store
instead. No errors are surfaced to callers.

### Troubleshooting

```bash
# Confirm the host gateway IP from inside the container
ip route | grep default   # typically 172.17.0.1 or 172.18.0.1

# Quick connectivity test (run inside the hermes container)
python3 -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://172.17.0.1:17687', auth=None, connection_timeout=2)
with d.session() as s:
    s.run('RETURN 1').consume()
print('Memgraph reachable')
"
```

## What you get once running

- Per-session context graphs grow as you query
- Agent upserts `Entity`, `Chunk`, `ReasoningStep`, and `Topic` nodes per conversation turn
- Edges: `MENTIONS`, `RELATES_TO`, `CO_OCCURS`, `RETRIEVED`, `IN_COMMUNITY`
- Frontend graph panel shows only entities relevant to **YOUR** session (`session_id` property scopes every node)
- Traverse between sessions is isolated — each session is a fully independent subgraph

## Graceful degradation (no Memgraph)

When Memgraph is not running:

- `create_session()` → initialises in-memory dict, returns `session_id`
- `get_session_graph()` → returns `{"nodes": [], "edges": []}` (empty until seeded)
- `seed_from_neo4j()` → pulls from Neo4j and stores in-memory only
- `add_reasoning_step()` / `add_graph_enrichments()` / `add_cooccurs_edges()` → in-memory only
- `write_back_to_neo4j()` → no-op (returns 0)
- The agent and frontend continue to work — graph panel starts empty and populates from
  the in-memory store as queries are made

## Frontend response format

`GET /session/{session_id}/graph` returns:

```json
{
  "nodes": [
    {
      "id": "entity-InflationRate-abc123",
      "label": "Entity",
      "type": "Entity",
      "name": "InflationRate",
      "properties": {"name": "InflationRate", "label": "Entity", "session_id": "abc123"}
    }
  ],
  "edges": [
    {
      "id": "chunk-c1-abc123-MENTIONS-entity-InflationRate-abc123",
      "source": "chunk-c1-abc123",
      "target": "entity-InflationRate-abc123",
      "type": "MENTIONS",
      "properties": {}
    }
  ]
}
```

Every edge carries an `id`, `source`, `target`, `type`, and `properties` — matching
the D3.js force-graph schema expected by the frontend.
