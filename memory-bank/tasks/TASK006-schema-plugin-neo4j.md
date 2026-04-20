# TASK006 — Phase 3: Schema.py + Plugin System + Neo4j Uploader

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Create a declarative `schema.py` as the single source of truth for all node/edge types. Build a `plugins/` auto-discovery system for extending the schema. Replace `kg/falkordb/` with `kg/neo4j/` using the official async Neo4j driver.

## Thought Process
The current schema is implicit — labels are hard-coded in the uploader, retriever, and index files. The new system makes the schema explicit and the uploader schema-driven. 

Key insight: the uploader should NEVER know about specific labels. It reads from `GraphSchema` to generate Cypher. Adding a label = adding a `NodeSpec` to a plugin. Zero other changes.

Community detection is preserved as `plugins/topics.py` — it registers TOPIC/SUBTOPIC nodes and hooks into `on_entities_extracted` to run Leiden.

## Interface Contracts

```python
class PropertySpec(BaseModel):
    name: str
    type: Literal["string","int","float","bool","datetime","list","dict","vector"]
    required: bool = False
    indexed: bool = False
    vector_dim: int | None = None

class NodeSpec(BaseModel):
    label: str
    properties: list[PropertySpec]
    id_property: str = "id"
    cypher_merge_template: str  # Cypher template, receives `props` dict

class EdgeSpec(BaseModel):
    rel_type: str
    from_label: str
    to_label: str
    properties: list[PropertySpec] = []
    directed: bool = True

class GraphPlugin(ABC):
    name: ClassVar[str]
    @abstractmethod
    def register(self, schema: GraphSchema) -> None: ...
    def on_document_ingested(self, ctx, doc): ...
    def on_chunk_created(self, ctx, chunk): ...
    def on_entities_extracted(self, ctx, chunk_id, ents): ...
    def on_pipeline_complete(self, ctx): ...
```

## Core Schema (DOCUMENT/CHUNK/ENTITY)
```cypher
// Nodes
(:DOCUMENT {id, title, source, file_type, content_hash, created_at, updated_at,
            chunk_count, entity_count, status, error_message})
(:CHUNK    {id, doc_id, content, heading_path, position, token_count, embedding_id})
(:ENTITY   {id, name, label, description, embedding_id,
            degree_centrality, pagerank, community_id})

// Edges  
(:DOCUMENT)-[:CONTAINS]->(:CHUNK)
(:CHUNK)-[:MENTIONS {confidence}]->(:ENTITY)
(:ENTITY)-[:RELATED_TO {relation_type, confidence, chunk_ids}]-(:ENTITY)
```

## Neo4j Migration
- Replace `falkordb` Python client with `neo4j` async driver (`AsyncGraphDatabase.driver`)
- Use parameterized Cypher throughout — NO f-string interpolation into queries
- Lifespan context manager manages driver open/close
- `MERGE` on `id` property for idempotent upserts
- Vector indexes: `CREATE VECTOR INDEX chunk_embedding FOR (c:CHUNK) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}`

## Implementation Plan
- [ ] Create `services/graphgen/src/kg/schema.py` — NodeSpec, EdgeSpec, PropertySpec, GraphSchema, CORE_SCHEMA
- [ ] Create `services/graphgen/src/kg/plugins/__init__.py` — auto-discovery
- [ ] Create `services/graphgen/src/kg/plugins/base.py` — GraphPlugin ABC
- [ ] Create `services/graphgen/src/kg/plugins/topics.py` — community detection (TOPIC/SUBTOPIC)
- [ ] Create `services/graphgen/src/kg/neo4j/__init__.py`
- [ ] Create `services/graphgen/src/kg/neo4j/driver.py` — async singleton + lifespan
- [ ] Create `services/graphgen/src/kg/neo4j/uploader.py` — schema-driven batch uploader
- [ ] Create `services/graphgen/src/kg/neo4j/indexes.py` — auto-generate from schema
- [ ] Update `services/graphgen/src/kg/pipeline/core.py` — new stage sequence with plugin events
- [ ] Update `services/graphgen/src/kg/graph/extraction.py` — read allowed labels from schema
- [ ] Update `services/graphgen/src/kg/config/settings.py` — swap falkordb_* for neo4j_*
- [ ] Delete `services/graphgen/src/kg/falkordb/` directory
- [ ] Delete `services/graphgen/src/kg/graph/schema.py` (replaced)
- [ ] Update `services/graphgen/pyproject.toml` — add neo4j, remove falkordb
- [ ] Audit `community/subcommunities.py` for SEGMENT/EPISODE references

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 6.1 | schema.py (NodeSpec, EdgeSpec, CORE_SCHEMA) | Not Started | 2026-04-20 | |
| 6.2 | plugins/__init__.py auto-discovery | Not Started | 2026-04-20 | |
| 6.3 | plugins/base.py GraphPlugin ABC | Not Started | 2026-04-20 | |
| 6.4 | plugins/topics.py (community detection) | Not Started | 2026-04-20 | |
| 6.5 | neo4j/driver.py async singleton | Not Started | 2026-04-20 | lifespan-managed |
| 6.6 | neo4j/uploader.py schema-driven | Not Started | 2026-04-20 | batch, parameterized |
| 6.7 | neo4j/indexes.py auto-generated | Not Started | 2026-04-20 | |
| 6.8 | pipeline/core.py new stage sequence | Not Started | 2026-04-20 | |
| 6.9 | extraction.py use schema labels | Not Started | 2026-04-20 | |
| 6.10 | settings.py neo4j vars | Not Started | 2026-04-20 | |
| 6.11 | Delete falkordb/ + old schema.py | Not Started | 2026-04-20 | |
| 6.12 | Audit community/subcommunities.py | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.
