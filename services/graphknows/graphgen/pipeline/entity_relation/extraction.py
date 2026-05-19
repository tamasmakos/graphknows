import asyncio
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

import networkx as nx

# --- Graphgen Imports ---
from graphgen.data_types import PipelineContext, ChunkExtractionTask
from graphgen.pipeline.entity_relation.extractors import BaseExtractor, get_extractor
from graphgen.evaluation import summarize_entity_resolution_effects
from graphgen.pipeline.graph_cleaning.canonicalization import classify_surface_form
from graphgen.pipeline.graph_cleaning.canonicalization import normalize_surface_form
from graphgen.pipeline.entity_relation.ontology import (
    build_top_level_label_space,
    refine_entities_with_ontology,
)
from graphgen.utils.utils import standardize_label
# We assume resolution is in graph_cleaning as previously found
from graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences
from graphgen.utils.diagnostics import diagnostics_enabled, write_diagnostic_json

from gliner2 import GLiNER2

logger = logging.getLogger(__name__)

GLINER2_MODEL = None

def get_gliner2_model(config: Optional[Dict[str, Any]] = None):
    global GLINER2_MODEL
    if GLINER2_MODEL is None:
        try:
            model_name = "fastino/gliner2-base-v1"
            if config:
                extraction_cfg = config.get('extraction', {})
                if hasattr(extraction_cfg, 'model_dump'):
                    extraction_cfg = extraction_cfg.model_dump()
                model_name = extraction_cfg.get('gliner_model', model_name)
            logger.info(f"Loading GLiNER2 model ({model_name})...")
            GLINER2_MODEL = GLiNER2.from_pretrained(model_name)
            logger.info("GLiNER2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER2: {e}")
            raise RuntimeError(
                f"GLiNER2 model could not be loaded ({type(e).__name__}: {e}). "
                "Check RAM availability — model requires ~2 GB. No silent fallback."
            ) from e
    return GLINER2_MODEL

# --- Helper Functions ---

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
    return extraction_cfg.get('max_concurrent_chunks', default)

async def extract_relations_with_llm_async(
    text: str,
    extractor: BaseExtractor,
    keywords: List[str] = None,
    entities: List[str] = None,
    abstract_concepts: List[str] = None,
    allowed_relation_types: Optional[List[str]] = None,
    candidate_relations: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
    """Extract relations using configured extractor."""
    if not extractor or not text:
        return [], [], {}

    try:
        result = await extractor.extract_relations(
            text=text,
            keywords=keywords,
            entities=entities,
            abstract_concepts=abstract_concepts,
            allowed_relation_types=allowed_relation_types,
            candidate_relations=candidate_relations,
        )
        if isinstance(result, tuple):
            if len(result) == 3:
                relations, nodes, diagnostics = result
                return relations or [], nodes or [], diagnostics or {}
            if len(result) == 2:
                relations, nodes = result
                return relations or [], nodes or [], {}
        logger.warning("Extractor returned unexpected result shape; treating as empty extraction")
        return [], [], {}
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"extract_relations_with_llm_async failed: {e}") from e


async def _extract_entities_for_chunk(
    text: str,
    config: Dict[str, Any],
    labels: List[str],
    return_diagnostics: bool = False,
    taxonomy: Optional[Any] = None,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run GLiNER2 NER for a single chunk via the Schema API.

    Uses ``model.create_schema()`` → ``model.extract()`` to perform
    taxonomy-guided entity recognition and candidate relation extraction in a
    single model forward pass.

    The returned diagnostics dict always contains a ``candidate_relations`` key
    so callers can forward them to the DSPy verification pass.
    """
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    diagnostics: Dict[str, Any] = {
        'backend': 'gliner2',
        'label_count': len(labels),
        'label_sample': labels[:20],
        'candidate_relations': [],
    }

    try:
        model = get_gliner2_model(config)
        if not model:
            raise RuntimeError(
                "GLiNER2 model failed to load (OOM or missing weights). "
                "Check available RAM — model requires ~2 GB."
            )

        threshold = extraction_config.get('gliner_threshold', 0.5)

        if taxonomy is None:
            raise RuntimeError(
                "No taxonomy supplied to _extract_entities_for_chunk. "
                "GLiNER2 Schema API requires a loaded TaxonomyConfig."
            )

        descriptions = {
            k: v.description.strip()
            for k, v in taxonomy.entity_groups.items()
            if hasattr(v, 'description') and v.description.strip()
        }
        thresholds = {
            k: v.min_ner_confidence
            for k, v in taxonomy.entity_groups.items()
            if hasattr(v, 'min_ner_confidence')
        }
        entity_spec = {
            label.lower(): {
                "description": descriptions.get(label, label.lower()),
                "threshold": thresholds.get(label) or threshold,
            }
            for label in taxonomy.entity_groups
        }
        gliner_schema = model.create_schema()
        gliner_schema.entities(entity_spec)

        if hasattr(taxonomy, 'relation_types') and taxonomy.relation_types:
            rel_descs: Dict[str, str] = {}
            for rt in taxonomy.relation_types:
                if isinstance(rt, dict):
                    rel_descs.update(rt)
                else:
                    rel_descs[str(rt)] = str(rt).replace("_", " ").lower()
            if rel_descs:
                gliner_schema.relations(rel_descs)

        raw = await asyncio.to_thread(
            model.extract,
            text,
            gliner_schema,
            threshold,
            True,   # format_results
            True,   # include_confidence
            True,   # include_spans
        )

        child_to_group: Dict[str, str] = {}
        for group_label, group_cfg in taxonomy.entity_groups.items():
            child_to_group[group_label.lower()] = group_label
            for child in getattr(group_cfg, 'children', []):
                child_to_group[child.lower()] = group_label

        flat_entities: List[Dict[str, Any]] = []
        for raw_label, spans in (raw.get('entities') or {}).items():
            canonical = child_to_group.get(raw_label.lower(), raw_label.upper())
            for span in spans or []:
                flat_entities.append({
                    'text': span.get('text', ''),
                    'label': standardize_label(canonical),
                    'ontology_label': standardize_label(canonical),
                    'confidence': float(span.get('confidence', 0.0)),
                    'start': span.get('start'),
                    'end': span.get('end'),
                    'backend': 'gliner2',
                })

        candidate_relations: List[Dict[str, Any]] = []
        for rel_type, triplets in (raw.get('relation_extraction') or {}).items():
            for triplet in triplets or []:
                head = triplet.get('head', {})
                tail = triplet.get('tail', {})
                src_text = head.get('text', '')
                tgt_text = tail.get('text', '')
                if not src_text or not tgt_text or src_text == tgt_text:
                    continue
                candidate_relations.append({
                    'source': src_text,
                    'relation': rel_type.upper(),
                    'target': tgt_text,
                    'confidence': min(
                        float(head.get('confidence', 0.0)),
                        float(tail.get('confidence', 0.0)),
                    ),
                    'extractor': 'gliner2',
                })

        label_space = build_top_level_label_space(labels or [], taxonomy=taxonomy)
        refined_entities = refine_entities_with_ontology(flat_entities, label_space, taxonomy=taxonomy)
        diagnostics.update({
            'candidate_relations': candidate_relations,
            'raw_entities': flat_entities,
            'entities': refined_entities,
            'threshold': threshold,
            'schema_mode': 'taxonomy',
        })
        return (refined_entities, diagnostics) if return_diagnostics else refined_entities

    except Exception as e:
        logger.error(f"GLiNER2 extraction failed for chunk: {e}")
        return ([], diagnostics) if return_diagnostics else []


def _build_relation_eligible_entities(
    gliner_entities: List[Dict[str, Any]],
    taxonomy: Any = None,
) -> List[str]:
    """Filter and rank NER entities for relation-extraction eligibility.

    Applies per-label minimum confidence thresholds from *taxonomy* when
    provided; otherwise uses the built-in thresholds.  The PERSON
    underscore-check (requiring a multi-token name) is a NLP heuristic kept
    in code regardless of taxonomy.
    """
    # Determine the set of known entity labels for the short-name heuristic
    if taxonomy is not None:
        known_labels = set(taxonomy.entity_groups.keys())
    else:
        known_labels = {'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'POLICY'}

    def _min_conf(label: str) -> float:
        if taxonomy is not None:
            return taxonomy.get_min_ner_confidence(label)
        # built-in thresholds
        return {'EVENT': 0.5, 'ORGANIZATION': 0.6, 'LOCATION': 0.6}.get(label, 0.0)

    eligible = []
    fallback = []

    for entity in gliner_entities or []:
        text = standardize_label(entity.get('text', ''))
        ontology_label = standardize_label(entity.get('ontology_label') or entity.get('label', ''))
        confidence = float(entity.get('confidence', 0.0) or 0.0)

        if not text:
            continue

        fallback.append(text)

        # Heuristic: skip very short single-token names for entity-type labels
        if len(text) <= 3 and '_' not in text and ontology_label in known_labels:
            continue
        # Heuristic: PERSON must be multi-token (e.g. URSULA_VON_DER_LEYEN, not just LEYEN)
        if ontology_label == 'PERSON' and '_' not in text:
            continue
        # Taxonomy-driven confidence gate
        if confidence < _min_conf(ontology_label):
            continue

        eligible.append(text)

    deduped_eligible = list(dict.fromkeys(eligible))
    if deduped_eligible:
        return deduped_eligible
    return list(dict.fromkeys(fallback))

def _build_entity_surface_metadata(gliner_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata = []
    for entity in gliner_entities or []:
        text = entity.get('text', '')
        metadata.append({
            'text': text,
            'canonical_form': normalize_surface_form(text),
            'surface_form_class': classify_surface_form(text),
            'ontology_label': entity.get('ontology_label') or entity.get('label'),
            'confidence': entity.get('confidence'),
            'merge_candidate': classify_surface_form(text) == 'named_entity',
        })
    return metadata


async def process_extraction_task(
    deps: PipelineContext,
    task: ChunkExtractionTask,
    semaphore: asyncio.Semaphore,
    extractor: BaseExtractor,
    config: Dict[str, Any],
    ontology_labels: List[str]
) -> Dict[str, Any]:
    """Process a single chunk extraction task"""
    async with semaphore:
        # Use full chunk_id to avoid confusion with similar short IDs
        logger.debug(f"      Starting {task.chunk_id}")
        
        try:
            # 1. Load taxonomy early — needed by Schema API and DSPy gate.
            _taxonomy = None
            try:
                from graphgen.config.taxonomy import get_default_taxonomy, load_taxonomy
                extraction_config_inner = config.get('extraction', {})
                if hasattr(extraction_config_inner, 'model_dump'):
                    extraction_config_inner = extraction_config_inner.model_dump()
                _taxonomy_file = extraction_config_inner.get('taxonomy_file')
                _taxonomy = load_taxonomy(_taxonomy_file) if _taxonomy_file else get_default_taxonomy()
            except Exception:
                pass  # non-fatal; pipeline degrades to legacy label-space path

            # 2. Run NER for this chunk.
            # Always request diagnostics so candidate_relations from the
            # GLiNER2 Schema pass can be retrieved regardless of the disk-write flag.
            gliner_entities, ner_diagnostics = await _extract_entities_for_chunk(
                task.chunk_text,
                config,
                ontology_labels,
                return_diagnostics=True,
                taxonomy=_taxonomy,
            )

            # 3. Candidate relations produced by GLiNER2 Schema pass.
            candidate_relations: List[Dict[str, Any]] = (
                ner_diagnostics.get('candidate_relations', []) if ner_diagnostics else []
            )

            # 4. Allowed relation types from taxonomy (DSPy gate).
            allowed_relation_types: List[str] = (
                list(_taxonomy.relation_types) if _taxonomy and _taxonomy.relation_types else []
            )

            # 5. Prepare allowed nodes/types for LLM (Gatekeeper Logic)
            found_labels = set()
            for ent in gliner_entities:
                label = ent.get('ontology_label') or ent.get('label')
                if label:
                    found_labels.add(standardize_label(label))

            node_labels = list(found_labels)
            if not node_labels:
                logger.debug(f"Chunk {task.chunk_id}: Found labels: []")
            else:
                logger.debug(f"Chunk {task.chunk_id}: Found labels: {node_labels}")

            if not node_labels and ontology_labels:
                node_labels = ontology_labels
                logger.debug(f"Chunk {task.chunk_id}: No labels found by NER, using full ontology set as fallback")

            # 6. Build entity hints (relation-eligible, deduplicated)
            discovered_entities = []
            if task.entities:
                discovered_entities.extend(task.entities)
            relation_eligible_entities = (
                _build_relation_eligible_entities(gliner_entities, taxonomy=_taxonomy)
                if gliner_entities else []
            )
            if gliner_entities:
                discovered_entities.extend(relation_eligible_entities)
            discovered_entities = list(dict.fromkeys(discovered_entities))

            entity_surface_metadata = _build_entity_surface_metadata(gliner_entities)

            chunk_diag_payload = None
            chunk_diag_path = None
            if diagnostics_enabled(config):
                chunk_diag_payload = {
                    'chunk_id': task.chunk_id,
                    'chunk_text': task.chunk_text,
                    'keywords': task.keywords,
                    'task_entities': task.entities,
                    'ontology_labels': ontology_labels,
                    'ner': ner_diagnostics,
                    'entity_surface_metadata': entity_surface_metadata,
                    'found_labels': sorted(found_labels),
                    'node_labels': node_labels,
                    'relation_eligible_entities': relation_eligible_entities,
                    'discovered_entities': sorted(discovered_entities),
                    'candidate_relations': candidate_relations,
                    'allowed_relation_types': allowed_relation_types,
                }

            # Run LLM Relation Extraction
            raw_relations, raw_nodes, relation_diagnostics = await extract_relations_with_llm_async(
                text=task.chunk_text,
                extractor=extractor,
                keywords=task.keywords,
                entities=discovered_entities,
                abstract_concepts=node_labels,
                allowed_relation_types=allowed_relation_types,
                candidate_relations=candidate_relations,
            )

            if chunk_diag_payload is not None:
                chunk_diag_payload.update({
                    'raw_relations': raw_relations,
                    'raw_nodes': raw_nodes,
                    'accepted_relations': raw_relations,
                    'accepted_nodes': raw_nodes,
                    'relation_decisions': relation_diagnostics.get('triplet_decisions', []),
                    'raw_triplets': relation_diagnostics.get('raw_triplets', []),
                    'relation_diagnostics_available': bool(relation_diagnostics),
                })
                chunk_diag_path = write_diagnostic_json(config, f'extraction_chunk_{task.chunk_id}', chunk_diag_payload)
                deps.diagnostics.setdefault('chunk_diagnostics', []).append(chunk_diag_path)
            
            chunk_data = {
                'knowledge_triplets': raw_relations,
                'raw_extraction': {
                    'relations': raw_relations,
                    'nodes': raw_nodes
                },
                'gliner_entities': gliner_entities
            }
            if diagnostics_enabled(config):
                chunk_data['diagnostic_snapshot'] = {
                    'found_labels': sorted(found_labels),
                    'node_labels': node_labels,
                    'entity_surface_metadata': entity_surface_metadata,
                    'relation_eligible_entities': relation_eligible_entities,
                    'discovered_entities': sorted(discovered_entities),
                    'raw_relations': raw_relations,
                    'raw_nodes': raw_nodes,
                }
            deps.graph.nodes[task.chunk_id].update(chunk_data)
            
            unique_relation_entities = {
                x for tr in raw_relations for x in (tr[0], tr[2]) if x
            }
            unique_gliner_entities = {
                e.get('text') for e in gliner_entities if e.get('text')
            }
            rel_count = len(raw_relations)
            ent_count = len(unique_relation_entities or unique_gliner_entities)
            deps.graph.nodes[task.chunk_id]['extraction_successful'] = bool(raw_relations or gliner_entities)
            logger.debug(f"      Completed {task.chunk_id}: stored {ent_count} entities, {rel_count} relations")
            
            return {
                "success": True, 
                "chunk_id": task.chunk_id,
                "entity_count": ent_count,
                "relation_count": rel_count
            }
            
        except Exception as e:
            logger.error(f"      Failed {task.chunk_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "chunk_id": task.chunk_id}

async def extract_all_entities_relations(deps: PipelineContext, config: Dict[str, Any], extractor: BaseExtractor = None) -> Dict[str, Any]:
    """Phase 2: Parallel entity/relation extraction"""
    if not deps.extraction_tasks:
        return {"processed": 0, "successful": 0, "errors": []}
    
    # Deduplicate tasks
    seen_chunk_ids = set()
    unique_tasks = []
    for task in deps.extraction_tasks:
        if task.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(task.chunk_id)
            unique_tasks.append(task)
    
    if len(unique_tasks) < len(deps.extraction_tasks):
        logger.warning(f"Removed {len(deps.extraction_tasks) - len(unique_tasks)} duplicate extraction tasks")
    
    should_close_extractor = False
    if extractor is None:
        extractor = get_extractor(config)
        should_close_extractor = True
        
    logger.info(f"Using {extractor.__class__.__name__} for relation extraction")
    
    # Resolve ontology labels ONCE
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    from graphgen.utils.labels import resolve_entity_labels
    labels = resolve_entity_labels(extraction_cfg)
    logger.info(f"Using {len(labels)} entity labels for extraction pipeline. Processing {len(unique_tasks)} chunks in parallel...")

    # Preload GLiNER2 model to prevent race conditions in parallel threads
    if extraction_cfg.get('gliner_preload', True):
        get_gliner2_model(config)
    else:
        logger.info("Skipping eager GLiNER2 preload because extraction.gliner_preload is false.")

    try:
        # Use max_concurrent from config for controlled parallelization
        max_concurrent = get_max_concurrent(config, default=8)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [process_extraction_task(
            deps, 
            task, 
            semaphore, 
            extractor, 
            config,
            labels
        ) for task in unique_tasks]
        
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r.get("success"))
        errors = [r.get("error") for r in results if not r.get("success") and r.get("error")]
        
        logger.info(f"Extraction complete: {successful}/{len(results)} successful")
        
        # Enrich graph per segment
        enrich_result = await enrich_graph_per_segment(deps)
        
        total_entities = sum(r.get("entity_count", 0) for r in results if r.get("success"))
        total_relations = sum(r.get("relation_count", 0) for r in results if r.get("success"))

        return {
            "processed": len(results),
            "successful": successful,
            "total_entities_extracted": total_entities,
            "total_relations_extracted": total_relations,
            "errors": errors + enrich_result.get("errors", [])
        }
    finally:
        if should_close_extractor and hasattr(extractor, 'close'):
            await extractor.close()

# --- Graph Enrichment ---


def _get_chunks_for_segment(graph: nx.DiGraph, segment_id: str) -> List[str]:
    """Find all chunks associated with a segment, including via Conversations and Contexts."""
    chunk_ids = set()
    
    # 1. Direct Chunks (old schema)
    for neighbor in graph.neighbors(segment_id):
        edge = graph.get_edge_data(segment_id, neighbor) or {}
        # Relaxed check: Accept TextChunk or CHUNK, or just trust the edge
        neighbor_type = graph.nodes[neighbor].get('node_type', '')
        if edge.get('label') == 'HAS_CHUNK' and neighbor_type in ['CHUNK', 'TextChunk', 'Chunk']:
            chunk_ids.add(neighbor)
            
    # 2. Via Conversations
    # Segment -> HAS_CONVERSATION -> Conversation
    for neighbor in graph.neighbors(segment_id):
        edge = graph.get_edge_data(segment_id, neighbor) or {}
        if edge.get('label') == 'HAS_CONVERSATION':
            conv_id = neighbor
            # Conversation -> HAS_CHUNK -> Chunk
            for conv_neighbor in graph.neighbors(conv_id):
                conv_edge = graph.get_edge_data(conv_id, conv_neighbor) or {}
                if conv_edge.get('label') == 'HAS_CHUNK' and graph.nodes[conv_neighbor].get('node_type') == 'CHUNK':
                    chunk_ids.add(conv_neighbor)
                
                # Conversation -> HAS_CONTEXT -> Context -> HAS_DESCRIPTION_CHUNK -> Chunk
                if conv_edge.get('label') == 'HAS_CONTEXT':
                    ctx_id = conv_neighbor
                    for ctx_neighbor in graph.neighbors(ctx_id):
                        ctx_edge = graph.get_edge_data(ctx_id, ctx_neighbor) or {}
                        if ctx_edge.get('label') == 'HAS_DESCRIPTION_CHUNK' and graph.nodes[ctx_neighbor].get('node_type') == 'CHUNK':
                            chunk_ids.add(ctx_neighbor)

    return list(chunk_ids)

async def add_triplets_to_graph_for_segment(
    deps: PipelineContext,
    relations: List[Tuple[str, str, str, Dict[str, Any]]],
    entity_mappings: Dict[str, str],
    segment_id: str,
    chunk_entity_map: Dict[str, Set[str]],
    gliner_label_map: Dict[str, str] = None,  # New parameter
    llm_type_map: Dict[str, str] = None # Map of entity -> llm_type
):
    """Write entities/edges to graph for a segment"""
    graph = deps.graph
    gliner_label_map = gliner_label_map or {}
    llm_type_map = llm_type_map or {}
    
    # Add entities
    for _, mapped_ent in entity_mappings.items():
        # Classify entity using LLM type map or GLiNER map
        # LLM type is highly reliable now as it follows ontology strict_mode
        llm_type = llm_type_map.get(mapped_ent)
        gliner_label_entry = gliner_label_map.get(mapped_ent)
        if not gliner_label_entry:
            gliner_label_entry = gliner_label_map.get(mapped_ent.lower())
        gliner_label = gliner_label_entry.get('label') if isinstance(gliner_label_entry, dict) else gliner_label_entry
            
        # Use LLM type if available, fallback to GLiNER label, then default
        ontology_label = llm_type or gliner_label or "UNKNOWN"
            
        if not graph.has_node(mapped_ent):
            # Create new node
            graph.add_node(mapped_ent, 
                         node_type="ENTITY_CONCEPT", 
                         ontology_class=ontology_label,
                         llm_type=llm_type,
                         name=mapped_ent, 
                         graph_type="entity_relation")
        else:
            # Update existing node
            node_data = graph.nodes[mapped_ent]
            
            # Ensure name is set
            if 'name' not in node_data:
                node_data['name'] = mapped_ent
                
            # Update llm_type if available
            if llm_type:
                node_data['llm_type'] = llm_type
                
            # Smart update for ontology_class: prioritize LLM type, then GLiNER
            if llm_type:
                node_data['ontology_class'] = llm_type
            elif not node_data.get('ontology_class') or node_data.get('ontology_class') in ('Concept', 'UNKNOWN'):
                if gliner_label:
                    node_data['ontology_class'] = gliner_label
                elif 'ontology_class' not in node_data:
                    node_data['ontology_class'] = 'UNKNOWN'
                
            # Ensure node_type is set (if it was created implicitly)
            if 'node_type' not in node_data:
                node_data['node_type'] = "ENTITY_CONCEPT"
                node_data['graph_type'] = "entity_relation"
 
    
    # Add relations
    for h, r, t, props in relations:
        if not graph.has_node(h) or not graph.has_node(t):
            continue
            
        # Props usually contains confidence, evidence
        edge_attrs = {
            "label": r, 
            "relation_type": r, 
            "graph_type": "entity_relation", 
            "segment_id": segment_id, 
            "source": "extraction"
        }
        if props:
            edge_attrs.update(props)
            
        graph.add_edge(h, t, **edge_attrs)

    # Link chunks to entities
    for chunk_id, entities in chunk_entity_map.items():
        for ent in entities:
            mapped = entity_mappings.get(ent, ent)
            if graph.has_node(mapped):
                graph.add_edge(chunk_id, mapped, label="HAS_ENTITY", graph_type="lexical_graph")

async def enrich_graph_per_segment(deps: PipelineContext) -> Dict[str, Any]:
    """Aggregate chunk-level extractions per segment, run coref, and enrich graph."""
    graph = deps.graph
    # Text-subdivision nodes are typed 'PASSAGE' after E6 rename (was 'SEGMENT').
    # Temporal pipeline-run containers are typed 'SEGMENT' and must not be processed here.
    segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'PASSAGE']

    # Fallback to Document nodes if no PASSAGE nodes found (e.g. simplified schema)
    if not segment_nodes:
        segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') in ['Document', 'DAY', 'Doc', 'DOC']]
        if segment_nodes:
            logger.info(f"Enriching graph using {len(segment_nodes)} Document nodes (no SEGMENT nodes found)")

    segments_processed = 0
    errors = []
    
    for segment_id in segment_nodes:
        try:
            chunk_ids = _get_chunks_for_segment(graph, segment_id)
            if not chunk_ids:
                continue
            
            aggregated_relations = []
            chunk_entity_map = {}
            all_entities = set()
            
            # Collect GLiNER labels from all chunks
            gliner_label_map = {}
            
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_extraction = node.get('raw_extraction') or {}
                raw_relations = raw_extraction.get('relations') or []
                raw_nodes = raw_extraction.get('nodes') or []
                
                gliner_entities = node.get('gliner_entities', [])
                
                # Populate GLiNER map
                for entity in gliner_entities:
                    text = standardize_label(entity.get('text', ''))
                    ontology_label = standardize_label(entity.get('ontology_label') or entity.get('label', ''))
                    if text and ontology_label:
                        current = gliner_label_map.get(text)
                        current_conf = 0.0 if not isinstance(current, dict) else float(current.get('confidence', 0.0) or 0.0)
                        new_conf = float(entity.get('confidence', 0.0) or 0.0)
                        if current is None or new_conf >= current_conf:
                            gliner_label_map[text] = {
                                'label': ontology_label,
                                'confidence': new_conf,
                                'backend': entity.get('backend'),
                                'start': entity.get('start'),
                                'end': entity.get('end'),
                                'candidate_labels': entity.get('candidate_labels', []),
                            }
                
                entities_in_chunk = set()
                for (h, r, t, _) in raw_relations:
                    aggregated_relations.append((h, r, t, _))
                    entities_in_chunk.add(h)
                    entities_in_chunk.add(t)
                
                # Also add entities found by GLiNER even if no relations found
                if gliner_entities:
                     for entity in gliner_entities:
                        text = entity.get('text', '').strip()
                        if text:
                            entities_in_chunk.add(text)
                
                if not entities_in_chunk:
                    initial_ents = node.get('initial_entities') or []
                    entities_in_chunk.update([standardize_label(e) for e in initial_ents])
                
                if entities_in_chunk:
                    chunk_entity_map[cid] = entities_in_chunk
                    all_entities.update(entities_in_chunk)
            
            if not aggregated_relations and not all_entities:
                continue

            coref_result = resolve_extraction_coreferences(aggregated_relations, list(all_entities))
            cleaned_relations = coref_result.get('cleaned_relations', [])
            entity_mappings = coref_result.get('entity_mappings', {})

            if graph.nodes.get(segment_id, {}).get('diagnostic_snapshot') is None:
                graph.nodes[segment_id]['diagnostic_snapshot'] = {}
            segment_diag = None
            if diagnostics_enabled(deps.stats.get('pipeline_config', {})):
                segment_diag = {
                    'segment_id': segment_id,
                    'chunk_ids': chunk_ids,
                    'aggregated_relations': aggregated_relations,
                    'all_entities': sorted(all_entities),
                    'chunk_entity_map': {k: sorted(v) for k, v in chunk_entity_map.items()},
                    'gliner_label_map': gliner_label_map,
                    'coref_result': coref_result,
                    'cleaned_relations': cleaned_relations,
                    'entity_mappings': entity_mappings,
                }
            
            # Build LLM type map
            llm_type_map = {}
            
            # Re-iterating to be safe and simple
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_nodes = (node.get('raw_extraction') or {}).get('nodes') or []
                for n in raw_nodes:
                    ent_id = n.get('id')
                    ent_type = n.get('type')
                    if ent_id and ent_type:
                        llm_type_map[ent_id] = ent_type
                        
            canonical_llm_type_map = {}
            for ent_id, ent_type in llm_type_map.items():
                # ent_id is the raw extracted name
                canonical = entity_mappings.get(ent_id, ent_id)
                if canonical:
                    canonical_llm_type_map[canonical] = ent_type

            await add_triplets_to_graph_for_segment(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                segment_id=segment_id,
                chunk_entity_map=chunk_entity_map,
                gliner_label_map=gliner_label_map, # Pass the map
                llm_type_map=canonical_llm_type_map
            )

            if diagnostics_enabled(deps.stats.get('pipeline_config', {})):
                final_entities = []
                final_relations = []
                entity_ids = set(entity_mappings.values())
                for node_id in entity_ids:
                    if graph.has_node(node_id):
                        node_data = graph.nodes[node_id]
                        final_entities.append({
                            'id': node_id,
                            'name': node_data.get('name', node_id),
                            'node_type': node_data.get('node_type'),
                            'ontology_class': node_data.get('ontology_class'),
                            'llm_type': node_data.get('llm_type'),
                            'graph_type': node_data.get('graph_type'),
                        })
                for source, target, edge_data in graph.edges(data=True):
                    if edge_data.get('graph_type') == 'entity_relation' and edge_data.get('segment_id') == segment_id:
                        final_relations.append({
                            'source': source,
                            'target': target,
                            'label': edge_data.get('label'),
                            'relation_type': edge_data.get('relation_type'),
                            'properties': {
                                key: value
                                for key, value in edge_data.items()
                                if key not in {'label', 'relation_type', 'graph_type', 'segment_id', 'source'}
                            },
                        })
                segment_diag['final_entities'] = sorted(final_entities, key=lambda item: item['id'])
                segment_diag['final_relations'] = sorted(final_relations, key=lambda item: (item['source'], item['label'], item['target']))
                diag_path = write_diagnostic_json(deps.stats['pipeline_config'], f'segment_{segment_id}', segment_diag)
                deps.diagnostics.setdefault('segment_diagnostics', []).append(diag_path)
            
            segments_processed += 1
        except Exception as e:
            logger.warning(f"Segment-level enrichment failed for {segment_id}: {e}")
            errors.append(f"{segment_id}: {e}")
            continue
    
    return {"segments_processed": segments_processed, "errors": errors}
