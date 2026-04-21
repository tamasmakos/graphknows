import asyncio
import functools
import logging
import re
from typing import List, Dict, Any, Tuple, Set

import networkx as nx
from kg.types import PipelineContext, ChunkExtractionTask
from kg.graph.resolution import resolve_extraction_coreferences
from kg.graph.extractors import BaseExtractor, get_extractor

# --- GLiNER Helper ---
from gliner import GLiNER
import torch

logger = logging.getLogger(__name__)

GLINER_MODEL = None

def get_gliner_model():
    global GLINER_MODEL
    if GLINER_MODEL is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading GLiNER model (urchade/gliner_medium-v2.1) on {device}...")
            GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", device=device)
            logger.info(f"GLiNER model loaded on {device}.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER: {e}")
            return None
    return GLINER_MODEL

# --- Helper Functions ---

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    return extraction_cfg.get('max_concurrent_chunks', default)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Simple regex for sentence splitting
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]

# --- Async Extraction ---

async def extract_relations_with_llm_async(
    text: str,
    extractor: BaseExtractor,
    keywords: List[str] = None,
    entities: List[str] = None,
    abstract_concepts: List[str] = None
) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    """Extract relations using configured extractor."""
    if not extractor or not text:
        return [], []

    try:
        return await extractor.extract_relations(
            text=text,
            keywords=keywords,
            entities=entities,
            abstract_concepts=abstract_concepts
        )
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return [], []

async def process_extraction_task(
    deps: PipelineContext,
    task: ChunkExtractionTask,
    semaphore: asyncio.Semaphore,
    extractor: BaseExtractor,
    precomputed_gliner_entities: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a single chunk extraction task"""
    async with semaphore:
        # Use full chunk_id to avoid confusion with similar short IDs
        logger.info(f"      🚀 Starting {task.chunk_id}")
        
        try:
            # 1. Use precomputed GLiNER entities or fallback to empty
            gliner_entities = precomputed_gliner_entities or []

            # 2. Run LLM Relation Extraction
            # Use chunk text directly for extraction, pass keywords as allowed nodes
            # Note: We use keywords as both entities and concepts for simplicity,             
            # Combine all available "allowed node" sources
            allowed_nodes = []
            if task.keywords:
                allowed_nodes.extend(task.keywords)
            if task.entities:
                allowed_nodes.extend(task.entities)
            if task.abstract_concepts:
                allowed_nodes.extend(task.abstract_concepts)
            
            # Add GLINER entities to allowed_nodes
            if gliner_entities:
                gliner_texts = [e.get('text') for e in gliner_entities if e.get('text')]
                allowed_nodes.extend(gliner_texts)
            
            # Deduplicate
            allowed_nodes = list(set(allowed_nodes))
            
            raw_relations, raw_nodes = await extract_relations_with_llm_async(
                text=task.chunk_text,
                extractor=extractor,
                keywords=task.keywords,
                entities=allowed_nodes, # Pass as entities for LangChain extractor
                abstract_concepts=[] # Already merged into entities
            )
            
            chunk_data = {
                'knowledge_triplets': raw_relations,
                'raw_extraction': {
                    'relations': raw_relations,
                    'nodes': raw_nodes
                },
                'gliner_entities': gliner_entities  # Store GLiNER predictions
            }
            deps.graph.nodes[task.chunk_id].update(chunk_data)
            
            deps.graph.nodes[task.chunk_id]['extraction_successful'] = bool(raw_relations)
            rel_count = len(raw_relations)
            ent_count = len(set([x for tr in raw_relations for x in (tr[0], tr[2])])) if raw_relations else 0
            logger.info(f"      ✅ Completed {task.chunk_id}: stored {ent_count} entities, {rel_count} relations")
            
            return {"success": True, "chunk_id": task.chunk_id}
            
        except Exception as e:
            logger.error(f"      ❌ Failed {task.chunk_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "chunk_id": task.chunk_id}

async def _generate_entity_hints(
    tasks: List[ChunkExtractionTask], 
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, str]]]:
    """Generate entity hints using GLiNER."""
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    results_map = {}

    # --- GLiNER Extraction ---
    logger.info(f"Step 2.1: Bulk GLiNER Pass for {len(tasks)} chunks...")
    try:
        model = get_gliner_model()
        if model:
            labels = extraction_config.get('gliner_labels', ["Person", "Organization", "Location", "Event", "Date", "Award", "Competitions", "Teams", "Concept"])
            
            all_sentences = []
            sentence_to_task_map = []
            
            for task in tasks:
                sents = split_sentences(task.chunk_text)
                if not sents:
                    sents = [task.chunk_text]
                all_sentences.extend(sents)
                sentence_to_task_map.extend([task.chunk_id] * len(sents))
            
            if all_sentences:
                predict_func = functools.partial(
                    model.batch_predict_entities, 
                    all_sentences, 
                    labels, 
                    threshold=0.5
                )
                all_predictions = await asyncio.to_thread(predict_func)
                
                for chunk_id, preds in zip(sentence_to_task_map, all_predictions):
                    if chunk_id not in results_map:
                        results_map[chunk_id] = []
                    results_map[chunk_id].extend(preds)
                
                logger.info(f"Bulk GLiNER complete. Extracted entities for {len(results_map)} chunks.")
    except Exception as e:
        logger.error(f"Bulk GLiNER extraction failed: {e}")

    return results_map

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
    
    # Generate Entity Hints
    gliner_results_map = await _generate_entity_hints(unique_tasks, config)

    try:
        # Use max_concurrent from config for controlled parallelization
        max_concurrent = get_max_concurrent(config, default=8)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [process_extraction_task(deps, task, semaphore, extractor, gliner_results_map.get(task.chunk_id, [])) for task in unique_tasks]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r.get("success"))
        errors = [r.get("error") for r in results if not r.get("success") and r.get("error")]
        
        logger.info(f"Extraction complete: {successful}/{len(results)} successful")
        
        # Enrich graph per episode
        enrich_result = await enrich_graph_per_episode(deps)
        
        return {
            "processed": len(results),
            "successful": successful,
            "errors": errors + enrich_result.get("errors", [])
        }
    finally:
        if should_close_extractor and hasattr(extractor, 'close'):
            await extractor.close()

# --- Graph Enrichment ---


def _get_chunks_for_episode(graph: nx.DiGraph, episode_id: str) -> List[str]:
    """Find all chunks associated with an episode."""
    chunk_ids = set()
    
    # Direct Chunks: EPISODE -> HAS_CHUNK -> CHUNK
    for neighbor in graph.neighbors(episode_id):
        edge = graph.get_edge_data(episode_id, neighbor) or {}
        if edge.get('label') == 'HAS_CHUNK' and graph.nodes[neighbor].get('node_type') == 'CHUNK':
            chunk_ids.add(neighbor)
            
    return list(chunk_ids)

async def add_triplets_to_graph_for_episode(
    deps: PipelineContext,
    relations: List[Tuple[str, str, str]],
    entity_mappings: Dict[str, str],
    episode_id: str,
    chunk_entity_map: Dict[str, Set[str]],
    gliner_label_map: Dict[str, str] = None,
    llm_type_map: Dict[str, str] = None
):
    """Write entities/edges to graph for an episode"""
    graph = deps.graph
    gliner_label_map = gliner_label_map or {}
    llm_type_map = llm_type_map or {}
    
    # Add entities
    for _, mapped_ent in entity_mappings.items():
        # Classify entity using GLiNER map
        ontology_label = gliner_label_map.get(mapped_ent)
        if not ontology_label:
            ontology_label = gliner_label_map.get(mapped_ent.lower(), "Concept")
            
        if not graph.has_node(mapped_ent):
            # Create new node
            graph.add_node(mapped_ent, 
                         node_type="ENTITY_CONCEPT", 
                         ontology_class=ontology_label,
                         llm_type=llm_type_map.get(mapped_ent),
                         name=mapped_ent, 
                         graph_type="entity_relation")
        else:
            # Update existing node
            node_data = graph.nodes[mapped_ent]
            
            # Ensure name is set
            if 'name' not in node_data:
                node_data['name'] = mapped_ent
                
            # Update llm_type if available and not set
            if llm_type_map.get(mapped_ent) and not node_data.get('llm_type'):
                node_data['llm_type'] = llm_type_map.get(mapped_ent)
                
            # Smart update for ontology_class
            current_class = node_data.get('ontology_class')
            if not current_class or (current_class == 'Concept' and ontology_label != 'Concept'):
                node_data['ontology_class'] = ontology_label
                
            # Ensure node_type is set (if it was created implicitly)
            if 'node_type' not in node_data:
                node_data['node_type'] = "ENTITY_CONCEPT"
                node_data['graph_type'] = "entity_relation"

    
    # Add relations
    for h, r, t in relations:
        if not graph.has_node(h) or not graph.has_node(t):
            continue
            
        graph.add_edge(h, t, label=r, relation_type=r, graph_type="entity_relation", episode_id=episode_id, source="extraction")

    # Link chunks to entities
    for chunk_id, entities in chunk_entity_map.items():
        for ent in entities:
            mapped = entity_mappings.get(ent, ent)
            if graph.has_node(mapped):
                graph.add_edge(chunk_id, mapped, label="HAS_ENTITY", graph_type="lexical_graph")

async def enrich_graph_per_episode(deps: PipelineContext) -> Dict[str, Any]:
    """Aggregate chunk-level extractions per episode, run coref, and enrich graph."""
    graph = deps.graph
    episode_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'EPISODE']
    episodes_processed = 0
    errors = []
    
    for episode_id in episode_nodes:
        try:
            chunk_ids = _get_chunks_for_episode(graph, episode_id)
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
                    text = entity.get('text', '').strip()
                    label = entity.get('label', '')
                    if text and label:
                        gliner_label_map[text] = label
                        gliner_label_map[text.lower()] = label # Add lower case for robustness
                
                entities_in_chunk = set()
                for (h, r, t) in raw_relations:
                    aggregated_relations.append((h, r, t))
                    entities_in_chunk.add(h)
                    entities_in_chunk.add(t)
                
                if not entities_in_chunk:
                    initial_ents = node.get('initial_entities') or []
                    entities_in_chunk.update(initial_ents)
                
                if entities_in_chunk:
                    chunk_entity_map[cid] = entities_in_chunk
                    all_entities.update(entities_in_chunk)
            
            if not aggregated_relations and not all_entities:
                continue

            coref_result = resolve_extraction_coreferences(aggregated_relations, list(all_entities))
            cleaned_relations = coref_result.get('cleaned_relations', [])
            entity_mappings = coref_result.get('entity_mappings', {})
            
            # Build LLM type map
            llm_type_map = {}
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

            await add_triplets_to_graph_for_episode(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                episode_id=episode_id,
                chunk_entity_map=chunk_entity_map,
                gliner_label_map=gliner_label_map,
                llm_type_map=canonical_llm_type_map
            )
            
            episodes_processed += 1
        except Exception as e:
            logger.warning(f"Episode-level enrichment failed for {episode_id}: {e}")
            errors.append(f"{episode_id}: {e}")
            continue
    
    return {"episodes_processed": episodes_processed, "errors": errors}
