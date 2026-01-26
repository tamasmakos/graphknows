import asyncio
import functools
import logging
import os
import re
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Set, Optional
import re

import networkx as nx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from kg.types import PipelineContext, ChunkExtractionTask
from kg.graph.resolution import resolve_extraction_coreferences
from kg.graph.extractors import BaseExtractor, get_extractor
from kg.graph.parsing import SegmentData
from kg.graph.parsers.life import LifeLogParser

# --- GLiNER Helper ---
from gliner import GLiNER
import spacy
import torch

logger = logging.getLogger(__name__)

GLINER_MODEL = None
SPACY_MODEL = None

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

def get_spacy_model(model_name: str = "en_core_web_lg"):
    global SPACY_MODEL
    if SPACY_MODEL is None:
        try:
            logger.info(f"Loading Spacy model ({model_name})...")
            SPACY_MODEL = spacy.load(model_name)
            logger.info("Spacy model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Spacy model {model_name}: {e}")
            logger.info(f"Trying to download {model_name}...")
            try:
                from spacy.cli import download
                download(model_name)
                SPACY_MODEL = spacy.load(model_name)
                logger.info("Spacy model loaded after download.")
            except Exception as e2:
                logger.error(f"Failed to download/load Spacy model: {e2}")
                return None
    return SPACY_MODEL

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
    max_retries: int = 3,
    keywords: List[str] = None,
    entities: List[str] = None,
    abstract_concepts: List[str] = None
) -> List[Tuple[str, str, str]]:
    """Extract relations using configured extractor."""
    if not extractor:
        return []
        
    # If no text provided, return empty
    if not text:
        return []

    retry_delay = 2

    for attempt in range(max_retries):
        try:
            relations = await extractor.extract_relations(
                text=text,
                keywords=keywords,
                entities=entities,
                abstract_concepts=abstract_concepts
            )
            return relations
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                return []
    
    return []

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
            
            raw_relations = await extract_relations_with_llm_async(
                text=task.chunk_text,
                extractor=extractor,
                keywords=task.keywords,
                entities=allowed_nodes, # Pass as entities for LangChain extractor
                abstract_concepts=[] # Already merged into entities
            )
            
            chunk_data = {
                'knowledge_triplets': raw_relations,
                'raw_extraction': {
                    'relations': raw_relations
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

async def extract_all_entities_relations(deps: PipelineContext, config: Dict[str, Any], extractor: BaseExtractor = None) -> Dict[str, Any]:
    """Phase 2: Parallel entity/relation extraction"""
    if not deps.extraction_tasks:
        return {"processed": 0, "successful": 0, "errors": []}
    
    # Deduplicate tasks by chunk_id to avoid processing the same chunk multiple times
    seen_chunk_ids = set()
    unique_tasks = []
    for task in deps.extraction_tasks:
        if task.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(task.chunk_id)
            unique_tasks.append(task)
    
    if len(unique_tasks) < len(deps.extraction_tasks):
        logger.warning(f"Removed {len(deps.extraction_tasks) - len(unique_tasks)} duplicate extraction tasks")
    
    # Create extractor instance if not provided
    should_close_extractor = False
    if extractor is None:
        extractor = get_extractor(config)
        should_close_extractor = True
        
    logger.info(f"Using {extractor.__class__.__name__} for relation extraction")
    
    # Check extraction backend
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()
        
    extraction_backend = extraction_config.get('backend', 'gliner')
    gliner_results_map = {}
    
    if extraction_backend == 'spacy':
        # --- SPACY Extraction ---
        logger.info(f"Step 2.1: Bulk Spacy Extraction for {len(unique_tasks)} chunks...")
        try:
            spacy_model_name = extraction_config.get('spacy_model', 'en_core_web_lg')
            nlp = get_spacy_model(spacy_model_name)
            
            if nlp:
                texts = [task.chunk_text for task in unique_tasks]
                # Use nlp.pipe for efficiency
                docs = list(nlp.pipe(texts))
                
                # Default mapping from Spacy labels to Ontology labels
                spacy_label_map = {
                    "PERSON": "Person",
                    "ORG": "Organization",
                    "GPE": "Location",
                    "LOC": "Location",
                    "DATE": "Date",
                    "EVENT": "Event",
                    "FAC": "Location",
                    "PRODUCT": "Concept",
                    "WORK_OF_ART": "Concept",
                    "LAW": "Concept",
                    "LANGUAGE": "Concept",
                    "MONEY": "Concept",
                    "NORP": "Group",
                    "PERCENT": "Concept",
                    "QUANTITY": "Concept",
                    "ORDINAL": "Concept",
                    "CARDINAL": "Concept"
                }
                
                for task, doc in zip(unique_tasks, docs):
                    chunk_id = task.chunk_id
                    extracted = []
                    for ent in doc.ents:
                        # Map label
                        label = spacy_label_map.get(ent.label_, "Concept")
                        extracted.append({"text": ent.text, "label": label})
                    
                    gliner_results_map[chunk_id] = extracted
                
                logger.info(f"Bulk Spacy complete. Extracted entities for {len(gliner_results_map)} chunks.")
        
        except Exception as e:
             logger.error(f"Bulk Spacy extraction failed: {e}", exc_info=True)
             
    else:
        # --- GLiNER Extraction (Default) ---
        logger.info(f"Step 2.1: Bulk GLiNER Pass for {len(unique_tasks)} chunks...")
        try:
            model = get_gliner_model()
            if model:
                labels = extraction_config.get('gliner_labels', ["Person", "Organization", "Location", "Event", "Date", "Award", "Competitions", "Teams", "Concept"])
                
                all_sentences = []
                sentence_to_task_map = []
                
                for task in unique_tasks:
                    sents = split_sentences(task.chunk_text)
                    if not sents:
                        sents = [task.chunk_text]
                    all_sentences.extend(sents)
                    sentence_to_task_map.extend([task.chunk_id] * len(sents))
                
                if all_sentences:
                    # Run in thread to avoid blocking event loop
                    predict_func = functools.partial(
                        model.batch_predict_entities, 
                        all_sentences, 
                        labels, 
                        threshold=0.5
                    )
                    all_predictions = await asyncio.to_thread(predict_func)
                    
                    # Re-map results to chunks
                    for chunk_id, preds in zip(sentence_to_task_map, all_predictions):
                        if chunk_id not in gliner_results_map:
                            gliner_results_map[chunk_id] = []
                        gliner_results_map[chunk_id].extend(preds)
                    
                    logger.info(f"Bulk GLiNER complete. Extracted entities for {len(gliner_results_map)} chunks.")
        except Exception as e:
            logger.error(f"Bulk GLiNER extraction failed: {e}")

    try:
        # Use max_concurrent from config for controlled parallelization
        max_concurrent = get_max_concurrent(config, default=8)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [process_extraction_task(deps, task, semaphore, extractor, gliner_results_map.get(task.chunk_id, [])) for task in unique_tasks]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r.get("success"))
        errors = [r.get("error") for r in results if not r.get("success") and r.get("error")]
        
        logger.info(f"Extraction complete: {successful}/{len(results)} successful")
        
        # Enrich graph per segment
        enrich_result = await enrich_graph_per_segment(deps)
        
        return {
            "processed": len(results),
            "successful": successful,
            "errors": errors + enrich_result.get("errors", [])
        }
    finally:
        if should_close_extractor and hasattr(extractor, 'close'):
            await extractor.close()

# --- Graph Enrichment ---

def _get_chunks_for_segment(graph: nx.DiGraph, segment_id: str) -> List[str]:
    if not graph.has_node(segment_id):
        return []
def _get_chunks_for_segment(graph: nx.DiGraph, segment_id: str) -> List[str]:
    """Find all chunks associated with a segment, including via Conversations and Contexts."""
    chunk_ids = set()
    
    # 1. Direct Chunks (old schema)
    for neighbor in graph.neighbors(segment_id):
        edge = graph.get_edge_data(segment_id, neighbor) or {}
        if edge.get('label') == 'HAS_CHUNK' and graph.nodes[neighbor].get('node_type') == 'CHUNK':
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
    relations: List[Tuple[str, str, str]],
    entity_mappings: Dict[str, str],
    segment_id: str,
    chunk_entity_map: Dict[str, Set[str]],
    gliner_label_map: Dict[str, str] = None  # New parameter
):
    """Write entities/edges to graph for a segment"""
    graph = deps.graph
    gliner_label_map = gliner_label_map or {}
    
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
                         name=mapped_ent, 
                         graph_type="entity_relation")
        else:
            # Update existing node
            node_data = graph.nodes[mapped_ent]
            
            # Ensure name is set
            if 'name' not in node_data:
                node_data['name'] = mapped_ent
                
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
            
        graph.add_edge(h, t, label=r, relation_type=r, graph_type="entity_relation", segment_id=segment_id, source="extraction")
        # Log edge creation (verbose)
        # logger.debug(f"Created edge: {h} --[{r}]--> {t}")

    # Link chunks to entities
    for chunk_id, entities in chunk_entity_map.items():
        for ent in entities:
            mapped = entity_mappings.get(ent, ent)
            if graph.has_node(mapped):
                graph.add_edge(chunk_id, mapped, label="HAS_ENTITY", graph_type="lexical_graph")

async def enrich_graph_per_segment(deps: PipelineContext) -> Dict[str, Any]:
    """Aggregate chunk-level extractions per segment, run coref, and enrich graph."""
    graph = deps.graph
    segment_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SEGMENT']
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
                raw_relations = (node.get('raw_extraction') or {}).get('relations') or []
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
            
            await add_triplets_to_graph_for_segment(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                segment_id=segment_id,
                chunk_entity_map=chunk_entity_map,
                gliner_label_map=gliner_label_map # Pass the map
            )
            
            segments_processed += 1
        except Exception as e:
            logger.warning(f"Segment-level enrichment failed for {segment_id}: {e}")
            errors.append(f"{segment_id}: {e}")
            continue
    
    return {"segments_processed": segments_processed, "errors": errors}

# --- Lexical Graph Construction with LangChain ---
 
async def process_document_splitting(
    content: str, 
    config: Dict[str, Any]
) -> List[str]: # Returns list of chunk strings
    """Process text using LangChain splitter."""
    
    try:
        # Configure splitter
        extraction_config = config.get('extraction', {})
        if hasattr(extraction_config, 'model_dump'):
            extraction_config = extraction_config.model_dump()
            
        chunk_size = extraction_config.get('chunk_size', 512)
        chunk_overlap = extraction_config.get('chunk_overlap', 50)
        
        # Ensure overlap is not larger than chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(20, chunk_size // 10)
            logger.warning(f"chunk_overlap too large, adjusted to {chunk_overlap}")
        
        logger.debug(f"RecursiveCharacterTextSplitter configured: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = splitter.split_text(content)
        return chunks
        
    except Exception as e:
        logger.error(f"Error in splitting: {e}")
        raise

async def process_single_segment(
    deps: PipelineContext,
    segment: SegmentData,
    doc_id: str,
    segment_index: int,
    global_segment_order: int,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    async with semaphore:
        chunk_count = 0
        
        # Set name field: first 20 chars of content
        segment_name = segment.content[:20].strip() if segment.content else f"Segment {segment_index}"
        if len(segment.content) > 20:
            segment_name += "..."
        
        # Add Segment node
        deps.graph.add_node(segment.segment_id,
                          node_type="SEGMENT",
                          graph_type="lexical_graph",
                          content=segment.content,
                          content_length=len(segment.content),
                          line_number=segment.line_number,
                          document_date=segment.date.isoformat(),
                          date=segment.date,
                          local_segment_order=segment_index,
                          global_segment_order=global_segment_order,
                          sentiment=segment.sentiment,
                          name=segment_name,
                          **segment.metadata)
        
        # Edge: DAY -> HAS_SEGMENT -> SEGMENT
        deps.graph.add_edge(doc_id, segment.segment_id, label="HAS_SEGMENT", graph_type="lexical_graph")
        
        # Check if this is a Life Graph Episode with structured conversations
        if segment.metadata.get('conversations'):
            try:
                conversations = segment.metadata['conversations']
                
                # Image Descriptions Accumulator
                image_descriptions = []
                
                for i, row in enumerate(conversations):
                    # No longer creating CONVERSATION nodes
                    # Direct Chunking from Audio
                    chunk_text = row.get('Audio')
                    if chunk_text:
                        # Split text if too long to prevent GLINER truncation
                        sub_chunks = await process_document_splitting(chunk_text, config)
                        
                        for sub_idx, sub_text in enumerate(sub_chunks):
                            # Chunk ID derived from segment
                            # Use simplified ID if only one chunk, otherwise append sub_index
                            if len(sub_chunks) == 1:
                                chunk_id = f"{segment.segment_id}_CHUNK_{i}"
                            else:
                                chunk_id = f"{segment.segment_id}_CHUNK_{i}_{sub_idx}"
                            
                            chunk_name = sub_text[:20].strip() + "..." if len(sub_text) > 20 else sub_text

                            deps.graph.add_node(chunk_id,
                                              node_type="CHUNK",
                                              graph_type="lexical_graph",
                                              text=sub_text,
                                              length=len(sub_text),
                                              initial_entities=[],
                                              llama_metadata={}, 
                                              name=chunk_name)
                            
                            # Link SEGMENT -> HAS_CHUNK -> CHUNK (Directly)
                            deps.graph.add_edge(segment.segment_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
                            
                            # Add retrieval task for this chunk
                            deps.extraction_tasks.append(ChunkExtractionTask(
                                chunk_id=chunk_id,
                                chunk_text=sub_text,
                                entities=[], 
                                abstract_concepts=[], 
                                keywords=[]
                            ))
                            chunk_count += 1
                    
                    # Extract Entities (Spacy/Rule-based mappings)
                    # 1. Location -> PLACE
                    loc_name = row.get('Location')
                    if loc_name:
                        # Simple linking: Location string is the Place ID/Name
                        place_id = f"PLACE_{loc_name.replace(' ', '_').upper()}"
                        if not deps.graph.has_node(place_id):
                            deps.graph.add_node(place_id, node_type="PLACE", name=loc_name, graph_type="entity_relation")
                        
                        # Link Segment to Place directly
                        deps.graph.add_edge(segment.segment_id, place_id, label="HAPPENED_AT", graph_type="entity_relation")

                    # 2. Image -> Attribute on Segment
                    image_desc = row.get('Image')
                    if image_desc:
                        image_descriptions.append(image_desc)

                # Store collected image descriptions on the Segment node
                if image_descriptions:
                    deps.graph.nodes[segment.segment_id]['image_descriptions'] = image_descriptions
                    # Keep single field for backward compat if needed, or just use list
                    deps.graph.nodes[segment.segment_id]['image_description'] = "; ".join(image_descriptions)

            except Exception as e:
                logger.error(f"Error processing Life Graph segment {segment.segment_id}: {e}", exc_info=True)
                
        else:
            # Fallback to standard Text Splitter for non-LifeGraph segments
            try:
                chunks = await process_document_splitting(segment.content, config)
                
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{segment.segment_id}_CHUNK_{i}"
                    
                    metadata = {}
                    
                    chunk_name = chunk_text[:20].strip() if chunk_text else f"Chunk {i}"
                    if len(chunk_text) > 20:
                        chunk_name += "..."
                    
                    deps.graph.add_node(chunk_id,
                                      node_type="CHUNK",
                                      graph_type="lexical_graph",
                                      text=chunk_text,
                                      length=len(chunk_text),
                                      initial_entities=[],
                                      initial_concepts=[],
                                      llama_metadata=metadata, 
                                      name=chunk_name)
                    
                    deps.graph.add_edge(segment.segment_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
                    
                    deps.extraction_tasks.append(ChunkExtractionTask(
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                        entities=[],  # No longer extracting keywords/entities
                        abstract_concepts=[], 
                        keywords=[]
                    ))
                    chunk_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing segment {segment.segment_id} with splitting: {e}", exc_info=True)
        
        return {"chunk_count": chunk_count, "segment_id": segment.segment_id}

async def add_segments_to_graph(deps: PipelineContext, segments: List[SegmentData], doc_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    config = config or {}
    chunk_count = 0
    segment_count = 0
    
    # Get max_concurrent from config for controlled parallelization
    max_concurrent = get_max_concurrent(config, default=8)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process segments concurrently
    tasks = []
    
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    limit = extraction_config.get('speech_limit', 0)
    # Also support old key if needed, or just rely on 'speech_limit'
    if 'segment_limit' in config:
         limit = config['segment_limit']
    
    for idx, segment in enumerate(segments):
        # Check limit before creating task
        if limit > 0 and deps.total_segments >= limit:
            break
        
        # Capture current global_segment_order before incrementing
        current_global_order = deps.total_segments
            
        task = process_single_segment(
            deps=deps,
            segment=segment,
            doc_id=doc_id,
            segment_index=idx,
            global_segment_order=current_global_order,
            config=config,
            semaphore=semaphore
        )
        tasks.append(task)
        segment_count += 1
        deps.total_segments += 1
        
        # Check limit after incrementing
        if limit > 0 and deps.total_segments >= limit:
            break
    
    # Wait for all segment processing tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    errors = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
        else:
            chunk_count += result.get("chunk_count", 0)
    
    return {"segments_count": segment_count, "chunks_count": chunk_count, "errors": errors}

async def process_single_document_lexical(deps: PipelineContext, filename: str, input_dir: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a single document."""
    config = config or {}
    
    parser = LifeLogParser()
    
    try:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading {filename}: {str(e)}")
        return {"error": f"Error reading {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

    # Try to extract date from filename first
    doc_date_str = parser.extract_date(filename)
    
    # Fallback to content-based extraction
    if not doc_date_str:
        doc_date_str = parser.extract_date_from_content(text)
        
    if not doc_date_str:
        logger.warning(f"Could not extract date from {filename}, using today's date")
        doc_date_str = datetime.now().strftime("%Y-%m-%d")
        
    doc_date_obj = datetime.strptime(doc_date_str, "%Y-%m-%d").date()
    day_id = f"DAY_{doc_date_str}"
    
    try:
        logger.info(f"Processing document {filename} ({len(text)} chars)")
        
        # Add DAY node if not exists
        if not deps.graph.has_node(day_id):
            deps.graph.add_node(day_id, 
                               node_type="DAY", 
                               graph_type="lexical_graph",
                               date=doc_date_str,
                               name=doc_date_str,
                               segment_count=0)
        
        # Parse segments
        segments = parser.parse(text, filename, doc_date_obj)
        
        segments_result = await add_segments_to_graph(deps, segments, day_id, config)
        
        # Update segment count on DAY node
        deps.graph.nodes[day_id]['segment_count'] = (deps.graph.nodes[day_id].get('segment_count', 0) + 
                                                   segments_result.get("segments_count", 0))
        
        return {
            "day_id": day_id,
            "segments_added": segments_result.get("segments_count", 0),
            "chunks_added": segments_result.get("chunks_count", 0),
            "errors": segments_result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return {"error": f"Error processing {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

async def build_lexical_graph(deps: PipelineContext, input_dir: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Phase 1: Build the complete lexical graph structure sequentially"""
    config = config or {}
    results = {"documents_processed": 0, "total_segments": 0, "total_chunks": 0, "errors": []}
    
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} not found")
        
        # Check if a specific file pattern is provided (for incremental processing)
        extraction_config = config.get('extraction', {})
        if hasattr(extraction_config, 'model_dump'):
            extraction_config = extraction_config.model_dump()
            
        file_pattern = extraction_config.get('file_pattern')
        if file_pattern:
            # Process only the specified file
            import fnmatch
            all_files = os.listdir(input_dir)
            filenames = [f for f in all_files if fnmatch.fnmatch(f, file_pattern)]
            logger.info(f"Processing specific file(s) matching pattern '{file_pattern}': {filenames}")
        else:
            # Process all .txt files
            filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
            logger.info(f"Found {len(filenames)} files to process")
        
        for filename in filenames:
            limit = config.get('segment_limit', config.get('speech_limit', 0))
            if limit > 0 and deps.total_segments >= limit:
                break
                
            doc_result = await process_single_document_lexical(deps, filename, input_dir, config)
                
            results["documents_processed"] += 1
            results["total_segments"] += doc_result.get("segments_added", 0)
            results["total_chunks"] += doc_result.get("chunks_added", 0)
            
            if doc_result.get("errors"):
                results["errors"].extend(doc_result["errors"])
                
        return results
        
    except Exception as e:
        error_msg = f"Error in build_lexical_graph: {str(e)}"
        logger.error(error_msg, exc_info=True)
        results["errors"].append(error_msg)
        return results

