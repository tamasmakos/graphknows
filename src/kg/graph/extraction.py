import asyncio
import logging
import os
import re
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Set, Optional

import networkx as nx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.kg.types import AgentDependencies, ChunkExtractionTask
from src.kg.graph.coref import resolve_coreferences_simple
from src.kg.graph.extractors import BaseExtractor, get_extractor
from src.kg.graph.parsing import SegmentData
from src.kg.graph.parsers import get_parser

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    return config.get('max_concurrent_extractions', default)

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
    deps: AgentDependencies,
    task: ChunkExtractionTask,
    semaphore: asyncio.Semaphore,
    extractor: BaseExtractor
) -> Dict[str, Any]:
    """Process a single chunk extraction task"""
    async with semaphore:
        # Use full chunk_id to avoid confusion with similar short IDs
        logger.info(f"      🚀 Starting {task.chunk_id}")
        
        try:
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
                }
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

async def extract_all_entities_relations(deps: AgentDependencies, config: Dict[str, Any], extractor: BaseExtractor = None) -> Dict[str, Any]:
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
    
    try:
        # Use max_concurrent from config for controlled parallelization
        max_concurrent = get_max_concurrent(config, default=8)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = [process_extraction_task(deps, task, semaphore, extractor) for task in unique_tasks]
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
    chunk_ids = []
    for neighbor in graph.neighbors(segment_id):
        edge_data = graph.get_edge_data(segment_id, neighbor) or {}
        if edge_data.get('label') == 'HAS_CHUNK' and graph.nodes[neighbor].get('node_type') == 'CHUNK':
            chunk_ids.append(neighbor)
    return chunk_ids

async def add_triplets_to_graph_for_segment(
    deps: AgentDependencies,
    relations: List[Tuple[str, str, str]],
    entity_mappings: Dict[str, str],
    segment_id: str,
    chunk_entity_map: Dict[str, Set[str]]
):
    """Write entities/edges to graph for a segment"""
    graph = deps.graph
    
    # Add entities
    for _, mapped_ent in entity_mappings.items():
        if not graph.has_node(mapped_ent):
            graph.add_node(mapped_ent, node_type="ENTITY_CONCEPT", name=mapped_ent, graph_type="entity_relation")
    
    # Add relations
    for h, r, t in relations:
        if not graph.has_node(h) or not graph.has_node(t):
            continue
            
        graph.add_edge(h, t, label=r, relation_type=r, graph_type="entity_relation", segment_id=segment_id)

    # Link chunks to entities
    for chunk_id, entities in chunk_entity_map.items():
        for ent in entities:
            mapped = entity_mappings.get(ent, ent)
            if graph.has_node(mapped):
                graph.add_edge(chunk_id, mapped, label="HAS_ENTITY", graph_type="lexical_graph")

async def enrich_graph_per_segment(deps: AgentDependencies) -> Dict[str, Any]:
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
            
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_relations = (node.get('raw_extraction') or {}).get('relations') or []
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

            coref_result = resolve_coreferences_simple(aggregated_relations, list(all_entities))
            cleaned_relations = coref_result.get('cleaned_relations', [])
            entity_mappings = coref_result.get('entity_mappings', {})
            
            await add_triplets_to_graph_for_segment(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                segment_id=segment_id,
                chunk_entity_map=chunk_entity_map 
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
        chunk_size = config.get('chunk_size', 512)
        chunk_overlap = config.get('chunk_overlap', 50)
        
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
    deps: AgentDependencies,
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
        
        # Process segment with splitting
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

async def add_segments_to_graph(deps: AgentDependencies, segments: List[SegmentData], doc_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    config = config or {}
    chunk_count = 0
    segment_count = 0
    
    # Get max_concurrent from config for controlled parallelization
    max_concurrent = get_max_concurrent(config, default=8)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process segments concurrently
    tasks = []
    limit = config.get('segment_limit', config.get('speech_limit', 0))
    
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

async def process_single_document_lexical(deps: AgentDependencies, filename: str, input_dir: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a single document."""
    config = config or {}
    
    # Try to extract date from filename
    # Use GenericParser's logic implicitly via get_parser check or manual check
    # We'll use a temporary parser instance to check support and extract date
    
    temp_parser = get_parser('auto', filename=filename)
    doc_date_str = temp_parser.extract_date(filename)
    
    if not doc_date_str:
        logger.warning(f"Skipping {filename} - cannot extract date")
        return {"error": f"Skipping {filename} - cannot extract date", "segments_added": 0, "chunks_added": 0}
        
    doc_date_obj = datetime.strptime(doc_date_str, "%Y-%m-%d").date()
    # Create DAY node ID
    day_id = f"DAY_{doc_date_str}"
    # Use generic DOC ID if preferred, but DAY hierarchy is established.
    # Let's keep DAY node as root for these segments.
    
    try:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
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
        parser = get_parser('auto', filename=filename)
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

async def build_lexical_graph(deps: AgentDependencies, input_dir: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Phase 1: Build the complete lexical graph structure sequentially"""
    config = config or {}
    results = {"documents_processed": 0, "total_segments": 0, "total_chunks": 0, "errors": []}
    
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} not found")
        
        # Check if a specific file pattern is provided (for incremental processing)
        file_pattern = config.get('file_pattern')
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

