import os
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, Any, List
import networkx as nx

from graphgen.data_types import PipelineContext, ChunkExtractionTask, SegmentData
from graphgen.pipeline.lexical_graph_building.chunking import process_document_splitting
from graphgen.utils.parsers.life import LifeLogParser
from graphgen.config.schema import GraphSchema

logger = logging.getLogger(__name__)

def get_max_concurrent(config: Dict[str, Any], default: int = 8) -> int:
    """Get max_concurrent_extractions from config with fallback to default."""
    extraction_cfg = config.get('extraction', {})
    if hasattr(extraction_cfg, 'model_dump'):
        extraction_cfg = extraction_cfg.model_dump()
        
    return extraction_cfg.get('max_concurrent_chunks', default)

async def process_single_segment(
    deps: PipelineContext,
    segment: SegmentData,
    doc_id: str,
    segment_index: int,
    global_segment_order: int,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    schema: GraphSchema = None
) -> Dict[str, Any]:
    async with semaphore:
        chunk_count = 0
        
        # Node Labels
        segment_label = schema.nodes.get("Segment", {}).label if schema else "SEGMENT"
        chunk_label = schema.nodes.get("Chunk", {}).label if schema else "CHUNK"
        
        # 1. Add Segment Node
        # ID: DOC_ID + _S + Index
        segment_node_id = f"{doc_id}_S{segment_index}"
        
        segment_name = segment.content[:20].strip() if segment.content else f"Segment {segment_index}"
        if len(segment.content) > 20:
            segment_name += "..."

        deps.graph.add_node(segment_node_id,
                          node_type=segment_label,
                          graph_type="lexical_graph",
                          content=segment.content,
                          line_number=segment.line_number,
                          name=segment_name,
                          **segment.metadata)
        
        # 2. Connect Doc -> Segment
        deps.graph.add_edge(doc_id, segment_node_id, label="HAS_SEGMENT", graph_type="lexical_graph")
        
        # 3. Chunking (Lexical Splitting)
        try:
            chunks = await process_document_splitting(segment.content, config)
            
            for i, chunk_text in enumerate(chunks):
                # ID generation
                chunk_id = f"{segment_node_id}_C{i}"
                
                chunk_name = chunk_text[:20].strip() if chunk_text else f"Chunk {i}"
                if len(chunk_text) > 20:
                    chunk_name += "..."
                
                # 4. Add Chunk Node
                deps.graph.add_node(chunk_id,
                                  node_type=chunk_label,
                                  graph_type="lexical_graph",
                                  text=chunk_text,
                                  length=len(chunk_text),
                                  initial_entities=[],
                                  llama_metadata={}, 
                                  name=chunk_name)
                
                # 5. Connect Segment -> Chunk
                deps.graph.add_edge(segment_node_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
                
                deps.extraction_tasks.append(ChunkExtractionTask(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    entities=[],
                    abstract_concepts=[], 
                    keywords=[]
                ))
                chunk_count += 1
                
        except Exception as e:
            logger.error(f"Error processing segment {segment.segment_id} with splitting: {e}", exc_info=True)
        
        return {"chunk_count": chunk_count, "segment_id": segment.segment_id}

async def add_segments_to_graph(deps: PipelineContext, segments: List[SegmentData], doc_id: str, config: Dict[str, Any] = None, schema: GraphSchema = None) -> Dict[str, Any]:
    config = config or {}
    chunk_count = 0
    segment_count = 0
    
    # Get max_concurrent from config for controlled parallelization
    max_concurrent = get_max_concurrent(config, default=8)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process segments concurrently
    tasks = []
    
    for idx, segment in enumerate(segments):
        # Capture current global_segment_order before incrementing
        current_global_order = deps.total_segments
            
        task = process_single_segment(
            deps=deps,
            segment=segment,
            doc_id=doc_id,
            segment_index=idx,
            global_segment_order=current_global_order,
            config=config,
            semaphore=semaphore,
            schema=schema
        )
        tasks.append(task)
        segment_count += 1
        deps.total_segments += 1
    
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

async def process_single_document_lexical(deps: PipelineContext, filename: str, input_dir: str, config: Dict[str, Any] = None, schema: GraphSchema = None, parser: Any = None) -> Dict[str, Any]:
    """Process a single document."""
    config = config or {}
    
    # Use provided parser or default
    if not parser:
        if filename.lower().endswith('.csv'):
             parser = LifeLogParser()
        else:
             from graphgen.utils.parsers.custom import RegexParser
             # Use a regex that never matches to treat whole file as one segment?
             # NO: User requirement: "each new line is a segment"
             # So we should use a LineParser or configure RegexParser to split on newlines.
             # Or simply read lines here as we do in iterative_loader?
             # Since builder.py might be used differently, let's implement the line reading logic directly here 
             # OR ensure the parser does line splitting.
             # The existing RegexParser splits by `segment_splitter`.
             # Let's override to split by newline "\n".
             parser = RegexParser(segment_splitter="\n", attributes_map={})
    
    # Node Labels
    doc_label = schema.nodes.get("Doc", {}).label if schema else "DOC"
        
    try:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading {filename}: {str(e)}")
        return {"error": f"Error reading {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

    # No date extraction by default as requested.
    doc_date_str = None
        
    # Create Document Node ID
    doc_id = f"DOC_{filename}"
    
    try:
        logger.info(f"Processing document {filename} ({len(text)} chars)")
        
        # Add Document Node
        if not deps.graph.has_node(doc_id):
            deps.graph.add_node(doc_id, 
                               node_type=doc_label, 
                               graph_type="lexical_graph",
                               date=doc_date_str,
                               name=filename, 
                               segment_count=0)
        
        # Parse segments (Lines)
        # Re-using parser concept, but ensuring it yields lines.
        # If RegexParser(segment_splitter="\n") works, we use it. 
        # But `parser.parse` might expect something else.
        # Let's simplify and just split text by lines manually if it's the default case.
        
        if isinstance(parser, LifeLogParser):
            segments = parser.parse(text, filename, None)
        else:
            # Default line-based segmentation
            lines = text.splitlines()
            segments = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                # We reuse SegmentData
                segments.append(SegmentData(
                    segment_id=f"{doc_id}_raw_{i}", # temp ID, redefined in add_segments
                    content=line,
                    line_number=i,
                    date=None,
                    metadata={}
                ))
        
        segments_result = await add_segments_to_graph(deps, segments, doc_id, config, schema=schema)
        
        # Update segment count
        if deps.graph.has_node(doc_id):
             deps.graph.nodes[doc_id]['segment_count'] = (deps.graph.nodes[doc_id].get('segment_count', 0) + 
                                                    segments_result.get("segments_count", 0))
        
        return {
            "day_id": doc_id,
            "segments_added": segments_result.get("segments_count", 0),
            "chunks_added": segments_result.get("chunks_count", 0),
            "errors": segments_result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return {"error": f"Error processing {filename}: {str(e)}", "segments_added": 0, "chunks_added": 0}

async def build_lexical_graph(deps: PipelineContext, input_dir: str, config: Dict[str, Any] = None, schema: GraphSchema = None) -> Dict[str, Any]:
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
        all_files = sorted(os.listdir(input_dir))
        if file_pattern:
            import fnmatch
            filenames = sorted(f for f in all_files if fnmatch.fnmatch(f, file_pattern))
            logger.info(f"Processing specific file(s) matching pattern '{file_pattern}': {filenames}")
        else:
            filenames = sorted(f for f in all_files if f.endswith('.txt'))
            logger.info(f"Found {len(filenames)} files to process")

        test_mode_cfg = config.get('test_mode', {})
        if hasattr(test_mode_cfg, 'model_dump'):
            test_mode_cfg = test_mode_cfg.model_dump()

        test_mode_enabled = test_mode_cfg.get('enabled', False)
        max_documents = test_mode_cfg.get('max_documents', 0)
        if test_mode_enabled and max_documents > 0 and len(filenames) > max_documents:
            logger.info(f"Test mode enabled: limiting to {max_documents} documents (from {len(filenames)} available)")
            filenames = filenames[:max_documents]

        max_chunks = test_mode_cfg.get('max_chunks', 0) or extraction_config.get('max_chunks', 0)
        chunk_budget_remaining = max_chunks if max_chunks and max_chunks > 0 else None
        chunk_node_labels = {'CHUNK'}
        if schema:
            for node_schema in schema.nodes.values():
                if getattr(node_schema, 'source_type', None) == 'chunk' and getattr(node_schema, 'label', None):
                    chunk_node_labels.add(str(node_schema.label))

        for filename in filenames:
            if chunk_budget_remaining is not None and chunk_budget_remaining <= 0:
                logger.info("Chunk budget exhausted before processing remaining documents.")
                break

            doc_result = await process_single_document_lexical(deps, filename, input_dir, config, schema=schema)

            if chunk_budget_remaining is not None:
                chunk_count = doc_result.get("chunks_added", 0)
                if chunk_count > chunk_budget_remaining:
                    overflow = chunk_count - chunk_budget_remaining
                    logger.info(
                        "Trimming %d excess chunk(s) from %s to respect max_chunks=%d.",
                        overflow,
                        filename,
                        max_chunks,
                    )
                    chunk_ids_to_remove = []
                    extraction_task_ids = set()
                    for node_id in reversed(list(deps.graph.nodes())):
                        if overflow <= 0:
                            break
                        node_data = deps.graph.nodes[node_id]
                        if str(node_data.get('node_type')) not in chunk_node_labels:
                            continue
                        if not node_id.startswith(f"DOC_{filename}_"):
                            continue
                        chunk_ids_to_remove.append(node_id)
                        extraction_task_ids.add(node_id)
                        overflow -= 1
                    for chunk_id in chunk_ids_to_remove:
                        if deps.graph.has_node(chunk_id):
                            deps.graph.remove_node(chunk_id)
                    deps.extraction_tasks = [
                        task for task in deps.extraction_tasks if task.chunk_id not in extraction_task_ids
                    ]
                    kept_chunks = chunk_count - len(chunk_ids_to_remove)
                    doc_result["chunks_added"] = kept_chunks

                chunk_budget_remaining -= doc_result.get("chunks_added", 0)
                logger.info(
                    "Chunk budget status after %s: remaining=%s",
                    filename,
                    chunk_budget_remaining,
                )

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
