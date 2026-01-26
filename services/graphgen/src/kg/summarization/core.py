import os
import logging
import asyncio
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate


from .models import SummarizationTask

logger = logging.getLogger(__name__)

# Configuration for topic summarization concurrency
MAX_CONCURRENT_SUMMARIES = 10  # Adjust based on API rate limits

def truncate_text_for_llm(text: str, max_chars: int = 15000) -> str:
    """Truncate text to fit LLM context window with graceful degradation"""
    if len(text) <= max_chars:
        return text
    
    # Try to break at sentence boundaries
    sentences = text.split('. ')
    truncated = ""
    for sentence in sentences:
        if len(truncated) + len(sentence) + 1 <= max_chars:
            truncated += sentence + ". "
        else:
            break
    
    # If we couldn't fit any complete sentences, just truncate
    if not truncated:
        truncated = text[:max_chars]
    
    return truncated.strip()


async def generate_title_internal(llm: Any, text: str) -> str:
    """Generate title for given text using LLM"""
    
    truncated_text = truncate_text_for_llm(text, max_chars=12000)
    
    title_prompt = ChatPromptTemplate.from_template(
        """Please generate a concise, descriptive title (maximum 10 words) for the following content. 
        The title should capture the main topic or theme regarding daily life, habits, or patterns without being too generic.
        
        Content:
        {text}
        
        Title:"""
    )
    
    for attempt in range(3):
        try:
            chain = title_prompt | llm
            response = await chain.ainvoke({"text": truncated_text})
            if hasattr(response, 'content'):
                title = response.content
            else:
                title = str(response)
            if len(title) > 100:
                title = title[:97] + "..."
            return title if title else "Untitled Topic"
        except RuntimeError as e:
            if "Event loop is closed" in str(e) and attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            logger.error(f"Title generation failed: {e}")
            return "Untitled Topic"
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            logger.error(f"Title generation failed: {e}")
            return "Untitled Topic"
    return "Untitled Topic"

async def summarize_text_internal(llm: Any, text: str) -> str:
    """Generate summary for given text using LLM"""
    
    truncated_text = truncate_text_for_llm(text, max_chars=12000)
    
    summary_prompt = ChatPromptTemplate.from_template(
        """Please provide a comprehensive summary (3-5 sentences) of the following content. 
        Focus on average person topics - life related things, things to remember, things to notice, habits, patterns and so on.
        
        Content:
        {text}
        
        Summary:"""
    )
    
    for attempt in range(3):
        try:
            chain = summary_prompt | llm
            response = await chain.ainvoke({"text": truncated_text})
            if hasattr(response, 'content'):
                summary = response.content
            else:
                summary = str(response)
            return summary if summary else "No summary available."
        except RuntimeError as e:
            if "Event loop is closed" in str(e) and attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            logger.error(f"Summary generation failed: {e}")
            return "No summary available."
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            logger.error(f"Summary generation failed: {e}")
            return "No summary available."
    return "No summary available."

async def find_entities_for_community_async(graph: nx.DiGraph, topic_node_id: str) -> List[str]:
    """Find entity IDs that belong to a topic"""
    entity_ids = []
    
    # Look for entity nodes connected to topic node
    if graph.has_node(topic_node_id):
        # Structure is Entity -> Subtopic -> Topic (incoming edges)
        for pred in graph.predecessors(topic_node_id):
            node_data = graph.nodes[pred]
            node_type = str(node_data.get('node_type', '')).upper()
            
            # If predecessor is a Subtopic, look for its entities
            if node_type == 'SUBTOPIC':
                for sub_pred in graph.predecessors(pred):
                    sub_node_type = str(graph.nodes[sub_pred].get('node_type', '')).upper()
                    if sub_node_type in ['ENTITY', 'ENTITY_CONCEPT', 'PLACE']:
                        entity_ids.append(sub_pred)
                        
            # If predecessor is directly an Entity
            elif node_type in ['ENTITY', 'ENTITY_CONCEPT', 'PLACE']:
                entity_ids.append(pred)
    
    return entity_ids

async def find_chunks_for_entities_async(graph: nx.DiGraph, entity_ids: List[str]) -> List[str]:
    """Find chunk IDs connected to the given entity IDs"""
    chunk_ids = set()
    
    for entity_id in entity_ids:
        if graph.has_node(entity_id):
            # Structure is Chunk -> Entity (HAS_ENTITY)
            for pred in graph.predecessors(entity_id):
                node_data = graph.nodes[pred]
                node_type = str(node_data.get('node_type', '')).upper()
                if node_type == 'CHUNK':
                    chunk_ids.add(pred)
    
    return list(chunk_ids)

async def sort_chunks_by_global_order_async(graph: nx.DiGraph, chunk_ids: List[str]) -> List[str]:
    """Sort chunks by their global order (speech_order, chunk_order)"""
    
    chunk_data = []
    for chunk_id in chunk_ids:
        if graph.has_node(chunk_id):
            node_data = graph.nodes[chunk_id]
            speech_order = node_data.get('speech_order', 0)
            chunk_order = node_data.get('chunk_order', 0)
            chunk_data.append((chunk_id, speech_order, chunk_order))
    
    # Sort by speech_order first, then chunk_order
    chunk_data.sort(key=lambda x: (x[1], x[2]))
    
    return [chunk_id for chunk_id, _, _ in chunk_data]

async def concatenate_chunk_texts_async(graph: nx.DiGraph, chunk_ids: List[str]) -> str:
    """Concatenate text from multiple chunks in order"""
    
    texts = []
    for chunk_id in chunk_ids:
        if graph.has_node(chunk_id):
            node_data = graph.nodes[chunk_id]
            chunk_text = node_data.get('text', '')
            if chunk_text:
                texts.append(chunk_text)
    
    return ' '.join(texts)

async def collect_community_tasks_async(graph: nx.DiGraph) -> List[SummarizationTask]:
    """Collect summarization tasks for all topic nodes"""
    
    tasks = []
    topic_nodes = []
    
    # Find all topic nodes (TOPIC_X)
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'TOPIC' and 
            isinstance(node_id, str) and 
            node_id.startswith('TOPIC_')):
            topic_nodes.append((node_id, node_data))
    
    logger.info(f"Found {len(topic_nodes)} topic nodes for summarization")
    
    for topic_node_id, topic_data in topic_nodes:
        try:
            # Extract topic ID from node name (e.g., "TOPIC_0" -> 0)
            community_id = int(topic_node_id.split('_')[1])
            
            # Find entities for this topic
            entity_ids = await find_entities_for_community_async(graph, topic_node_id)
            
            # Find chunks - try via entities first, then via topic node directly
            chunk_ids = []
            if entity_ids:
                chunk_ids = await find_chunks_for_entities_async(graph, entity_ids)
            
            # If no chunks via entities, try to find chunks connected to this topic
            if not chunk_ids:
                # Look for chunks connected through subtopics
                # Structure: CHUNK -> ENTITY -> SUBTOPIC -> TOPIC
                # So we need to go backwards (predecessors)
                for neighbor in graph.predecessors(topic_node_id):
                    if graph.nodes[neighbor].get('node_type') == 'SUBTOPIC':
                        for sub_neighbor in graph.predecessors(neighbor):
                            if graph.nodes[sub_neighbor].get('node_type') == 'ENTITY_CONCEPT':
                                # Entity -> Subtopic, now look for Chunk -> Entity
                                for entity_pred in graph.predecessors(sub_neighbor):
                                    if graph.nodes[entity_pred].get('node_type') == 'CHUNK':
                                        chunk_ids.append(entity_pred)
            
            # If still no chunks, use fallback: any chunks in graph
            if not chunk_ids:
                all_chunks = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'CHUNK']
                if all_chunks:
                    chunk_ids = all_chunks[:10] # Increased slightly
                    logger.info(f"Using fallback chunks for topic {community_id}: {len(chunk_ids)} chunks")
            
            if not chunk_ids:
                logger.warning(f"No chunks found for topic {community_id}, skipping")
                continue
            
            # Sort chunks by global order
            sorted_chunk_ids = await sort_chunks_by_global_order_async(graph, chunk_ids)
            
            # Concatenate chunk texts
            concatenated_text = await concatenate_chunk_texts_async(graph, sorted_chunk_ids)
            
            if not concatenated_text.strip():
                logger.warning(f"No text found for topic {community_id}")
                continue
            
            # Create summarization task
            task = SummarizationTask(
                task_id=f"topic_{community_id}",
                community_id=community_id,
                subcommunity_id=None,
                is_topic=True,
                concatenated_text=concatenated_text,
                chunk_ids=sorted_chunk_ids,
                entity_ids=entity_ids
            )
            
            tasks.append(task)
            logger.info(f"Created task for topic {community_id}: {len(entity_ids)} entities, {len(sorted_chunk_ids)} chunks, {len(concatenated_text)} chars")
            
        except Exception as e:
            logger.error(f"Error creating task for topic {topic_node_id}: {e}")
            continue
    
    return tasks

async def collect_subcommunity_tasks_async(graph: nx.DiGraph) -> List[SummarizationTask]:
    """Collect summarization tasks for all subtopic nodes"""
    
    tasks = []
    subtopic_nodes = []
    
    # Find all subtopic nodes (SUBTOPIC_X_Y)
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'SUBTOPIC' and 
            isinstance(node_id, str) and 
            node_id.startswith('SUBTOPIC_')):
            subtopic_nodes.append((node_id, node_data))
    
    logger.info(f"Found {len(subtopic_nodes)} subtopic nodes for summarization")
    
    for subtopic_node_id, _ in subtopic_nodes:
        try:
            # Extract IDs from node name (e.g., "SUBTOPIC_0_1" -> topic=0, subtopic=1)
            parts = subtopic_node_id.split('_')
            if len(parts) >= 3:
                community_id = int(parts[1])
                subcommunity_id = int(parts[2])
            else:
                logger.warning(f"Invalid subtopic node name format: {subtopic_node_id}")
                continue
            
            # Find entities for this subtopic
            # Structure is Entity -> Subtopic (incoming edges)
            entity_ids = []
            if graph.has_node(subtopic_node_id):
                for pred in graph.predecessors(subtopic_node_id):
                    if graph.nodes[pred].get('node_type') in ['ENTITY', 'ENTITY_CONCEPT']:
                        entity_ids.append(pred)
            
            if not entity_ids:
                logger.warning(f"No entities found for subtopic {community_id}_{subcommunity_id}")
                continue
            
            # Find chunks for entities
            chunk_ids = await find_chunks_for_entities_async(graph, entity_ids)
            
            if not chunk_ids:
                logger.warning(f"No chunks found for subtopic {community_id}_{subcommunity_id}")
                continue
            
            # Sort chunks by global order
            sorted_chunk_ids = await sort_chunks_by_global_order_async(graph, chunk_ids)
            
            # Concatenate chunk texts
            concatenated_text = await concatenate_chunk_texts_async(graph, sorted_chunk_ids)
            
            if not concatenated_text.strip():
                logger.warning(f"No text found for subtopic {community_id}_{subcommunity_id}")
                continue
            
            # Create summarization task
            task = SummarizationTask(
                task_id=f"subtopic_{community_id}_{subcommunity_id}",
                community_id=community_id,
                subcommunity_id=subcommunity_id,
                is_topic=False,
                concatenated_text=concatenated_text,
                chunk_ids=sorted_chunk_ids,
                entity_ids=entity_ids
            )
            
            tasks.append(task)
            logger.info(f"Created task for subtopic {community_id}_{subcommunity_id}: {len(entity_ids)} entities, {len(sorted_chunk_ids)} chunks, {len(concatenated_text)} chars")
            
        except Exception as e:
            logger.error(f"Error creating task for subtopic {subtopic_node_id}: {e}")
            continue
    
    return tasks

async def generate_title_and_summary_with_semaphore(llm: Any, task: SummarizationTask, semaphore: asyncio.Semaphore) -> SummarizationTask:
    """Generate title and summary for a task with semaphore control"""
    
    async with semaphore:
        try:
            logger.info(f"Processing task {task.task_id}...")
            title_task = generate_title_internal(llm, task.concatenated_text)
            summary_task = summarize_text_internal(llm, task.concatenated_text)
            title, summary = await asyncio.gather(title_task, summary_task, return_exceptions=True)
            
            if isinstance(title, Exception):
                logger.error(f"Title generation exception: {title}")
                title = "Untitled Topic"
            if isinstance(summary, Exception):
                logger.error(f"Summary generation exception: {summary}")
                summary = "No summary available."
            
            task.title = title
            task.summary = summary
            logger.info(f"Completed task {task.task_id}: '{title[:50]}...'")
        except Exception as e:
            logger.error(f"Failed to process task {task.task_id}: {e}")
            task.title = "Processing Failed"
            task.summary = f"Error during processing: {str(e)}"
    
    return task

async def process_all_summarization_tasks_internal(llm: Any, tasks: List[SummarizationTask]) -> Dict[str, Any]:
    """Process all summarization tasks in parallel with semaphore control"""
    
    if not tasks:
        return {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "errors": ["No tasks to process"]
        }
    
    logger.info(f"Processing {len(tasks)} summarization tasks with max concurrency {MAX_CONCURRENT_SUMMARIES}")
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIES)
    
    # Process all tasks concurrently
    processed_tasks = await asyncio.gather(*[
        generate_title_and_summary_with_semaphore(llm, task, semaphore)
        for task in tasks
    ])
    
    # Count results
    completed = sum(1 for task in processed_tasks if task.title and task.title != "Processing Failed")
    failed = len(processed_tasks) - completed
    
    logger.info(f"Summarization complete: {completed}/{len(processed_tasks)} successful")
    
    # Cleanup with retry
    async def _cleanup_llm():
        for i in range(3):
            try:
                await asyncio.sleep(0.2 * (i + 1))
                if hasattr(llm, 'async_client') and llm.async_client:
                    await llm.async_client.aclose()
                if hasattr(llm, 'client') and hasattr(llm.client, 'aclose'):
                    await llm.client.aclose()
                return
            except (RuntimeError, AttributeError) as e:
                if "Event loop is closed" in str(e) and i == 2:
                    return
            except Exception:
                if i < 2:
                    continue
    
    await _cleanup_llm()
    
    return {
        "tasks_processed": len(processed_tasks),
        "tasks_completed": completed,
        "tasks_failed": failed,
        "processed_tasks": processed_tasks,
        "errors": []
    }

async def update_community_node_with_summary_async(graph: nx.DiGraph, topic_node_id: str,
                                                  title: str = "", summary: str = "", chunk_ids: List[str] = None, 
                                                  entity_ids: List[str] = None) -> str:
    """Update a topic node with title and summary"""
    
    if not graph.has_node(topic_node_id):
        logger.warning(f"Topic node {topic_node_id} not found in graph")
        return topic_node_id
    
    # Update node data
    node_data = graph.nodes[topic_node_id]
    if title:
        node_data['title'] = title
        node_data['name'] = title  # Set name field to title
    if summary:
        node_data['summary'] = summary
    if chunk_ids:
        node_data['chunk_ids'] = chunk_ids
    if entity_ids:
        node_data['entity_ids'] = entity_ids
    
    # Mark as updated
    node_data['has_summary'] = True
    node_data['updated_at'] = datetime.now().isoformat()
    
    logger.info(f"Updated {topic_node_id} with title: '{title[:50]}...'")
    
    return topic_node_id

async def create_all_topic_nodes(graph: nx.DiGraph, processed_tasks: List[SummarizationTask]) -> Dict[str, Any]:
    """Create topic and subtopic nodes from processed tasks"""
    
    topics_updated = 0
    subtopics_updated = 0
    
    for task in processed_tasks:
        try:
            if task.is_topic:
                # Update topic node
                topic_node_id = f"TOPIC_{task.community_id}"
                await update_community_node_with_summary_async(
                    graph, topic_node_id, task.title, task.summary, 
                    task.chunk_ids, task.entity_ids
                )
                topics_updated += 1
            else:
                # Update subtopic node
                subtopic_node_id = f"SUBTOPIC_{task.community_id}_{task.subcommunity_id}"
                await update_community_node_with_summary_async(
                    graph, subtopic_node_id, task.title, task.summary, 
                    task.chunk_ids, task.entity_ids
                )
                subtopics_updated += 1
                
        except Exception as e:
            logger.error(f"Failed to update node for task {task.task_id}: {e}")
    
    logger.info(f"Updated {topics_updated} topics and {subtopics_updated} subtopics")
    
    return {
        "topics_updated": topics_updated,
        "subtopics_updated": subtopics_updated
    }

async def get_all_topic_nodes_async(graph: nx.DiGraph) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all topic and subtopic nodes that have titles"""
    
    topic_nodes = []
    
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') in ['TOPIC', 'SUBTOPIC'] and 
            node_data.get('title')):
            topic_nodes.append((node_id, node_data))
    
    return topic_nodes



async def generate_community_summaries(graph: nx.DiGraph, llm: Any) -> Dict[str, Any]:
    """
    Generate summaries for topics and subtopics using the provided LLM.
    This function orchestrates the summarization workflow:
    1. Collect tasks
    2. Process tasks
    3. Update nodes
    """
    start_time = datetime.now()
    
    try:
        # Phase 1: Collect summarization tasks
        logger.info("Phase 1: Collecting summarization tasks...")
        community_tasks = await collect_community_tasks_async(graph)
        subcommunity_tasks = await collect_subcommunity_tasks_async(graph)
        
        all_tasks = community_tasks + subcommunity_tasks
        logger.info(f"Created {len(all_tasks)} summarization tasks")
        
        if not all_tasks:
            return {"processed": 0, "errors": ["No tasks created"]}
            
        # Phase 2: Process tasks
        logger.info(f"Phase 2: Processing {len(all_tasks)} tasks...")
        summary_result = await process_all_summarization_tasks_internal(llm, all_tasks)
        
        # Phase 3: Update nodes
        logger.info("Phase 3: Updating topic nodes...")
        creation_result = await create_all_topic_nodes(graph, summary_result["processed_tasks"])
        
        return {
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "total_topics": len(community_tasks),
            "total_subtopics": len(subcommunity_tasks),
            "topics_updated": creation_result["topics_updated"],
            "subtopics_updated": creation_result["subtopics_updated"]
        }
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        return {"error": str(e)}
