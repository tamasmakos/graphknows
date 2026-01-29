from typing import Dict, List, Any, Optional, Union
import json
import logging
import time
import re

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field

from src.infrastructure.llm import get_llm, get_embedding_model
from src.infrastructure.graph_db import get_database_client, GraphDB
from src.infrastructure.config import get_app_config
from src.services.graph_retriever import (
    get_seed_entities,
    expand_subgraph,
    filter_subgraph_by_centrality,
    enrich_with_triplets,
    merge_graph_data,
)
from src.services.context_builder import format_graph_context

logger = logging.getLogger(__name__)

# --- Events ---

class InputEvent(Event):
    query: str
    messages: List[ChatMessage] = []

class KeywordsEvent(Event):
    keywords: List[str]
    query: str

class SeedsEvent(Event):
    seed_entities: List[str]
    seed_topics: List[str]
    query: str

class RefineQueryEvent(Event):
    original_query: str
    reason: str

class ContextEvent(Event):
    nodes: Dict[str, Any]
    edges: List[Dict[str, Any]]
    query: str
    context_str: str

# --- State ---

class GraphWorkflowState(BaseModel):
    accumulated_nodes: Dict[str, Any] = Field(default_factory=dict)
    accumulated_edges: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    trace_info: List[str] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)
    seed_topics: List[str] = Field(default_factory=list)
    step_timings: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

# --- Workflow ---

class GraphWorkflow(Workflow):
    def __init__(self, timeout: int = 60, verbose: bool = False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.config = get_app_config()
    
    def _get_db(self) -> GraphDB:
        # Helper to get a fresh DB connection
        return get_database_client(self.config, "falkordb")


    @step
    async def extract_keywords(self, ctx: Context, ev: StartEvent) -> KeywordsEvent:
        """
        Step 1: Analyze query and extract keywords.
        """
        t0 = time.time()
        query = ev.get("query")
        # Ensure messages are ChatMessage objects
        raw_messages = ev.get("messages", [])
        messages = []
        for m in raw_messages:
            if isinstance(m, ChatMessage):
                messages.append(m)
            elif hasattr(m, "role") and hasattr(m, "content"):
                messages.append(ChatMessage(role=m.role, content=m.content))
            elif isinstance(m, dict):
                 messages.append(ChatMessage(role=m.get("role"), content=m.get("content")))

        # Initialize state
        self.state = GraphWorkflowState(conversation_history=messages)
        
        logger.info(f"Step 1: Extracting keywords for query: {query}")
        
        llm = get_llm(purpose="keywords")
        if not llm:
            logger.error("No LLM available")
            self.state.step_timings["extract_keywords"] = time.time() - t0
            return KeywordsEvent(keywords=[], query=query)

        prompt_str = f"""
            Extract the key entities and important terms from the following user query.
            Return ONLY a JSON object with a "keywords" key containing a list of strings.
            Do not include generic terms like "what", "who", "where", "tell me about".
            Focus on specific names, locations, concepts, organizations, and key adjectives/attributes.

            Query: {query}
            
            JSON Output:
        """
        
        try:
            response = await llm.acomplete(prompt_str)
            content = response.text
            
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            keywords = []
            try:
                json_match = re.search(r'\{.*?\}', content, flags=re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    keywords = parsed.get("keywords", [])
                else:
                    logger.warning(f"Could not parse JSON keywords from: {content[:100]}")
            except Exception as e:
                logger.error(f"JSON parse error: {e}")
            
            if not keywords:
                # Fallback
                keywords = [w for w in query.split() if len(w) > 3]
            
            self.state.trace_info.append(f"Keywords: {keywords}")
            self.state.step_timings["extract_keywords"] = time.time() - t0
            
            return KeywordsEvent(keywords=keywords, query=query)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            self.state.step_timings["extract_keywords"] = time.time() - t0
            return KeywordsEvent(keywords=[], query=query)

    @step
    async def identify_seeds(self, ctx: Context, ev: KeywordsEvent) -> SeedsEvent:
        """
        Step 2: Find seed entities in the graph.
        """
        logger.info(f"Step 2: Identifying seeds for keywords: {ev.keywords}")
        t0 = time.time()
        
        db = self._get_db()
        try:
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(ev.query)
            self.state.step_timings["query_embedding"] = time.time() - t0 # Partial time
            
            seed_entities, timings = get_seed_entities(db, query_embedding, ev.keywords)
            
            # Merge timings
            for k, v in timings.items():
                self.state.step_timings[f"seed_{k}"] = v
            
            self.state.seed_entities = seed_entities
            self.state.trace_info.append(f"Seeds Found: {len(seed_entities)}")
            self.state.step_timings["identify_seeds_total"] = time.time() - t0
            
            return SeedsEvent(seed_entities=seed_entities, seed_topics=[], query=ev.query)
            
        finally:
            db.close()

    @step
    async def expand_graph(self, ctx: Context, ev: SeedsEvent) -> Union[ContextEvent, RefineQueryEvent]:
        """
        Step 3: Expand subgraph from seeds.
        """
        logger.info(f"Step 3: Expanding graph from {len(ev.seed_entities)} seeds")
        t0 = time.time()
        
        if not ev.seed_entities:
             self.state.step_timings["expand_graph"] = time.time() - t0
             return RefineQueryEvent(original_query=ev.query, reason="No seed entities found")
        
        db = self._get_db()
        try:
            nodes, edges, timings = expand_subgraph(db, ev.seed_entities)
            
            # Merge timings
            for k, v in timings.items():
                self.state.step_timings[f"expand_{k}"] = v
            
            enrich_with_triplets(nodes, edges)
            
            nodes, edges = filter_subgraph_by_centrality(nodes, edges, ev.seed_entities)
            
            if not nodes:
                 self.state.step_timings["expand_graph"] = time.time() - t0
                 return RefineQueryEvent(original_query=ev.query, reason="Expansion yielded no nodes")

            self.state.accumulated_nodes = nodes
            self.state.accumulated_edges = edges
            self.state.trace_info.append(f"Expanded: {len(nodes)} nodes, {len(edges)} edges")
            
            context_str = format_graph_context(nodes, edges)
            
            # Add token count estimate (4 chars/token is standard approximation for English)
            token_count = len(context_str) // 4
            self.state.trace_info.append(f"Total Tokens: {token_count}")
            
            self.state.step_timings["expand_graph_total"] = time.time() - t0
            return ContextEvent(nodes=nodes, edges=edges, query=ev.query, context_str=context_str)
            
        finally:
            db.close()

    @step
    async def refine_query(self, ctx: Context, ev: RefineQueryEvent) -> KeywordsEvent:
        """
        Step 3b: Refine query if needed.
        """
        logger.info(f"Refining query: {ev.original_query} because {ev.reason}")
        
        keywords = ev.original_query.split()
        return KeywordsEvent(keywords=keywords, query=ev.original_query)

    @step
    async def synthesize_answer(self, ctx: Context, ev: ContextEvent) -> StopEvent:
        """
        Step 4: Generate final answer.
        """
        t0 = time.time()
        logger.info("Step 4: Synthesizing answer")
        
        history = list(self.state.conversation_history)
        
        system_prompt = f"""You are a Personal Life Assistant. You help users recall memories, understand their daily patterns, and answer questions about their life based on their logs.

You have access to a knowledge graph structured in XML tags:
- <topics>: High-level themes of the user's life.
- <timeline>: Chronological events (Days, Segments, Conversations).
- <entities>: Key people, places, and concepts.
- <relationships>: Connections between entities.
- <source_documents>: Full text transcripts and descriptions.

**Guidelines:**
- Answer differently based on the user's question type (memory recall, pattern analysis, etc.).
- BE PERSONAL. Use "you" and "your".
- Cite dates and times specifically.
- Use the <timeline> to order events chronologically.
- If asking about a specific day, summarize the flow of that day.
- If information is missing, say so gently.

**Available Information:**
{ev.context_str}
"""
        
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]
        
        # Append history (ensure they are ChatMessage objects)
        messages.extend(history)
            
        messages.append(ChatMessage(role=MessageRole.USER, content=ev.query))
        
        llm = get_llm(purpose="chat")
        
        # Using achat strictly
        response = await llm.achat(messages)
        answer = response.message.content
        
        self.state.step_timings["synthesize_answer"] = time.time() - t0
        
        return StopEvent(result={
            "answer": answer,
            "context": ev.context_str,
            "graph_data": {"nodes": list(ev.nodes.values()), "edges": ev.edges},
            "trace": self.state.trace_info,
            "step_timings": self.state.step_timings,
            "seed_entities": self.state.seed_entities,
            "seed_topics": self.state.seed_topics
        })

