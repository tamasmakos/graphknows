"""
Graph Extractor Abstraction Layer.

Provides a unified interface for different graph extraction backends:
- LangChain LLMGraphTransformer
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
import time
# from gliner import GLiNER

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.kg.llm import get_langchain_llm

logger = logging.getLogger(__name__)



DEFAULT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at extracting knowledge graph entities and relationships from text.
    
    Focus on extracting information relevant to building a "Life Graph" or memory graph for the user.
    Strictly identify and extract:
    1. Life Patterns & Habits: Recurring activities, behaviors, or routines (e.g., "goes for a run every morning", "drinks coffee at 8am").
    2. Things to Remember: Specific preferences, tasks, deadlines, or important details (e.g., "allergic to peanuts", "meeting on Friday").
    3. Entities: People, Places, Organizations, Concepts involved in the user's life.
    4. Contextual Relations: How these entities relate to the user's daily life (e.g., LIVES_IN, VISITS, HAS_HABIT, PREFERS, OWNS).
    
    When extracting relationships, use descriptive types like:
    - HAS_HABIT
    - IS_A
    - LOCATED_AT
    - OCCURRED_AT
    - INVOLVES
    - HAS_PREFERENCE
    - REMINDER_FOR
    
    Text:
    {input}
    """
)

class BaseExtractor(ABC):
    """Base class for graph extractors."""
    
    @abstractmethod
    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relations from text.
        
        Args:
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
        Returns:
            List of (source, relation_type, target) triplets
        """
        pass
    
    async def close(self):
        """Cleanup resources."""
        pass


class LangChainExtractor(BaseExtractor):
    """LangChain LLMGraphTransformer-based extractor."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config
        # Initialize GLiNER for entity extraction
        # gliner_model = config.get('gliner_model', 'urchade/gliner_medium-v2.1')
        # self.gliner = GLiNER.from_pretrained(gliner_model)
        # self.entity_labels = config.get('entity_labels', ["person", "organization", "location", "event", "concept", "product", "date", "time"])
        # logger.info(f"Initialized LangChain extractor with GLiNER ({gliner_model})")
    
    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> List[Tuple[str, str, str]]:
        """Extract relations using LangChain LLMGraphTransformer."""
        entities = entities or []
        abstract_concepts = abstract_concepts or []
        
        # Use GLiNER to extract entities when none provided
        # if not entities and not abstract_concepts:
        #     gliner_entities = self.gliner.predict_entities(text, self.entity_labels, threshold=0.4)
        #     entities = list(set(ent["text"] for ent in gliner_entities))
        #     if not entities:
        #         return []
        
        allowed_nodes = list(set(entities + abstract_concepts))
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = DEFAULT_EXTRACTION_PROMPT  # Use our life-graph focused default
        
        def _extract_sync():
            # Initialize LLM in the worker thread to ensure event loop safety
            llm = get_langchain_llm(self.config, purpose='extraction')
            
            transformer = LLMGraphTransformer(
                llm=llm,
                # allowed_nodes=allowed_nodes,
                prompt=prompt,
                strict_mode=False,
                node_properties=False,
                relationship_properties=False
            )
            
            document = Document(page_content=text)
            return transformer.convert_to_graph_documents([document])

        retries = 3
        retry_delay = 1
        
        for attempt in range(retries):
            try:
                # Run in executor to avoid blocking
                graph_docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _extract_sync
                )
                
                if not graph_docs:
                    return []
                
                # Extract triplets
                relations = []
                for graph_doc in graph_docs:
                    for relationship in graph_doc.relationships:
                        source = relationship.source.id
                        target = relationship.target.id
                        relation_type = relationship.type
                        relations.append((source, relation_type, target))
                
                return relations
                
            except Exception as e:
                # Check for 400 Bad Request / Tool use failed
                error_str = str(e)
                if "400" in error_str or "tool_use_failed" in error_str or "BadRequest" in error_str:
                    logger.warning(f"LangChain extraction failed with 400/Tool Error (attempt {attempt+1}/{retries}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # Last retry for 400 error, return empty
                        logger.error(f"All {retries} retries exhausted for 400/Tool Error")
                        return []
                
                # For other errors, log and retry or return empty
                logger.error(f"LangChain extraction failed: {e}", exc_info=True)
                if attempt < retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return []
        
        return []

def get_extractor(config: Dict[str, Any]) -> BaseExtractor:
    """
    Factory function to get the appropriate extractor based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured extractor instance
    """
    extractor_type = 'langchain'
    
    logger.info(f"Initializing graph extractor: {extractor_type}")
    
    return LangChainExtractor(config)
