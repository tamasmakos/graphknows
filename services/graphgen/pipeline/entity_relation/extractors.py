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

from graphgen.config.llm import _extract_secret
from graphgen.config.llm import normalize_groq_model
from graphgen.pipeline.entity_relation.dspy_module import GraphExtractorModule
import dspy
import os

from graphgen.utils.utils import standardize_label

logger = logging.getLogger(__name__)


def _normalize_relation_candidate(value: Any) -> str:
    return standardize_label(value) if value else ""


def _candidate_grounded_in_evidence(candidate: str, evidence: str) -> bool:
    candidate_norm = _normalize_relation_candidate(candidate)
    evidence_norm = _normalize_relation_candidate(evidence)
    if not candidate_norm or not evidence_norm:
        return False
    return f"_{candidate_norm}_" in f"_{evidence_norm}_"


def _endpoint_matches_hint(endpoint: str, entity_hints: List[str]) -> bool:
    endpoint_norm = _normalize_relation_candidate(endpoint)
    if not endpoint_norm:
        return False

    endpoint_tokens = {token for token in endpoint_norm.split('_') if len(token) > 2}
    for hint in entity_hints or []:
        hint_norm = _normalize_relation_candidate(hint)
        if not hint_norm:
            continue
        if endpoint_norm == hint_norm:
            return True
        if hint_norm in endpoint_norm or endpoint_norm in hint_norm:
            return True

        hint_tokens = {token for token in hint_norm.split('_') if len(token) > 2}
        overlap = endpoint_tokens & hint_tokens
        if overlap and (len(overlap) >= 2 or len(endpoint_tokens) == 1 or len(hint_tokens) == 1):
            return True

    return False


def _is_grounded_relation_endpoint(
    endpoint: str,
    entity_hints: List[str],
    ontology_classes: List[str],
    evidence: str,
) -> bool:
    endpoint_norm = _normalize_relation_candidate(endpoint)
    if not endpoint_norm:
        return False

    if _endpoint_matches_hint(endpoint, entity_hints):
        return True

    if _candidate_grounded_in_evidence(endpoint, evidence):
        return True

    ontology_set = {_normalize_relation_candidate(item) for item in (ontology_classes or []) if item}
    if endpoint_norm in ontology_set:
        return False

    return False


def _is_ungrounded_relation_triplet(
    source: str,
    target: str,
    entity_hints: List[str],
    ontology_classes: List[str],
    evidence: str,
) -> bool:
    return not (
        _is_grounded_relation_endpoint(source, entity_hints, ontology_classes, evidence)
        and _is_grounded_relation_endpoint(target, entity_hints, ontology_classes, evidence)
    )


def _relation_endpoints_in_hints(source: str, target: str, entity_hints: List[str]) -> bool:
    return _endpoint_matches_hint(source, entity_hints) and _endpoint_matches_hint(target, entity_hints)


DEFAULT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(

    """You are an expert at extracting knowledge graph entities and relationships from text.    
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
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Extract relations from text.
        
        Args:
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
            text: Text to extract relations from
            custom_prompt: Optional custom prompt template
            keywords: Optional list of keywords to guide extraction
            entities: Optional list of entities to focus on (used by LangChain)
            abstract_concepts: Optional list of abstract concepts (used by LangChain)
            
        Returns:
            Tuple containing:
            - List of (source, relation_type, target, properties) tuples
            - List of extracted nodes with metadata (id, type, properties)
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
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """Extract relations using LangChain LLMGraphTransformer."""
        # abstract_concepts now contains the ontology labels (Types)
        allowed_nodes = abstract_concepts or []
        # entities contains the hints from GLiNER/Spacy
        discovered_hints = entities or []
        
        # Inject hints into the prompt if available
        hints_text = ""
        if discovered_hints:
            hints_text = f"\nPre-identified entities found in text: {', '.join(discovered_hints)}\n"
        
        prompt = custom_prompt or DEFAULT_EXTRACTION_PROMPT
        
        # We can wrap the prompt to include hints if needed, but for now let's use it as is
        # or append hints to the input text
        full_text = text
        if hints_text:
            full_text = f"{hints_text}\nInput Text:\n{text}"
        
        def _extract_sync():
            llm = get_langchain_llm(self.config, purpose='extraction')
            
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=allowed_nodes,
                prompt=prompt,
                strict_mode=True, # Set to True to strictly follow ontology
                node_properties=False,
                relationship_properties=False
            )
            
            document = Document(page_content=full_text)
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
                    return [], []
                
                # Extract triplets and nodes
                relations = []
                nodes_data = []
                
                for graph_doc in graph_docs:
                    # Extract relations
                    for relationship in graph_doc.relationships:
                        source = standardize_label(relationship.source.id)
                        target = standardize_label(relationship.target.id)
                        relation_type = standardize_label(relationship.type)
                        props = dict(relationship.properties or {})
                        relations.append((source, relation_type, target, props))
                        
                    # Extract nodes
                    for node in graph_doc.nodes:
                        nodes_data.append({
                            "id": standardize_label(node.id),
                            "type": standardize_label(node.type),
                            "properties": dict(node.properties or {})
                        })
                
                return relations, nodes_data
                
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
                        return [], []
                
                # For other errors, log and retry or return empty
                logger.error(f"LangChain extraction failed: {e}", exc_info=True)
                if attempt < retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return [], []
        
        return [], []


class DSPyExtractor(BaseExtractor):
    """DSPy-based extractor."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self.config = config

        # Configure DSPy with the LLM from config
        llm_config = config.get('llm', {})
        # Flatten if it's a pydantic model dump
        if hasattr(llm_config, 'model_dump'):
            llm_config = llm_config.model_dump()

        # Try to find the best model name from config
        model = llm_config.get('extraction_model') or llm_config.get('base_model') or llm_config.get('model') or 'gpt-4o'

        # Check for Groq API Key
        infra_config = config.get('infra', {})
        groq_api_key = _extract_secret(infra_config, 'groq_api_key') or _extract_secret(llm_config, 'groq_api_key')
        if not groq_api_key:
            groq_api_key = os.environ.get('GROQ_API_KEY')

        # Determine provider and configure
        try:
             if groq_api_key:
                 # Configure for Groq using OpenAI compatibility
                 logger.info(f"Configuring DSPy for Groq with model {model}")
                 groq_model = normalize_groq_model(model)
                 lm = dspy.LM(
                     model=groq_model,
                     api_key=groq_api_key,
                     api_base="https://api.groq.com/openai/v1",
                     temperature=0.0,
                     max_tokens=2048
                 )
                 dspy.configure(lm=lm)
             else:
                 # Fallback to OpenAI or other default
                 api_key = _extract_secret(llm_config, 'api_key') or os.environ.get('OPENAI_API_KEY')
                 base_url = llm_config.get('base_url')
                 lm = dspy.LM(
                     model=model,
                     api_key=api_key,
                     api_base=base_url,
                     temperature=0.0,
                     max_tokens=2048
                 )
                 dspy.configure(lm=lm)

        except Exception as e:
             logger.warning(f"Failed to configure DSPy LM: {e}")
             # Fallback or already configured
             pass

        self.module = GraphExtractorModule()
        logger.info(f"Initialized DSPy extractor with model {model}")

    async def extract_relations(
        self,
        text: str,
        custom_prompt: ChatPromptTemplate = None,
        keywords: List[str] = None,
        entities: List[str] = None,
        abstract_concepts: List[str] = None
    ) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
        """Extract relations using DSPy."""
        
        ontology_classes = abstract_concepts or []
        entity_hints = entities or []
        
        try:
            # DSPy calls are synchronous, so we run in a thread
            def _extract_sync():
                # The dspy module returns a Prediction object which has the output fields as attributes
                prediction = self.module(text=text, ontology_classes=ontology_classes, entity_hints=entity_hints)
                return prediction.triplets
                
            triplets = await asyncio.to_thread(_extract_sync)
            
            # Convert to expected format
            relations = []
            nodes_data = []
            seen_nodes = set()
            raw_triplets = []
            triplet_decisions = []
            
            # triplets is expected to be a list of Triplet objects (pydantic models)
            if triplets:
                for triplet in triplets:
                    # Handle both dict and object access just in case
                    if isinstance(triplet, dict):
                        source = triplet.get('source')
                        relation = triplet.get('relation')
                        target = triplet.get('target')
                        source_type = triplet.get('source_type')
                        target_type = triplet.get('target_type')
                        confidence = triplet.get('confidence', 1.0)
                        evidence = triplet.get('evidence', "")
                    else:
                        source = getattr(triplet, 'source', None)
                        relation = getattr(triplet, 'relation', None)
                        target = getattr(triplet, 'target', None)
                        source_type = getattr(triplet, 'source_type', None)
                        target_type = getattr(triplet, 'target_type', None)
                        confidence = getattr(triplet, 'confidence', 1.0)
                        evidence = getattr(triplet, 'evidence', "")

                    raw_triplet = {
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "source_type": source_type,
                        "target_type": target_type,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                    raw_triplets.append(raw_triplet)

                    if not (source and relation and target):
                        triplet_decisions.append({
                            **raw_triplet,
                            "kept": False,
                            "drop_reason": "missing_triplet_field",
                            "source_matches_hint": _endpoint_matches_hint(source, entity_hints),
                            "target_matches_hint": _endpoint_matches_hint(target, entity_hints),
                            "source_grounded": _is_grounded_relation_endpoint(source, entity_hints, ontology_classes, evidence),
                            "target_grounded": _is_grounded_relation_endpoint(target, entity_hints, ontology_classes, evidence),
                        })
                        continue

                    decision = {
                        **raw_triplet,
                        "kept": False,
                        "drop_reason": None,
                        "source_matches_hint": _endpoint_matches_hint(source, entity_hints),
                        "target_matches_hint": _endpoint_matches_hint(target, entity_hints),
                        "source_grounded": _is_grounded_relation_endpoint(source, entity_hints, ontology_classes, evidence),
                        "target_grounded": _is_grounded_relation_endpoint(target, entity_hints, ontology_classes, evidence),
                    }

                    if not _relation_endpoints_in_hints(source, target, entity_hints):
                        decision["drop_reason"] = "endpoint_not_grounded_in_hints"
                        triplet_decisions.append(decision)
                        logger.debug(
                            "Dropping DSPy triplet without both endpoints grounded in hints source=%s relation=%s target=%s",
                            source,
                            relation,
                            target,
                        )
                        continue
                    if _is_ungrounded_relation_triplet(
                        source,
                        target,
                        entity_hints,
                        ontology_classes,
                        evidence,
                    ):
                        decision["drop_reason"] = "endpoint_not_grounded_after_evidence_check"
                        triplet_decisions.append(decision)
                        logger.debug(
                            "Dropping ungrounded DSPy triplet source=%s relation=%s target=%s evidence=%s",
                            source,
                            relation,
                            target,
                            evidence,
                        )
                        continue

                    decision["kept"] = True
                    triplet_decisions.append(decision)

                    # Standardize upstream
                    source = standardize_label(source)
                    target = standardize_label(target)
                    relation = standardize_label(relation)
                    source_type = standardize_label(source_type) if source_type else "ENTITY"
                    target_type = standardize_label(target_type) if target_type else "ENTITY"
                    
                    props = {
                        "confidence": confidence,
                        "evidence": evidence
                    }
                    relations.append((source, relation, target, props))
                    
                    if source not in seen_nodes:
                        nodes_data.append({"id": source, "type": source_type, "properties": {}})
                        seen_nodes.add(source)
                    if target not in seen_nodes:
                        nodes_data.append({"id": target, "type": target_type, "properties": {}})
                        seen_nodes.add(target)
                        
            return relations, nodes_data, {
                "raw_triplets": raw_triplets,
                "triplet_decisions": triplet_decisions,
            }
            
        except Exception as e:
            logger.error(f"DSPy extraction failed: {e}", exc_info=True)
            return [], [], {}

def get_extractor(config: Dict[str, Any]) -> BaseExtractor:
    """
    Factory function to get the appropriate relation extractor based on config.

    The legacy `extraction.backend` flag is overloaded: values like `gliner`,
    `gliner2`, and `spacy` select the NER backend, while `llm`/`dspy` select
    relation extraction behavior. To preserve backward compatibility, any
    non-`llm` backend defaults to the DSPy relation extractor.
    """
    extraction_config = config.get('extraction', {})
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()

    ner_backend = extraction_config.get('ner_backend') or extraction_config.get('backend', 'dspy')
    relation_backend = extraction_config.get('relation_backend')
    if not relation_backend:
        relation_backend = 'langchain' if ner_backend == 'llm' else 'dspy'

    logger.info(
        f"Initializing graph extractor: relation_backend={relation_backend} "
        f"(configured backend={ner_backend})"
    )

    if relation_backend == 'dspy':
        return DSPyExtractor(config)

    return LangChainExtractor(config)
