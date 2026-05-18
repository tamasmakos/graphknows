"""
GLiNER Label Resolution.

Resolves GLiNER entity extraction labels from configuration,
supporting manual labels, ontology-based extraction, or both.
"""

import logging
from typing import List, Dict, Any

from graphgen.utils.utils import standardize_label

logger = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "Person", 
    "Organization", 
    "Location", 
    "Policy", 
    "Political Party", 
    "Concept", 
    "Event"
]


def resolve_entity_labels(extraction_config: Dict[str, Any]) -> List[str]:
    """
    Resolve GLiNER labels from configuration.
    
    Supports multiple sources for entity labels:
    1. Manual labels from config (gliner_labels)
    2. Ontology-derived labels from RDF/OWL files
    3. Merged combination of both sources
    
    Args:
        extraction_config: Extraction configuration dictionary.
                          Expected to have 'gliner_labels' and optionally 'ontology' keys.
    
    Returns:
        List of unique entity labels for GLiNER extraction.
    """
    # Handle Pydantic models
    if hasattr(extraction_config, 'model_dump'):
        extraction_config = extraction_config.model_dump()
    
    # Get manual labels. If none are provided, keep a curated default seed so
    # ontology-only runs still preserve core extraction categories like PERSON and EVENT.
    manual_labels = extraction_config.get('entity_labels') or extraction_config.get('gliner_labels') or DEFAULT_LABELS
    
    # Check ontology configuration
    ontology_config = extraction_config.get('ontology', {})
    if hasattr(ontology_config, 'model_dump'):
        ontology_config = ontology_config.model_dump()
    
    ontology_enabled = ontology_config.get('enabled', False)
    
    result = []
    if not ontology_enabled:
        logger.debug(f"Using {len(manual_labels)} manual GLiNER labels")
        result = manual_labels
    else:
        # Extract labels from ontology
        ontology_labels = _extract_ontology_labels(ontology_config)
        
        if not ontology_labels:
            logger.warning("Ontology extraction enabled but no labels extracted, using manual labels")
            result = manual_labels
        else:
            # Merge or replace based on configuration
            merge_with_manual = ontology_config.get('merge_with_manual', True)
            
            if merge_with_manual:
                # Combine and deduplicate
                combined = list(set(manual_labels + ontology_labels))
                logger.info(f"Merged {len(manual_labels)} manual + {len(ontology_labels)} ontology = {len(combined)} total labels")
                result = sorted(combined)
            else:
                logger.info(f"Using {len(ontology_labels)} ontology-derived labels (replacing manual)")
                result = ontology_labels
        
    # Final safety check: if still empty, use defaults
    if not result:
        logger.warning(f"No entity labels resolved from manual or ontology sources. Using {len(DEFAULT_LABELS)} default labels.")
        result = DEFAULT_LABELS
    
    # Standardize all labels
    result = [standardize_label(label) for label in result]
    # Remove duplicates after standardization (e.g. "Person" and "person" both become "PERSON")
    result = sorted(list(set(result)))
        
    return result


def _extract_ontology_labels(ontology_config: Dict[str, Any]) -> List[str]:
    """
    Extract labels from ontology based on configuration.
    
    Args:
        ontology_config: Ontology configuration dictionary.
        
    Returns:
        List of labels extracted from ontology, or empty list on failure.
    """
    try:
        from graphgen.utils.ontology_parser import OntologyLabelExtractor
        
        ontology_dir = ontology_config.get('ontology_dir', 'input/ontology/cdm-4.13.2')
        namespace_filter = ontology_config.get('namespace_filter')
        include_local_names = ontology_config.get('include_local_names', True)
        top_level_only = ontology_config.get('top_level_only', True)
        min_subclasses = ontology_config.get('min_subclasses', 0)
        
        logger.info(f"Extracting labels from ontology: {ontology_dir} (top_level={top_level_only}, min_children={min_subclasses})")
        
        extractor = OntologyLabelExtractor(
            ontology_dir=ontology_dir,
            namespace_filter=namespace_filter
        )
        
        labels = extractor.extract_labels(
            include_local_names=include_local_names,
            top_level_only=top_level_only,
            min_subclasses=min_subclasses
        )
        
        logger.info(f"Extracted {len(labels)} labels from ontology")
        return labels
        
    except ImportError as e:
        logger.error(f"Failed to import ontology parser: {e}")
        return []
    except Exception as e:
        logger.error(f"Ontology label extraction failed: {e}")
        return []
