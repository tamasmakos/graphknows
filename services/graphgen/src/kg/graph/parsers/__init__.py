"""Parser registry and factory for document parsers.

This module provides a central registry for all available document parsers
and a factory function to get the appropriate parser based on configuration
or file detection.
"""

from typing import Optional, Dict, Type
from kg.graph.parsers.base import BaseDocumentParser
from kg.graph.parsers.generic import GenericParser
from kg.graph.parsers.life import LifeLogParser

__all__ = [
    'BaseDocumentParser',
    'GenericParser',
    'LifeLogParser',
    'get_parser',
    'PARSER_REGISTRY'
]

# Parser registry mapping format names to parser classes
PARSER_REGISTRY: Dict[str, Type[BaseDocumentParser]] = {
    'life': LifeLogParser,
    'generic': GenericParser,
}


def get_parser(format_name: Optional[str] = None, 
               filename: Optional[str] = None,
               **parser_kwargs) -> BaseDocumentParser:
    """Get a parser instance based on format name or auto-detection.
    
    Args:
        format_name: Name of the format ('generic', 'auto')
                    If None or 'auto', will attempt auto-detection
        filename: Optional filename for auto-detection
        **parser_kwargs: Additional arguments to pass to parser constructor
        
    Returns:
        Parser instance
        
    Examples:
        # Get default generic parser
        parser = get_parser()
        
        # Auto-detect from filename
        parser = get_parser('auto', filename='2023-02-20_doc.txt')
    """
    # Default to generic if no format specified
    if format_name is None or format_name == 'generic':
        return GenericParser(**parser_kwargs)
    
    # Auto-detection mode
    if format_name == 'auto' and filename:
        # Try each parser's supports_file method
        for parser_name, parser_class in PARSER_REGISTRY.items():
            parser_instance = parser_class(**parser_kwargs)
            if parser_instance.supports_file(filename):
                return parser_instance
        # Fall back to generic if no match
        return GenericParser(**parser_kwargs)
    
    # Get specific parser by name
    if format_name in PARSER_REGISTRY:
        return PARSER_REGISTRY[format_name](**parser_kwargs)
    
    # Unknown format, default to generic
    return GenericParser(**parser_kwargs)
