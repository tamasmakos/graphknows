"""Base parser interface for document processing.

This module defines the abstract base class for all document parsers,
enabling a pluggable parser system that can handle various document formats.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date
from kg.graph.parsing import SegmentData


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers.
    
    All document parsers should extend this class and implement its methods.
    This enables the pipeline to work with any document format through a
    consistent interface.
    """
    
    @abstractmethod
    def parse(self, content: str, filename: str, doc_date: date) -> List[SegmentData]:
        """Parse document content into a list of segments.
        
        Args:
            content: The full text content of the document
            filename: The name of the file being parsed
            doc_date: The date extracted from the filename or metadata
            
        Returns:
            List of SegmentData objects representing segments in the document
        """
        pass
    
    @abstractmethod
    def extract_date(self, filename: str) -> Optional[str]:
        """Extract date string from filename.
        
        Args:
            filename: The name of the file to extract date from
            
        Returns:
            Date string in YYYY-MM-DD format, or None if no date found
        """
        pass
    
    @abstractmethod
    def supports_file(self, filename: str) -> bool:
        """Check if this parser supports the given file.
        
        Args:
            filename: The name of the file to check
            
        Returns:
            True if this parser can handle the file, False otherwise
        """
        pass

    def extract_date_from_content(self, content: str) -> Optional[str]:
        """Extract date string from document content.
        
        Args:
            content: The full text content of the document
            
        Returns:
            Date string in YYYY-MM-DD format, or None if no date found
        """
        return None
    

