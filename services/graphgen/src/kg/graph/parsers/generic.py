"""Generic document parser for tab-separated format.

This is the default parser for the document processing pipeline.
It handles a simple, flexible tab-separated format that works for any domain.

Format:
    {segment_id}\t{text_content}

Filename pattern:
    *_{YYYY-MM-DD}.txt or {YYYY-MM-DD}_*.txt
"""

import re
from typing import List, Optional
from datetime import date, datetime
from kg.graph.parsers.base import BaseDocumentParser
from kg.graph.parsing import SegmentData


class GenericParser(BaseDocumentParser):
    """Generic parser for simple tab-separated document format.
    
    This parser is domain-agnostic and works with any type of sequential
    documents organized by date. It expects:
    - Tab-separated lines: segment_id <TAB> content
    - Filename contains date in YYYY-MM-DD format
    
    Example:
        segment_001\tThis is the first segment content.
        segment_002\tThis is the second segment content.
    """
    
    def parse(self, content: str, filename: str, doc_date: date) -> List[SegmentData]:
        """Parse tab-separated content into segments.
        
        Args:
            content: Document content with tab-separated segments
            filename: Name of the file being parsed
            doc_date: Date of the document
            
        Returns:
            List of SegmentData objects
        """
        segments = []
        
        for line_num, line in enumerate(content.split('\n')):
            if not line.strip():
                continue
                
            # Tab-separated format: segment_id\tcontent
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
                
            segment_id_prefix = parts[0].strip()
            segment_content = parts[1].strip()
            
            if not segment_content:
                continue
                
            # Generate IDs
            segment_id = f"SEGMENT_{doc_date.isoformat()}_{segment_id_prefix}"
            
            segments.append(SegmentData(
                segment_id=segment_id,
                content=segment_content,
                date=doc_date,
                line_number=line_num,
                metadata={'original_segment_id': segment_id_prefix}
            ))
        
        return segments
    
    def extract_date(self, filename: str) -> Optional[str]:
        """Extract date from filename.
        
        Supports patterns:
        - prefix_YYYY-MM-DD.txt
        - YYYY-MM-DD_suffix.txt
        - prefix_YYYY-MM-DD_suffix.txt
        
        Args:
            filename: Filename to extract date from
            
        Returns:
            Date string in YYYY-MM-DD format or None
        """
        # Look for YYYY-MM-DD pattern anywhere in filename
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return date_match.group(1)
        return None
    
    def supports_file(self, filename: str) -> bool:
        """Check if filename contains a date pattern.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if file contains YYYY-MM-DD pattern
        """
        return self.extract_date(filename) is not None
