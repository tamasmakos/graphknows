import re
from typing import List, Dict, Any, Optional
from datetime import date
from graphgen.data_types import SegmentData
from graphgen.utils.parsers.base import BaseDocumentParser

class RegexParser(BaseDocumentParser):
    """
    Generic Parser that uses Regex to split documents into segments.
    """
    def __init__(self, 
                 segment_splitter: str, # Regex pattern to split segments
                 attributes_map: Dict[str, str], # Map regex groups to attributes
                 file_date_pattern: str = None
                 ):
        self.segment_splitter = segment_splitter
        self.attributes_map = attributes_map
        self.file_date_pattern = file_date_pattern

    def parse(self, content: str, filename: str, doc_date: date) -> List[SegmentData]:
        # 1. Split content
        # Note: re.split might consume delimiters. We might want finditer.
        # Implemenation using finditer to capture attributes
        
        segments = []
        matches = list(re.finditer(self.segment_splitter, content, re.MULTILINE | re.DOTALL))
        
        if not matches:
             # Treat whole file as one segment if no match
             return [SegmentData(
                 segment_id=f"{filename}_0",
                 content=content,
                 line_number=1,
                 date=doc_date,
                 metadata={}
             )]

        for i, match in enumerate(matches):
            full_match = match.group(0)
            
            # Text is often what follows the match header, or the group named 'text'
            if 'text' in match.groupdict():
                segment_text = match.group('text').strip()
            else:
                # Fallback: The match itself is the header, content is between this and next match?
                # This simple regex parser assumes the regex matches the Whole Segment (Header + Content)
                # OR specific groups capture the content.
                segment_text = full_match.strip()
                
            metadata = {}
            for attr_name, group_name in self.attributes_map.items():
                if group_name in match.groupdict():
                    metadata[attr_name] = match.group(group_name).strip()
            
            segments.append(SegmentData(
                segment_id=f"{filename}_{i}",
                content=segment_text,
                line_number=match.start(),
                date=doc_date,
                metadata=metadata
            ))
            
        return segments

    def extract_date(self, filename: str) -> Optional[str]:
        if self.file_date_pattern:
            match = re.search(self.file_date_pattern, filename)
            if match:
                return match.group(1)
        return None

    def supports_file(self, filename: str) -> bool:
        return True # Supports anything it's thrown at
