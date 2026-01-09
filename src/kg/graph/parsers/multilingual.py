"""
Multilingual Document Parser.

This parser extends the generic tab-separated parser to automatically detect language
and tag segments, enabling routing to specific embedding models.
"""

import logging
import re
from typing import List, Optional
from datetime import date
from langdetect import detect, LangDetectException

from src.kg.graph.parsers.generic import GenericParser
from src.kg.graph.parsing import SegmentData

logger = logging.getLogger(__name__)

class MultilingualParser(GenericParser):
    """
    Parser that detects language for each segment.
    
    Inherits from GenericParser for basic parsing but adds language detection.
    Tag format provided in metadata: 'language': 'en' | 'zh' | etc.
    """
    
    def parse(self, content: str, filename: str, doc_date: date) -> List[SegmentData]:
        """
        Parse and detect language for segments.
        """
        # Use parent class to get basic segments
        segments = super().parse(content, filename, doc_date)
        
        # Augment with language detection
        for segment in segments:
            text = segment.content
            try:
                # Detect language
                lang = detect(text)
                
                # Normalize common Chinese codes
                if lang in ['zh-cn', 'zh-tw']:
                    lang = 'zh'
                    
                segment.metadata['language'] = lang
                
                # Optional: Add language tag to segment ID for uniqueness if needed?
                # segment.segment_id = f"{segment.segment_id}_{lang}"
                
            except LangDetectException:
                logger.warning(f"Could not detect language for segment {segment.segment_id}. Defaulting to 'en'.")
                segment.metadata['language'] = 'en'
            except Exception as e:
                logger.error(f"Error detecting language: {e}")
                segment.metadata['language'] = 'en'
                
        return segments

    def supports_file(self, filename: str) -> bool:
        """
        Reuse GenericParser's date-based detection.
        We might want to be more specific if we only want this for specific files,
        but for now, it can replace the GenericParser.
        """
        return super().supports_file(filename)
