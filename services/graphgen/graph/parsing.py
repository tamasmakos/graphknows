from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import date

@dataclass
class SegmentData:
    """Standardized data structure for a segment in a document.
    
    This structure is domain-agnostic and can represent paragraphs,
    segments, or any sequential content units.
    """
    segment_id: str
    content: str
    date: date
    line_number: int
    
    # Optional metadata
    sentiment: Optional[float] = None
    
    # Additional metadata dict for flexibility
    metadata: Dict[str, Any] = field(default_factory=dict)
