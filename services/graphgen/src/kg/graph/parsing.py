from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SegmentData(BaseModel):
    segment_id: str
    text: str
    metadata: Dict[str, Any] = {}
