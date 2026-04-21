from typing import Dict, Any, List
from kg.graph.parsing import SegmentData

class BaseDocumentParser:
    def __init__(self, **kwargs):
        pass

    def parse(self, filepath: str, **kwargs) -> List[SegmentData]:
        raise NotImplementedError("Subclasses must implement parse")
