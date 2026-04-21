from typing import Dict, Any, List
from kg.graph.parsing import SegmentData
from kg.graph.parsers.base import BaseDocumentParser

class LifeLogParser(BaseDocumentParser):
    def parse(self, filepath: str, **kwargs) -> List[SegmentData]:
        return []
