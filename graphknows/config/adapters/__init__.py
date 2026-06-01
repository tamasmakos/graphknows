"""graphknows adapters — port interfaces and concrete infrastructure adapters.

Ports (protocols) define the boundary between application and infrastructure.
Concrete adapters implement the ports by wrapping the existing store classes.
"""

from graphknows.config.adapters.dspy_relation_extractor import DspyRelationExtractorAdapter
from graphknows.config.adapters.ports.embedder import IEmbedder
from graphknows.config.adapters.ports.extractor import IEntityExtractor
from graphknows.config.adapters.ports.graph_db import IGraphDb
from graphknows.config.adapters.ports.llm import ILlm
from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor
from graphknows.config.adapters.ports.session_db import ISessionDb
from graphknows.config.adapters.ports.vector_db import IVectorDb

__all__ = [
    "DspyRelationExtractorAdapter",
    "IEmbedder",
    "IEntityExtractor",
    "IGraphDb",
    "ILlm",
    "IRelationExtractor",
    "ISessionDb",
    "IVectorDb",
]
