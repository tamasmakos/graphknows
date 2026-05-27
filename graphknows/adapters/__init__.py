"""graphknows adapters — port interfaces and concrete infrastructure adapters.

Ports (protocols) define the boundary between application and infrastructure.
Concrete adapters implement the ports by wrapping the existing store classes.
"""

from graphknows.adapters.ports.embedder import IEmbedder
from graphknows.adapters.ports.extractor import IEntityExtractor
from graphknows.adapters.ports.graph_db import IGraphDb
from graphknows.adapters.ports.llm import ILlm
from graphknows.adapters.ports.session_db import ISessionDb
from graphknows.adapters.ports.vector_db import IVectorDb

__all__ = [
    "IEmbedder",
    "IEntityExtractor",
    "IGraphDb",
    "ILlm",
    "ISessionDb",
    "IVectorDb",
]
