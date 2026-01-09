from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import networkx as nx
from langchain_groq import ChatGroq

@dataclass
class SummarizationTask:
    """Task for generating title and summary for a topic/subtopic"""
    task_id: str
    community_id: int
    subcommunity_id: Optional[int]
    is_topic: bool
    concatenated_text: str
    chunk_ids: List[str]
    entity_ids: List[str]
    title: Optional[str] = None
    summary: Optional[str] = None



@dataclass
class SimilarTopicPair:
    """Represents a pair of similar topics found by Levenshtein analysis"""
    topic1_id: str
    topic1_title: str
    topic1_level: str
    topic2_id: str
    topic2_title: str
    topic2_level: str
    similarity_score: float
    levenshtein_distance: int
    is_potential_duplicate: bool
