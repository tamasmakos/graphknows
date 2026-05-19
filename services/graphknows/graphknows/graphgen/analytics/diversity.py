"""
Topic Diversity Metrics.

Calculates:
- Topic Diversity (proportion of unique words in top-N words of all topics)
- Semantic Diversity (embedding distance)
"""

import numpy as np
from typing import List, Dict

def calculate_topic_diversity(topics_top_words: List[List[str]], top_k: int = 10) -> float:
    """
    Kullback-Leibler divergence or simple unique word ratio.
    Standard metric: Ratio of unique words in the combined vocab of all top-k lists.
    TD = |Unique(Union(TopKw))| / (K * T)
    """
    if not topics_top_words:
        return 0.0
        
    unique_words = set()
    total_words = 0
    
    for words in topics_top_words:
        # Take top k
        selection = words[:top_k]
        unique_words.update(selection)
        total_words += len(selection)
        
    if total_words == 0:
        return 0.0
        
    return len(unique_words) / total_words

def calculate_inverted_rbo(topics_top_words: List[List[str]]) -> float:
    """
    Inverted Rank Biased Overlap. 
    Can be used as a diversity measure (1 - RBO).
    """
    # Placeholder
    return 0.0
