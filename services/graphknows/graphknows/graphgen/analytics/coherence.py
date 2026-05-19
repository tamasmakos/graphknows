"""
Topic Coherence Metrics.

Calculates:
- UMass (intrinsic, using document co-occurrence)
- UCI / NPMI (extrinsic, using reference corpus or sliding window)
- Cv (embedding based, usually)

Note: Calculating coherence properly requires access to the original text corpus or a reference corpus.
Here we assume we have access to the original documents through the graph or passed as arguments.
"""

import logging
import math
import numpy as np
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)

class TopicCoherence:
    """
    Calculates topic coherence metrics.
    """
    
    def __init__(self, texts: List[List[str]] = None):
        """
        Args:
            texts: List of tokenized documents (for calculating co-occurrence).
                   If None, coherence metrics requiring corpus cannot be run.
        """
        self.texts = texts
        self.word_counts: Dict[str, int] = {}
        self.co_occurrences: Dict[Tuple[str, str], int] = {}
        self.total_docs = 0
        if texts:
            self._build_vocab(texts)
            
    def _build_vocab(self, texts: List[List[str]]):
        self.total_docs = len(texts)
        for doc in texts:
            unique_words = set(doc)
            # Word counts
            for w in unique_words:
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
            
            # Co-occurrences (simple window=entire document for now)
            # Ideally use sliding window for larger docs
            sorted_words = sorted(list(unique_words))
            for i in range(len(sorted_words)):
                for j in range(i + 1, len(sorted_words)):
                    pair = (sorted_words[i], sorted_words[j])
                    self.co_occurrences[pair] = self.co_occurrences.get(pair, 0) + 1

    def get_co_occurrence(self, w1: str, w2: str) -> int:
        if w1 > w2:
            w1, w2 = w2, w1
        return self.co_occurrences.get((w1, w2), 0)

    def calculate_umass(self, topic_words: List[str]) -> float:
        """
        Calculates UMass coherence.
        Formula: sum_{i=2}^N sum_{j=1}^{i-1} log( (D(w_i, w_j) + 1) / D(w_i) )
        Prioritizes frequent words co-occurring.
        """
        if not self.texts:
            return 0.0
            
        score = 0.0
        # Usually calculate on top N words
        top_words = topic_words[:20] 
        
        for i in range(1, len(top_words)):
            w_i = top_words[i]
            for j in range(0, i):
                w_j = top_words[j]
                
                # UMass uses conditional probability P(wi|wj) empirically
                # count(wi, wj) is co-occurrence
                # count(wj) is doc frequency of wj
                
                count_wi_wj = self.get_co_occurrence(w_i, w_j)
                count_wj = self.word_counts.get(w_j, 0)
                
                if count_wj == 0:
                    continue
                    
                val = math.log( (count_wi_wj + 1.0) / count_wj )
                score += val
                
        return score

    def calculate_npmi(self, topic_words: List[str]) -> float:
        """
        Calculates Normalized Pointwise Mutual Information (NPMI).
        Range: -1 to 1. 1 is best.
        """
        if not self.texts or self.total_docs == 0:
            return 0.0
            
        score = 0.0
        top_words = topic_words[:10]
        n = 0
        
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                w_i = top_words[i]
                w_j = top_words[j]
                
                count_wi = self.word_counts.get(w_i, 0)
                count_wj = self.word_counts.get(w_j, 0)
                count_wi_wj = self.get_co_occurrence(w_i, w_j)
                
                if count_wi_wj == 0:
                    continue
                
                p_wi = count_wi / self.total_docs
                p_wj = count_wj / self.total_docs
                p_wi_wj = count_wi_wj / self.total_docs
                
                pmi = math.log( p_wi_wj / (p_wi * p_wj) )
                npmi = pmi / (-math.log(p_wi_wj))
                
                score += npmi
                n += 1
                
        return score / n if n > 0 else 0.0

    def calculate_cv(self, topic_words: List[str]) -> float:
        """
        Placeholder for Cv coherence. 
        Full Cv requires sliding windows and Boolean vectors, which is heavy to implement processing raw text.
        Often better to use gensim.models.CoherenceModel if available.
        """
        # TODO: Implement full sliding window Cv or integrate Gensim
        return 0.0
