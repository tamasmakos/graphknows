from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

TRANSCRIPT_PREFIX_RE = re.compile(r"^\[EN\]\s*.*?said from the .*? that\s*", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_segment_text(text: str) -> str:
    text = TRANSCRIPT_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_segment_corpus(input_dir: str | Path, min_words: int = 8) -> List[str]:
    root = Path(input_dir)
    segments: List[str] = []
    for path in sorted(root.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = clean_segment_text(line)
            if cleaned and len(cleaned.split()) >= min_words:
                segments.append(cleaned)
    return segments


def tokenize_document(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


class CoherenceScorer:
    def __init__(self, tokenized_docs: Sequence[Sequence[str]]):
        self.total_docs = len(tokenized_docs)
        self.word_counts: Counter[str] = Counter()
        self.co_occurrences: Counter[tuple[str, str]] = Counter()
        for doc in tokenized_docs:
            unique = sorted(set(doc))
            for word in unique:
                self.word_counts[word] += 1
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    self.co_occurrences[(unique[i], unique[j])] += 1

    def _co_occurrence(self, w1: str, w2: str) -> int:
        if w1 > w2:
            w1, w2 = w2, w1
        return self.co_occurrences.get((w1, w2), 0)

    def umass(self, topic_words: Sequence[str], top_n: int = 10) -> float:
        words = list(topic_words[:top_n])
        score = 0.0
        pairs = 0
        for i in range(1, len(words)):
            wi = words[i]
            for j in range(i):
                wj = words[j]
                count_wj = self.word_counts.get(wj, 0)
                if count_wj == 0:
                    continue
                score += math.log((self._co_occurrence(wi, wj) + 1.0) / count_wj)
                pairs += 1
        return score / pairs if pairs else float("-inf")

    def npmi(self, topic_words: Sequence[str], top_n: int = 10) -> float:
        if self.total_docs == 0:
            return -1.0
        words = list(topic_words[:top_n])
        score = 0.0
        pairs = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi, wj = words[i], words[j]
                co_count = self._co_occurrence(wi, wj)
                if co_count == 0:
                    continue
                p_wi = self.word_counts.get(wi, 0) / self.total_docs
                p_wj = self.word_counts.get(wj, 0) / self.total_docs
                p_wi_wj = co_count / self.total_docs
                if p_wi == 0 or p_wj == 0 or p_wi_wj == 0:
                    continue
                pmi = math.log(p_wi_wj / (p_wi * p_wj))
                score += pmi / (-math.log(p_wi_wj))
                pairs += 1
        return score / pairs if pairs else -1.0


def extract_topic_words(model: LatentDirichletAllocation, feature_names: Sequence[str], top_words: int = 15) -> List[List[str]]:
    topics: List[List[str]] = []
    for topic in model.components_:
        indices = topic.argsort()[:-top_words - 1:-1]
        topics.append([feature_names[i] for i in indices])
    return topics


def topic_diversity(topic_words: Sequence[Sequence[str]], top_n: int = 10) -> float:
    flattened = [word for topic in topic_words for word in topic[:top_n]]
    if not flattened:
        return 0.0
    return len(set(flattened)) / len(flattened)


def evaluate_single_lda_model(
    documents: Sequence[str],
    tokenized_documents: Sequence[Sequence[str]],
    k: int,
    *,
    vectorizer_kwargs: Dict[str, Any] | None = None,
    lda_kwargs: Dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    top_words: int = 15,
) -> Dict[str, Any]:
    vectorizer_options = {
        "stop_words": "english",
        "max_df": 0.7,
        "min_df": 3,
        "max_features": 3000,
        "ngram_range": (1, 2),
    }
    vectorizer_options.update(vectorizer_kwargs or {})

    lda_options = {
        "n_components": k,
        "max_iter": 50,
        "learning_method": "batch",
        "random_state": random_state,
    }
    lda_options.update(lda_kwargs or {})

    train_docs, test_docs = train_test_split(list(documents), test_size=test_size, random_state=random_state)
    vectorizer = CountVectorizer(**vectorizer_options)
    train_matrix = vectorizer.fit_transform(train_docs)
    test_matrix = vectorizer.transform(test_docs)

    lda = LatentDirichletAllocation(**lda_options)
    lda.fit(train_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = extract_topic_words(lda, feature_names, top_words=top_words)
    scorer = CoherenceScorer(tokenized_documents)
    umass_scores = [scorer.umass(topic) for topic in topics]
    npmi_scores = [scorer.npmi(topic) for topic in topics]
    doc_topic = lda.transform(train_matrix)

    return {
        "k": k,
        "train_documents": int(train_matrix.shape[0]),
        "test_documents": int(test_matrix.shape[0]),
        "vocabulary_size": int(len(feature_names)),
        "train_log_likelihood": float(lda.score(train_matrix)),
        "test_perplexity": float(lda.perplexity(test_matrix)),
        "mean_umass": float(sum(umass_scores) / len(umass_scores)),
        "mean_npmi": float(sum(npmi_scores) / len(npmi_scores)),
        "topic_diversity": float(topic_diversity(topics)),
        "mean_dominant_topic_prob": float(doc_topic.max(axis=1).mean()),
        "active_topics": int(len(set(doc_topic.argmax(axis=1)))),
        "topics": topics,
    }


def select_best_model(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No LDA results provided")

    def rank_positions(metric: str, reverse: bool) -> Dict[int, int]:
        ordered = sorted(results, key=lambda item: item[metric], reverse=reverse)
        return {item["k"]: idx + 1 for idx, item in enumerate(ordered)}

    npmi_ranks = rank_positions("mean_npmi", True)
    umass_ranks = rank_positions("mean_umass", True)
    diversity_ranks = rank_positions("topic_diversity", True)
    perplexity_ranks = rank_positions("test_perplexity", False)

    def score(item: Dict[str, Any]) -> tuple[float, float, float, float]:
        k = item["k"]
        aggregate_rank = npmi_ranks[k] + umass_ranks[k] + diversity_ranks[k] + perplexity_ranks[k]
        return (
            -aggregate_rank,
            item["mean_npmi"],
            item["topic_diversity"],
            -item["test_perplexity"],
        )

    return max(results, key=score)


def evaluate_lda_baseline(
    documents: Sequence[str],
    *,
    topic_counts: Sequence[int],
    vectorizer_kwargs: Dict[str, Any] | None = None,
    lda_kwargs: Dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    top_words: int = 15,
) -> Dict[str, Any]:
    tokenized_documents = [tokenize_document(doc) for doc in documents]
    tokenized_documents = [doc for doc in tokenized_documents if doc]
    if not documents:
        raise ValueError("Documents are required for LDA evaluation")
    model_results = [
        evaluate_single_lda_model(
            documents,
            tokenized_documents,
            k,
            vectorizer_kwargs=vectorizer_kwargs,
            lda_kwargs=lda_kwargs,
            test_size=test_size,
            random_state=random_state,
            top_words=top_words,
        )
        for k in topic_counts
    ]
    selected = select_best_model(model_results)
    return {
        "corpus": {
            "documents": len(documents),
            "tokenized_documents": len(tokenized_documents),
            "candidate_topic_counts": list(topic_counts),
        },
        "models": model_results,
        "selected_model": selected,
    }


def save_lda_results(payload: Dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
