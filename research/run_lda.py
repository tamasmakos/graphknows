#!/usr/bin/env python3
"""Run a thesis-grade LDA baseline on the 'This is Europe' corpus."""

from __future__ import annotations

from pathlib import Path

from graphgen.evaluation.lda_baseline import evaluate_lda_baseline, load_segment_corpus, save_lda_results


def main() -> None:
    corpus_dir = Path("/root/graphgen/input/txt/translated")
    output_path = Path("/root/graphgen/thesis/lda_results.json")

    documents = load_segment_corpus(corpus_dir, min_words=8)
    summary = evaluate_lda_baseline(
        documents,
        topic_counts=[6, 8, 10, 12, 14, 18, 24, 36],
        vectorizer_kwargs={
            "min_df": 3,
            "max_df": 0.7,
            "max_features": 3000,
            "ngram_range": (1, 2),
        },
        lda_kwargs={
            "max_iter": 50,
            "learning_method": "batch",
        },
        test_size=0.2,
        random_state=42,
        top_words=15,
    )

    save_lda_results(summary, output_path)

    selected = summary["selected_model"]
    print("LDA baseline complete")
    print(f"Segments: {summary['corpus']['documents']}")
    print(f"Selected k: {selected['k']}")
    print(f"Test perplexity: {selected['test_perplexity']:.3f}")
    print(f"Mean UMass: {selected['mean_umass']:.3f}")
    print(f"Mean NPMI: {selected['mean_npmi']:.3f}")
    print(f"Topic diversity: {selected['topic_diversity']:.3f}")
    print(f"Mean dominant-topic probability: {selected['mean_dominant_topic_prob']:.3f}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
