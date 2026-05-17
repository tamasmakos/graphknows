"""
Integration test: pgvector end-to-end retrieval with BAAI/bge-small-en-v1.5 embeddings.

Verifies:
1. PgVectorAdapter connects successfully.
2. The chunks table is populated (>= 1 row).
3. Cosine similarity search returns >= 3 results for a known query.
4. Each result has the expected fields and a valid score.
5. Embedding model name is consistent between graph_store and hybrid_agent.
"""

import sys
import os

# Add service paths so imports resolve regardless of install state
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "services", "graphgen"))
sys.path.insert(0, os.path.join(REPO, "services", "graphrag"))

PGVECTOR_DSN = "postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb"
QUERY = "EU security defence policy"
MIN_RESULTS = 3


def _get_embedding(text: str) -> list:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(text, normalize_embeddings=True).tolist()


def test_embed_model_constant_consistent():
    """EMBED_MODEL in hybridrag embedder must equal the expected model name."""
    from services.hybridrag.stores.embedder import EMBED_MODEL
    assert EMBED_MODEL == "BAAI/bge-small-en-v1.5", (
        f"Expected BAAI/bge-small-en-v1.5, got {EMBED_MODEL}"
    )


def test_pgvector_connection():
    """PgVectorAdapter.connect() must return True."""
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(PGVECTOR_DSN)
    assert pg.connect(), "PgVectorAdapter failed to connect"
    pg.close()


def test_chunks_table_populated():
    """chunks table must have at least one row."""
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(PGVECTOR_DSN)
    assert pg.connect()
    count = pg.chunk_count()
    pg.close()
    assert count >= 1, f"chunks table is empty (count={count})"


def test_search_chunks_returns_min_results():
    """search_chunks must return >= MIN_RESULTS for a known query."""
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(PGVECTOR_DSN)
    assert pg.connect()
    qemb = _get_embedding(QUERY)
    results = pg.search_chunks(qemb, k=MIN_RESULTS)
    pg.close()

    assert len(results) >= MIN_RESULTS, (
        f"Expected >= {MIN_RESULTS} results, got {len(results)}"
    )

    # Validate shape of each result
    for r in results:
        assert "chunk_id" in r, f"Missing 'chunk_id' in result: {r}"
        assert "text" in r,     f"Missing 'text' in result: {r}"
        assert "score" in r,    f"Missing 'score' in result: {r}"
        assert isinstance(r["score"], float), f"score is not float: {r['score']}"
        assert 0.0 <= r["score"] <= 1.0, f"score out of [0,1] range: {r['score']}"


def test_search_chunks_scores_are_ordered():
    """Results must be returned in descending cosine similarity order."""
    from utils.graphdb.pgvector_adapter import PgVectorAdapter
    pg = PgVectorAdapter(PGVECTOR_DSN)
    assert pg.connect()
    qemb = _get_embedding(QUERY)
    results = pg.search_chunks(qemb, k=5)
    pg.close()

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not in descending score order: {scores}"
    )


if __name__ == "__main__":
    # Run all tests manually
    tests = [
        test_embed_model_constant_consistent,
        test_pgvector_connection,
        test_chunks_table_populated,
        test_search_chunks_returns_min_results,
        test_search_chunks_scores_are_ordered,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)
