"""Tests for graphknows.retrieval.recall_stm."""

from graphknows.retrieval.recall_stm import RecallResult, _entity_boost


class TestEntityBoostAddsNewChunks:
    def test_entity_only_chunk_is_added_to_results(self):
        """Chunks found via entity graph but NOT in ANN results must be added."""
        results: dict = {}  # empty — simulates no ANN hits

        class FakeMg:
            def query_entity_chunks(self, session_id, entity_names, top_k):
                import asyncio

                async def _coro():
                    return [
                        {
                            "chunk_id": "c1",
                            "text": "entity chunk",
                            "entity_names": ["obama"],
                        }
                    ]

                return _coro()

        _entity_boost(FakeMg(), "s1", ["obama"], top_k=5, results=results)

        assert "c1" in results, "Entity-matched chunk must be added even if not in ANN results"
        score, result = results["c1"]
        assert score > 0
        assert "obama" in result.entity_names
