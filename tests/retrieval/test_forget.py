"""Tests for LTMForget — prune_disconnected, prune_by_ttl, forget_before, prune."""

from unittest.mock import MagicMock

import pytest


def _make_forget(deleted: int = 5):
    from graphknows.memory.forget import LTMForget

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    def _single_result(deleted_count):
        r = MagicMock()
        r.__getitem__ = lambda self, k: deleted_count if k == "deleted" else 0
        return r

    mock_session.run.return_value.single.return_value = _single_result(deleted)

    forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="postgresql://fake/db")
    return forget, mock_session


class TestPruneDisconnected:
    def test_returns_deleted_count(self):
        forget, _ = _make_forget(deleted=7)
        result = forget.prune_disconnected()
        assert result == {"deleted_entities": 7}

    def test_runs_correct_cypher(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.prune_disconnected()
        cypher = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" in cypher
        assert "Entity" in cypher or "ENTITY" in cypher

    def test_handles_none_result(self):
        from graphknows.memory.forget import LTMForget

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = None

        forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="pg://fake/db")
        result = forget.prune_disconnected()
        assert result == {"deleted_entities": 0}


class TestPruneByTtl:
    def test_returns_deleted_count(self):
        forget, _ = _make_forget(deleted=3)
        result = forget.prune_by_ttl(ttl_days=30)
        assert result == {"deleted_entities": 3}

    def test_passes_cutoff_and_floor_params(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.prune_by_ttl(ttl_days=14, pagerank_floor=0.005)
        kwargs = mock_session.run.call_args[1]
        assert "cutoff" in kwargs
        assert kwargs["floor"] == pytest.approx(0.005)


class TestForgetBefore:
    def test_returns_expected_keys(self):
        forget, mock_session = _make_forget(deleted=2)
        result = forget.forget_before("2024-01-01T00:00:00")
        assert "deleted_segments" in result
        assert "orphaned_entities_deleted" in result

    def test_runs_two_statements(self):
        forget, mock_session = _make_forget(deleted=0)
        forget.forget_before("2024-01-01")
        assert mock_session.run.call_count >= 2


class TestPrune:
    def _make_prune_forget(self, counts: list[int]):
        from graphknows.memory.forget import LTMForget

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        call_no = [0]

        def _single():
            idx = call_no[0]
            call_no[0] += 1
            if idx < len(counts):
                r = MagicMock()
                r.__getitem__ = lambda self, k: counts[idx] if k == "c" else 0
                return r
            return None

        mock_session.run.return_value.single = _single

        forget = LTMForget(neo4j_driver=mock_driver, pg_dsn="pg://fake/db")
        return forget

    def test_returns_all_keys(self):
        forget = self._make_prune_forget([0, 0, 0, 0])
        result = forget.prune()
        assert "documents_orphaned" in result
        assert "chunks_orphaned" in result
        assert "dialogs_empty" in result
        assert "utterances_ttl" in result

    def test_nonzero_counts_trigger_delete(self):
        forget = self._make_prune_forget([2, 1, 3, 0])
        result = forget.prune()
        assert result["documents_orphaned"] == 2
        assert result["chunks_orphaned"] == 1
        assert result["dialogs_empty"] == 3
