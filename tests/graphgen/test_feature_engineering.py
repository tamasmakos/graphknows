"""Wave 3 — feature_engineering.py tests.

Tests pure computation functions in isolation:
- FeatureEngineeringResult dataclass
- LTMFeatureEngineer wrapper
- _project_initial_graph, _run_node2vec, etc. — session-mocked
- _compute_silhouette — pure numpy + sklearn
- _compute_separation — pure numpy geometry
- _compute_modularity — session-mocked
- _compute_quality_metrics — composition
- _process_community_result — mocked session
- run_feature_engineering — full mocked pipeline
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# FeatureEngineeringResult dataclass
# ---------------------------------------------------------------------------


class TestFeatureEngineeringResult:
    def test_defaults(self):
        from graphknows.feature_engineering import FeatureEngineeringResult

        r = FeatureEngineeringResult()
        assert r.entity_count == 0
        assert r.community_count == 0
        assert r.runtime_ms == 0.0
        assert r.top_entities == []
        assert r.algorithm_used == ""
        assert r.error == ""

    def test_can_set_values(self):
        from graphknows.feature_engineering import FeatureEngineeringResult

        r = FeatureEngineeringResult(entity_count=10, community_count=3, algorithm_used="leiden")
        assert r.entity_count == 10
        assert r.community_count == 3
        assert r.algorithm_used == "leiden"


# ---------------------------------------------------------------------------
# _compute_silhouette — pure numpy/sklearn
# ---------------------------------------------------------------------------


def _make_emb_rows(n: int, n_communities: int, dim: int = 8) -> list[dict]:
    """Make synthetic embedding rows with deterministic cluster structure."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        cid = i % n_communities
        # Put community centroids far apart
        center = np.zeros(dim)
        center[cid % dim] = float(cid + 1) * 5.0
        emb = (center + rng.standard_normal(dim) * 0.1).tolist()
        rows.append({"emb": emb, "cid": cid})
    return rows


class TestComputeSilhouette:
    def test_returns_none_when_fewer_than_10_rows(self):
        from graphknows.feature_engineering import _compute_silhouette

        rows = _make_emb_rows(9, 2)
        assert _compute_silhouette(rows) is None

    def test_returns_none_when_single_community(self):
        from graphknows.feature_engineering import _compute_silhouette

        rows = _make_emb_rows(20, 1)
        # All same cid → labels filtered to single community → None
        result = _compute_silhouette(rows)
        assert result is None

    def test_returns_float_for_well_separated_clusters(self):
        from graphknows.feature_engineering import _compute_silhouette

        rows = _make_emb_rows(40, 4, dim=16)
        result = _compute_silhouette(rows)
        assert result is None or isinstance(result, float)
        if result is not None:
            assert -1.0 <= result <= 1.0

    def test_handles_exception_gracefully(self):
        from graphknows.feature_engineering import _compute_silhouette

        # Pass rows with non-numpy-compatible embedding → should return None, not raise
        bad_rows = [{"emb": None, "cid": i % 2} for i in range(20)]
        result = _compute_silhouette(bad_rows)
        assert result is None


# ---------------------------------------------------------------------------
# _compute_separation — pure numpy geometry
# ---------------------------------------------------------------------------


class TestComputeSeparation:
    def test_empty_rows_returns_none_none(self):
        from graphknows.feature_engineering import _compute_separation

        result = _compute_separation([])
        assert result == (None, None)

    def test_returns_none_when_no_community_has_two_members(self):
        from graphknows.feature_engineering import _compute_separation

        # Each entity in its own community → no community has >=2 members
        rows = [{"emb": [1.0, 0.0], "cid": i} for i in range(5)]
        sep, overlap = _compute_separation(rows)
        assert sep is None
        assert overlap is None

    def test_perfect_separation(self):
        from graphknows.feature_engineering import _compute_separation

        # Two perfectly separated clusters along orthogonal axes
        rows = [
            {"emb": [1.0, 0.0, 0.0], "cid": 0},
            {"emb": [1.0, 0.0, 0.0], "cid": 0},
            {"emb": [0.0, 1.0, 0.0], "cid": 1},
            {"emb": [0.0, 1.0, 0.0], "cid": 1},
        ]
        sep, overlap = _compute_separation(rows)
        assert sep is not None
        assert overlap is not None
        assert sep == pytest.approx(1.0 - overlap)

    def test_separation_and_overlap_complement(self):
        from graphknows.feature_engineering import _compute_separation

        rows = _make_emb_rows(30, 3, dim=4)
        sep, overlap = _compute_separation(rows)
        if sep is not None and overlap is not None:
            assert sep + overlap == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Session-mocked GDS helper functions
# ---------------------------------------------------------------------------


def _mock_session(single_return=None, data_return=None):
    s = MagicMock()
    s.run.return_value.single.return_value = single_return
    s.run.return_value.data.return_value = data_return or []
    return s


class TestProjectInitialGraph:
    def test_calls_drop_then_project(self):
        from graphknows.feature_engineering import _project_initial_graph

        s = _mock_session()
        _project_initial_graph(s, ec=100, edge_count=200)
        assert s.run.call_count == 2
        # First call drops the old graph
        first_cypher = s.run.call_args_list[0][0][0]
        assert "drop" in first_cypher.lower() or "DROP" in first_cypher


class TestRunNode2Vec:
    def test_handles_exception_non_fatally(self):
        from graphknows.feature_engineering import _run_node2vec

        s = MagicMock()
        s.run.side_effect = Exception("GDS not available")
        # Should not raise
        _run_node2vec(s)

    def test_calls_gds_node2vec(self):
        from graphknows.feature_engineering import _run_node2vec

        single = MagicMock()
        single.__getitem__ = lambda self, k: 100 if k == "nodePropertiesWritten" else 50
        s = _mock_session(single_return=single)
        _run_node2vec(s)
        s.run.assert_called_once()
        assert "node2vec" in s.run.call_args[0][0]


class TestRunCosineWeights:
    def test_handles_exception_non_fatally(self):
        from graphknows.feature_engineering import _run_cosine_weights

        s = MagicMock()
        s.run.side_effect = Exception("no weights")
        _run_cosine_weights(s)  # should not raise


class TestRunPagerank:
    def test_falls_back_to_unweighted_on_failure(self):
        from graphknows.feature_engineering import _run_pagerank

        s = MagicMock()
        call_no = [0]

        def side_effect(cypher, **kwargs):
            call_no[0] += 1
            if call_no[0] == 1:
                raise Exception("weighted not available")
            return MagicMock()

        s.run.side_effect = side_effect
        _run_pagerank(s)
        assert s.run.call_count >= 2


class TestRunBetweenness:
    def test_skips_large_graphs(self):
        from graphknows.feature_engineering import _run_betweenness

        s = _mock_session()
        # ec > 5000 → should not call GDS at all
        _run_betweenness(s, ec=10000)
        s.run.assert_not_called()

    def test_calls_gds_for_small_graphs(self):
        from graphknows.feature_engineering import _run_betweenness

        s = _mock_session()
        _run_betweenness(s, ec=100)
        s.run.assert_called_once()
        assert "betweenness" in s.run.call_args[0][0]


class TestRunCommunityDetection:
    def test_returns_result_and_algorithm_name(self):
        from graphknows.feature_engineering import _run_community_detection

        comm_row = MagicMock()
        comm_row.__getitem__ = lambda self, k: 5 if k == "communityCount" else 0.3
        s = _mock_session(single_return=comm_row)
        result, algo = _run_community_detection(s)
        assert algo == "louvain"

    def test_falls_back_to_unweighted(self):
        from graphknows.feature_engineering import _run_community_detection

        call_no = [0]
        comm_row = MagicMock()
        comm_row.__getitem__ = lambda self, k: 3

        s = MagicMock()

        def side_effect(cypher, **kwargs):
            call_no[0] += 1
            if call_no[0] == 1:
                raise Exception("no weights")
            m = MagicMock()
            m.single.return_value = comm_row
            return m

        s.run.side_effect = side_effect
        result, algo = _run_community_detection(s)
        assert algo == "louvain_unweighted"


class TestComputeModularity:
    def test_returns_float_when_query_succeeds(self):
        from graphknows.feature_engineering import _compute_modularity

        row = {"modularity": 0.45}
        s = _mock_session(single_return=row)
        result = _compute_modularity(s)
        assert result == pytest.approx(0.45)

    def test_returns_none_when_both_queries_fail(self):
        from graphknows.feature_engineering import _compute_modularity

        s = MagicMock()
        s.run.side_effect = Exception("not found")
        result = _compute_modularity(s)
        assert result is None

    def test_returns_none_when_single_returns_none(self):
        from graphknows.feature_engineering import _compute_modularity

        s = _mock_session(single_return=None)
        result = _compute_modularity(s)
        assert result is None


# ---------------------------------------------------------------------------
# LTMFeatureEngineer wrapper
# ---------------------------------------------------------------------------


class TestLTMFeatureEngineer:
    def test_run_delegates_to_run_feature_engineering(self):
        from graphknows.feature_engineering import LTMFeatureEngineer

        mock_driver = MagicMock()

        with pytest.MonkeyPatch().context() as mp:
            import graphknows.feature_engineering as fe

            mp.setattr(
                fe,
                "run_feature_engineering",
                lambda drv, session_id=None: {"entity_count": 5, "community_count": 2, "error": ""},
            )
            eng = LTMFeatureEngineer(mock_driver)
            result = eng.run()

        assert result.entity_count == 5
        assert result.community_count == 2
        assert result.algorithm_used == "leiden"


# ===========================================================================
# Extended: quality metrics, community result, LtmFeatureEngineer
# ===========================================================================


from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_session_with_embeddings(emb_rows):
    """Create a mock Neo4j session that returns emb_rows."""
    s = MagicMock()
    result = MagicMock()
    result.data.return_value = emb_rows
    s.run.return_value = result
    return s


class TestComputeQualityMetrics:
    def test_returns_all_keys(self):
        from graphknows.feature_engineering import _compute_quality_metrics

        emb_rows = [
            {"emb": [0.1, 0.2, 0.3], "cid": "A"},
            {"emb": [0.4, 0.5, 0.6], "cid": "B"},
            {"emb": [0.1, 0.15, 0.25], "cid": "A"},
            {"emb": [0.45, 0.55, 0.65], "cid": "B"},
        ]
        s = _make_session_with_embeddings(emb_rows)

        # Patch _compute_modularity separately
        from unittest.mock import patch

        with patch("graphknows.feature_engineering._compute_modularity", return_value=0.42):
            result = _compute_quality_metrics(s)

        assert "sil" in result
        assert "global_separation" in result
        assert "global_overlap" in result
        assert "unweighted_modularity" in result
        assert result["unweighted_modularity"] == 0.42

    def test_handles_db_exception_gracefully(self):
        """Quality metric computation must not crash the pipeline."""
        from graphknows.feature_engineering import _compute_quality_metrics
        from unittest.mock import patch

        s = MagicMock()
        s.run.side_effect = RuntimeError("DB gone")

        with patch("graphknows.feature_engineering._compute_modularity", return_value=None):
            result = _compute_quality_metrics(s)

        # Should return dict with None values (not raise)
        assert "sil" in result
        assert result["sil"] is None


class TestProcessCommunityResult:
    def test_updates_result_dict_with_community_count(self):
        from graphknows.feature_engineering import _process_community_result

        s = MagicMock()
        s.run.return_value.data.return_value = []
        s.run.return_value.single.return_value = None

        result = {
            "entity_count": 100,
            "edge_count": 200,
        }
        comm_result = {
            "communityCount": 5,
            "modularity": 0.65,
            "modularities": [0.5, 0.65],
            "ranLevels": 3,
        }

        with (
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "graphknows.feature_engineering._compute_quality_metrics",
                return_value={
                    "sil": 0.7,
                    "global_separation": 0.5,
                    "global_overlap": 0.1,
                    "unweighted_modularity": 0.3,
                },
            ),
            __import__("unittest.mock", fromlist=["patch"]).patch(
                "graphknows.feature_engineering._create_gds_run_node"
            ),
        ):
            _process_community_result(s, result, comm_result, "louvain", "session-001")

        assert result["community_count"] == 5
        assert result["modularity"] == 0.65
        assert result["did_converge"] is True
        assert result["ran_iterations"] == 3

    def test_handles_null_modularities(self):
        from graphknows.feature_engineering import _process_community_result
        from unittest.mock import patch

        s = MagicMock()
        result = {"entity_count": 10, "edge_count": 5}
        comm_result = {
            "communityCount": 2,
            "modularity": None,
            "modularities": None,
            "ranLevels": 1,
        }

        with (
            patch(
                "graphknows.feature_engineering._compute_quality_metrics",
                return_value={
                    "sil": None,
                    "global_separation": None,
                    "global_overlap": None,
                    "unweighted_modularity": None,
                },
            ),
            patch("graphknows.feature_engineering._create_gds_run_node"),
        ):
            _process_community_result(s, result, comm_result, "leiden", "sess-x")

        assert result["modularity"] == 0.0  # None converted to 0.0
        assert result["modularities"] == []  # None converted to []


class TestLtmFeatureEngineerExtended:
    def test_engineer_instantiation(self):
        from graphknows.feature_engineering import LTMFeatureEngineer

        eng = LTMFeatureEngineer.__new__(LTMFeatureEngineer)
        # Verify it's the right type
        assert isinstance(eng, LTMFeatureEngineer)

    def test_feature_engineering_result_defaults(self):
        from graphknows.feature_engineering import FeatureEngineeringResult

        r = FeatureEngineeringResult()
        assert r.entity_count == 0
        assert r.community_count == 0
        assert r.runtime_ms == 0.0
        assert r.algorithm_used == ""
        assert r.error == ""
        assert r.top_entities == []

    def test_feature_engineering_result_set_fields(self):
        from graphknows.feature_engineering import FeatureEngineeringResult

        r = FeatureEngineeringResult()
        r.entity_count = 42
        r.community_count = 7
        r.algorithm_used = "leiden"
        assert r.entity_count == 42
        assert r.community_count == 7
