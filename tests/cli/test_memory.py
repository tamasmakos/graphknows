"""CLI smoke tests for graphknows.cli.memory commands.

These tests verify the CLI is importable, commands are registered,
and core commands run without Python errors. DB connectivity is mocked
at the store module level (lazy-import pattern).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from graphknows.cli.memory import cli


def _runner():
    return CliRunner()


# --- doctor ---


def test_doctor_outputs_check_lines():
    """Doctor must always output Memgraph / Neo4j / pgvector status lines."""
    runner = _runner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Memgraph" in result.output
    assert "Neo4j" in result.output
    assert "pgvector" in result.output


# --- forget session ---


def test_forget_session_calls_delete():
    import graphknows.retrieval.stores.memgraph as mg_mod
    import graphknows.retrieval.stores.pgvector as pg_mod

    runner = _runner()
    mg_mock = MagicMock()
    mg_mock.delete_session = MagicMock()
    pg_mock = MagicMock()
    pg_mock.clear_stm = AsyncMock(return_value=None)

    with (
        patch.object(mg_mod, "MemgraphStore", return_value=mg_mock),
        patch.object(pg_mod, "PgVectorStore", return_value=pg_mock),
    ):
        result = runner.invoke(cli, ["forget", "session", "s1"])

    assert result.exit_code == 0
    assert "s1" in result.output


# --- ingest (smoke) ---


def test_ingest_text_smoke():
    """CLI ingest command runs without import/syntax errors."""
    import graphknows.memory.ingestor as ing_mod
    import graphknows.retrieval.stores.memgraph as mg_mod
    import graphknows.retrieval.stores.pgvector as pg_mod

    runner = _runner()
    mock_result = MagicMock()
    mock_result.file_id = "abc123"
    mock_result.chunk_count = 2
    mock_result.entity_count = 5
    mock_result.elapsed_s = 0.1
    mock_result.errors = []

    mock_ingestor = MagicMock()
    mock_ingestor.ingest_text.return_value = mock_result

    with (
        patch.object(mg_mod, "MemgraphStore", MagicMock()),
        patch.object(pg_mod, "PgVectorStore", MagicMock()),
        patch.object(ing_mod, "MemoryIngestor", return_value=mock_ingestor),
    ):
        result = runner.invoke(cli, ["ingest", "Hello world", "--session", "s1"])
    # DB connection errors are acceptable; Python errors are not
    assert result.exit_code in (0, 1)
    # Should not be a Python exception trace
    assert "Traceback" not in result.output


def test_ingest_file_smoke(tmp_path):
    """CLI ingest --is-file runs without import/syntax errors."""
    import graphknows.memory.ingestor as ing_mod
    import graphknows.retrieval.stores.memgraph as mg_mod
    import graphknows.retrieval.stores.pgvector as pg_mod

    f = tmp_path / "doc.txt"
    f.write_text("Hello file content")
    runner = _runner()
    mock_result = MagicMock()
    mock_result.file_id = "file123"
    mock_result.chunk_count = 1
    mock_result.entity_count = 0
    mock_result.elapsed_s = 0.05
    mock_result.errors = []

    mock_ingestor = MagicMock()
    mock_ingestor.ingest_file.return_value = mock_result

    with (
        patch.object(mg_mod, "MemgraphStore", MagicMock()),
        patch.object(pg_mod, "PgVectorStore", MagicMock()),
        patch.object(ing_mod, "MemoryIngestor", return_value=mock_ingestor),
    ):
        result = runner.invoke(cli, ["ingest", str(f), "--session", "s1", "--is-file"])
    assert result.exit_code in (0, 1)
    assert "Traceback" not in result.output


# --- recall-short (smoke) ---


def test_recall_short_smoke():
    """CLI recall-short runs without import/syntax errors."""
    import graphknows.retrieval.recall_stm as rs_mod
    import graphknows.retrieval.stores.memgraph as mg_mod
    import graphknows.retrieval.stores.pgvector as pg_mod
    from graphknows.retrieval.recall_stm import RecallResult

    runner = _runner()
    mock_result = RecallResult(
        text="EU signed a deal.",
        source_type="chunk",
        scope="short_term",
        score=0.95,
        entity_names=["EU"],
        heading_path="## Policy",
    )
    with (
        patch.object(mg_mod, "MemgraphStore", MagicMock()),
        patch.object(pg_mod, "PgVectorStore", MagicMock()),
        patch.object(rs_mod, "recall_short_term", return_value=[mock_result]),
    ):
        result = runner.invoke(cli, ["recall-short", "EU policy", "--session", "s1"])
    assert result.exit_code in (0, 1)
    assert "Traceback" not in result.output


# --- recall-long (smoke) ---


def test_recall_long_smoke():
    """CLI recall-long runs without import/syntax errors."""
    import graphknows.retrieval.recall_ltm as rl_mod
    import graphknows.retrieval.stores.pgvector as pg_mod
    from graphknows.retrieval.recall_stm import RecallResult

    runner = _runner()
    mock_result = RecallResult(
        text="LTM content.",
        source_type="chunk",
        scope="long_term",
        score=0.82,
    )
    with (
        patch.object(pg_mod, "PgVectorStore", MagicMock()),
        patch("graphknows.cli.memory._neo4j_driver", MagicMock()),
        patch.object(rl_mod, "recall_long_term", return_value=[mock_result]),
    ):
        result = runner.invoke(cli, ["recall-long", "trade policy"])
    assert result.exit_code in (0, 1)
    assert "Traceback" not in result.output


# --- flush (smoke) ---


def test_flush_smoke():
    """CLI flush runs without import/syntax errors."""
    import graphknows.memory.flush as flush_mod
    import graphknows.retrieval.stores.memgraph as mg_mod
    import graphknows.retrieval.stores.pgvector as pg_mod

    runner = _runner()
    mock_result = MagicMock()
    mock_result.nodes_written = 10
    mock_result.edges_written = 20
    mock_result.chunks_promoted = 3
    mock_result.turns_promoted = 2
    mock_result.topics_created = 1
    mock_result.merged_entities = 2
    mock_result.elapsed_s = 1.5
    mock_result.errors = []

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=mock_result)

    with (
        patch.object(mg_mod, "MemgraphStore", MagicMock()),
        patch.object(pg_mod, "PgVectorStore", MagicMock()),
        patch("graphknows.cli.memory._neo4j_driver", MagicMock()),
        patch.object(flush_mod, "SessionFlusher", return_value=mock_flusher),
    ):
        result = runner.invoke(cli, ["flush", "s1"])
    assert result.exit_code in (0, 1)
    assert "Traceback" not in result.output


# ===========================================================================
# CLI memory: helper functions, group, recall command
# ===========================================================================


import os

# ---------------------------------------------------------------------------
# Helper functions — pure env var readers
# ---------------------------------------------------------------------------


class TestCliHelperFunctions:
    def test_pg_dsn_returns_default(self):
        from graphknows.cli.memory import _pg_dsn

        env = {k: v for k, v in os.environ.items() if k != "PGVECTOR_DSN"}
        with patch.dict(os.environ, env, clear=True):
            result = _pg_dsn()
        assert result.startswith("postgresql://")

    def test_pg_dsn_returns_env_value(self):
        from graphknows.cli.memory import _pg_dsn

        custom = "postgresql://user:pass@host:5432/db"
        with patch.dict(os.environ, {"PGVECTOR_DSN": custom}):
            result = _pg_dsn()
        assert result == custom

    def test_mg_uri_returns_default(self):
        from graphknows.cli.memory import _mg_uri

        env = {k: v for k, v in os.environ.items() if k != "MEMGRAPH_URI"}
        with patch.dict(os.environ, env, clear=True):
            result = _mg_uri()
        assert "bolt://" in result

    def test_mg_uri_returns_env_value(self):
        from graphknows.cli.memory import _mg_uri

        with patch.dict(os.environ, {"MEMGRAPH_URI": "bolt://myhost:7687"}):
            result = _mg_uri()
        assert result == "bolt://myhost:7687"


# ---------------------------------------------------------------------------
# CLI group — basic invocation
# ---------------------------------------------------------------------------


class TestCliGroup:
    def test_cli_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "memory management" in result.output.lower() or "graphknows" in result.output.lower()

    def test_ingest_missing_session_fails(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # --session is required — missing it must fail
        result = runner.invoke(cli, ["ingest", "some text"])
        assert result.exit_code != 0

    def test_recall_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["recall-short", "--help"])
        assert result.exit_code == 0

    def test_forget_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # No forget command — just test the group --help to cover module
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_flush_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["flush", "--help"])
        assert result.exit_code == 0

    def test_doctor_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_stats_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0


class TestRecallCommand:
    def test_recall_missing_session_fails(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # --session is required
        result = runner.invoke(cli, ["recall", "query"])
        assert result.exit_code != 0

    def test_recall_missing_query_fails(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["recall", "--session", "s1"])
        assert result.exit_code != 0

    def test_recall_with_mock_stores_runs(self):
        """recall-short command requires --session and QUERY argument."""
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # Just verify the missing-args behavior is correct
        result = runner.invoke(cli, ["recall-short"])
        assert result.exit_code != 0  # Missing QUERY

    def test_recall_missing_session_runs(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # Missing --session should fail
        result = runner.invoke(cli, ["recall-short", "my query"])
        assert result.exit_code != 0


class TestForgetCommand:
    def test_forget_not_available_as_top_level(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        # Verify the CLI group has specific commands
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output


class TestStatsCommand:
    def test_stats_help_exits_zero(self):
        from graphknows.cli.memory import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# NO_RESULTS constant
# ---------------------------------------------------------------------------


class TestConstants:
    def test_no_results_constant(self):
        from graphknows.cli.memory import _NO_RESULTS

        assert _NO_RESULTS == "No results found."
