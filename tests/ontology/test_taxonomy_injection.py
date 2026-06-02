"""Tests for ontology taxonomy injection through corpus_ingest pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphknows.ingestion.domain.taxonomy import (
    build_relation_schema,
    list_bundled_taxonomies,
    load_taxonomy_artifacts,
    load_taxonomy,
    resolve_taxonomy_path,
)


# ---------------------------------------------------------------------------
# domain.taxonomy — resolution helpers
# ---------------------------------------------------------------------------


class TestListBundledTaxonomies:
    def test_returns_at_least_two_entries(self) -> None:
        items = list_bundled_taxonomies()
        assert len(items) >= 2, f"Expected >= 2 bundled taxonomies, got {items}"

    def test_each_entry_has_name_and_path(self) -> None:
        for item in list_bundled_taxonomies():
            assert "name" in item
            assert "path" in item
            assert item["name"].startswith("taxonomy.")
            assert item["name"].endswith(".yaml")

    def test_generic_and_eu_politics_present(self) -> None:
        names = {t["name"] for t in list_bundled_taxonomies()}
        assert "taxonomy.generic.yaml" in names
        assert "taxonomy.eu-politics.yaml" in names


class TestResolveTaxonomyPath:
    def test_short_name_resolves_to_absolute_path(self) -> None:
        path = resolve_taxonomy_path("taxonomy.generic.yaml")
        assert path.endswith("taxonomy.generic.yaml")
        from pathlib import Path
        assert Path(path).is_absolute()
        assert Path(path).exists()

    def test_eu_politics_short_name_resolves(self) -> None:
        from pathlib import Path
        path = resolve_taxonomy_path("taxonomy.eu-politics.yaml")
        assert Path(path).exists()
        assert Path(path).is_absolute()

    def test_unknown_name_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_taxonomy_path("taxonomy.does-not-exist.yaml")

    def test_absolute_path_passthrough(self) -> None:
        import tempfile
        import textwrap
        from pathlib import Path

        content = textwrap.dedent("""\
            name: test
            entity_groups:
              PERSON:
                children: [PERSON]
                aliases: [person]
                description: A person.
                classification_tokens: []
                classification_patterns: []
                min_ner_confidence: 0.0
            relation_types: [KNOWS]
        """)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            result = resolve_taxonomy_path(tmp)
            assert result == tmp
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestTaxonomyArtifacts:
    def test_build_relation_schema_uses_relation_types(self) -> None:
        taxonomy = load_taxonomy(resolve_taxonomy_path("taxonomy.generic.yaml"))
        relation_schema = build_relation_schema(taxonomy)
        assert relation_schema["PART_OF"] == "part of"
        assert relation_schema["RELATED_TO"] == "related to"

    def test_load_taxonomy_artifacts_returns_path_and_relation_schema(self) -> None:
        artifacts = load_taxonomy_artifacts("taxonomy.eu-politics.yaml")
        assert artifacts.taxonomy_path is not None
        assert "OPPOSES" in artifacts.relation_schema
        assert "PERSON" in artifacts.entity_labels


# ---------------------------------------------------------------------------
# OntologyRelExtractor — injected relation schema wiring
# ---------------------------------------------------------------------------


class TestOntologyRelExtractorRelationSchema:
    def test_relation_schema_overrides_default_routing(self) -> None:
        from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor

        custom = {"SUPPORTS": "entity A supports entity B", "OPPOSES": "entity A opposes entity B"}
        extractor = OntologyRelExtractor(relation_schema=custom)
        resolved = extractor._resolve_ontology("some text about politics", None)
        assert resolved == custom

    def test_taxonomy_file_loads_same_relation_schema(self) -> None:
        from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor

        artifacts = load_taxonomy_artifacts("taxonomy.generic.yaml")
        extractor = OntologyRelExtractor(taxonomy_file="taxonomy.generic.yaml")
        resolved = extractor._resolve_ontology("some text", None)
        assert resolved == artifacts.relation_schema

    def test_explicit_schema_still_overrides_taxonomy(self) -> None:
        from graphknows.ingestion.ontology.rel_extractor import OntologyRelExtractor

        custom_taxonomy = {"SUPPORTS": "support relation"}
        extractor = OntologyRelExtractor(relation_schema=custom_taxonomy)
        explicit = {"causes": "cause relation"}
        resolved = extractor._resolve_ontology("text", explicit)
        assert resolved == explicit

    def test_no_taxonomy_falls_through_to_domain_routing(self) -> None:
        from graphknows.ingestion.ontology.rel_extractor import (
            DOMAIN_GK_ONTOLOGY,
            OntologyRelExtractor,
        )

        extractor = OntologyRelExtractor()
        # graphknows-flavoured text should route to DOMAIN_GK_ONTOLOGY
        resolved = extractor._resolve_ontology("memgraph neo4j pgvector stm ltm", None)
        assert resolved == DOMAIN_GK_ONTOLOGY


# ---------------------------------------------------------------------------
# CorpusIngestWorkflow — taxonomy_file wiring
# ---------------------------------------------------------------------------


class TestCorpusIngestWorkflowTaxonomyWiring:
    """Unit tests for taxonomy propagation — no live DB required."""

    def _make_workflow(self) -> "Any":  # type: ignore[name-defined]  # noqa: F821
        from graphknows.memory.workflows.corpus_ingest import CorpusIngestWorkflow

        return CorpusIngestWorkflow(
            mg_store=MagicMock(),
            pg_store=MagicMock(),
            neo_store=MagicMock(),
            extractor=MagicMock(),
            embed_fn=MagicMock(),
            rel_extractor=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_taxonomy_file_generic_builds_fresh_extractor(self) -> None:
        """When taxonomy_file is set, a fresh GLiNER2EntityExtractor is created."""
        from unittest.mock import AsyncMock

        workflow = self._make_workflow()

        created_extractors: list = []

        class FakeExtractor:
            def __init__(self, taxonomy_path: str | None = None) -> None:
                created_extractors.append(taxonomy_path)

        flush_result = MagicMock(nodes_written=0, edges_written=0, errors=[])
        mock_stm = MagicMock()
        mock_stm.ingest.return_value = MagicMock(entities=1, chunks=1)
        mock_flusher = MagicMock()
        mock_flusher.flush = AsyncMock(return_value=flush_result)

        with (
            patch("graphknows.memory.extractor.GLiNER2EntityExtractor", FakeExtractor),
            patch(
                "graphknows.ingestion.ontology.rel_extractor.OntologyRelExtractor",
                MagicMock(return_value=MagicMock()),
            ),
            patch("graphknows.memory.stm._async_bridge._run"),
            patch("graphknows.memory.stm.service.STMService", return_value=mock_stm),
            patch("graphknows.memory.flush.SessionFlusher", return_value=mock_flusher),
            patch("graphknows.config.builders.stores.build_pgvector_store", MagicMock()),
            patch("graphknows.mcp._state.get_settings", MagicMock()),
        ):
            from graphknows.memory.workflows.corpus_ingest import CorpusIngestRequest

            req = CorpusIngestRequest(
                documents=[
                    {"text": "The council supports the directive.", "title": "Test", "id": "d1"}
                ],
                taxonomy_file="taxonomy.generic.yaml",
            )
            result = await workflow.run(req)
            assert len(created_extractors) >= 1
            assert created_extractors[0] is not None
            assert "generic" in created_extractors[0]
            # The fresh extractor should be used — not the default one
            assert result.documents_processed == 1

    @pytest.mark.asyncio
    async def test_invalid_taxonomy_file_adds_error_and_uses_default(self) -> None:
        """A bad taxonomy_file adds an error but does not crash the workflow."""
        from unittest.mock import AsyncMock

        workflow = self._make_workflow()

        flush_result = MagicMock(nodes_written=0, edges_written=0, errors=[])
        mock_stm = MagicMock()
        mock_stm.ingest.return_value = MagicMock(entities=0, chunks=0)
        mock_flusher = MagicMock()
        mock_flusher.flush = AsyncMock(return_value=flush_result)

        with (
            patch("graphknows.memory.stm._async_bridge._run"),
            patch("graphknows.memory.stm.service.STMService", return_value=mock_stm),
            patch("graphknows.memory.flush.SessionFlusher", return_value=mock_flusher),
            patch("graphknows.config.builders.stores.build_pgvector_store", MagicMock()),
            patch("graphknows.mcp._state.get_settings", MagicMock()),
        ):
            from graphknows.memory.workflows.corpus_ingest import CorpusIngestRequest

            req = CorpusIngestRequest(
                documents=[{"text": "Some text.", "title": "T", "id": "d1"}],
                taxonomy_file="taxonomy.does-not-exist.yaml",
            )
            result = await workflow.run(req)
            assert any("taxonomy" in e.lower() or "not found" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# MCP list_taxonomies tool
# ---------------------------------------------------------------------------


class TestListTaxonomiesMCPTool:
    @pytest.mark.asyncio
    async def test_returns_bundled_taxonomies(self) -> None:
        from graphknows.mcp.tools.taxonomy import list_taxonomies

        result = await list_taxonomies()
        assert "bundled_taxonomies" in result
        assert len(result["bundled_taxonomies"]) >= 2

    @pytest.mark.asyncio
    async def test_each_bundled_entry_has_usage(self) -> None:
        from graphknows.mcp.tools.taxonomy import list_taxonomies

        result = await list_taxonomies()
        for entry in result["bundled_taxonomies"]:
            assert "name" in entry
            assert "usage" in entry
            assert "corpus_ingest" in entry["usage"]

    @pytest.mark.asyncio
    async def test_has_custom_taxonomy_instructions(self) -> None:
        from graphknows.mcp.tools.taxonomy import list_taxonomies

        result = await list_taxonomies()
        assert "custom_taxonomy" in result
        assert "usage" in result["custom_taxonomy"]
