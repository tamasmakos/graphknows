"""The RDF store writers must converge on a re-run, not accumulate.

The prototype these came from ``CREATE``d its edges, so importing the same
folder twice doubled every RDF_REL and BROADER. An RDF import is exactly the
operation someone re-runs — after a source file changes, after a parse fix — so
both edge writers MERGE on the full edge identity, and the DDL tolerates a
schema that already exists rather than failing the second run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.symbolic.ontology.rdf import store


def _client(command_side_effect=None) -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[], side_effect=command_side_effect)
    client.query = AsyncMock(return_value=[])
    client.create_database = AsyncMock()
    client.drop_database = AsyncMock()
    return client


def _commands(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


class TestWritersMerge:
    @pytest.mark.asyncio
    async def test_rdf_rel_edges_merge_on_predicate_and_provenance(self) -> None:
        client = _client()
        await store.write_projection(
            client,
            "mem_x",
            [{"iri": "http://ex/Dog"}],
            [
                {
                    "s": "http://ex/Dog",
                    "o": "http://ex/Animal",
                    "p": "rdfs:subClassOf",
                    "pi": "x",
                    "via": "direct",
                }
            ],
        )
        edge_cypher = _commands(client)[1]
        assert "MERGE (s)-[r:RDF_REL {predicate_iri: row.pi, via: row.via}]->(o)" in edge_cypher
        assert "CREATE (" not in edge_cypher

    @pytest.mark.asyncio
    async def test_broader_edges_merge_on_the_pair_alone(self) -> None:
        # via/bridged describe the edge, they do not identify it: merging on them
        # would leave a stale edge behind whenever a re-run reclassifies one.
        client = _client()
        await store.write_taxonomy(
            client,
            "mem_x",
            [{"key": "dog"}],
            [{"c": "dog", "p": "animal", "via": "class", "bridged": False}],
        )
        broader_cypher = _commands(client)[1]
        assert "MERGE (c)-[b:BROADER]->(p)" in broader_cypher
        assert "SET b.via = row.via, b.bridged = row.bridged" in broader_cypher
        assert "CREATE (" not in broader_cypher

    @pytest.mark.asyncio
    async def test_vertices_merge_on_their_unique_key(self) -> None:
        client = _client()
        await store.write_projection(client, "mem_x", [{"iri": "http://ex/Dog"}], [])
        await store.write_taxonomy(client, "mem_x", [{"key": "dog"}], [])
        node_cypher, taxon_cypher = _commands(client)
        assert node_cypher.startswith("UNWIND $rows AS row MERGE (n:ONODE {iri: row.iri})")
        assert "MERGE (t:ONTOLOGY_CLASS {key: row.key})" in taxon_cypher

    @pytest.mark.asyncio
    async def test_vertices_are_written_before_the_edges_that_match_them(self) -> None:
        client = _client()
        await store.write_projection(client, "mem_x", [{"iri": "http://ex/Dog"}], [{"s": "a"}])
        assert "MERGE (n:ONODE" in _commands(client)[0]
        assert "MATCH (s:ONODE" in _commands(client)[1]

    @pytest.mark.asyncio
    async def test_rows_are_written_in_batches(self) -> None:
        client = _client()
        rows = [{"iri": str(i)} for i in range(store.BATCH + 1)]
        await store.write_projection(client, "mem_x", rows, [])
        sizes = [len(call.kwargs["params"]["rows"]) for call in client.command.await_args_list]
        assert sizes == [store.BATCH, 1]

    @pytest.mark.asyncio
    async def test_no_command_is_issued_for_empty_rows(self) -> None:
        client = _client()
        await store.write_projection(client, "mem_x", [], [])
        assert _commands(client) == []


class TestSchema:
    @pytest.mark.asyncio
    async def test_ddl_is_idempotent_when_the_schema_already_exists(self) -> None:
        client = _client(command_side_effect=Exception("Type 'ONODE' already exists"))
        await store.ensure_schema(client, "mem_x")
        assert len(_commands(client)) == len(store.DDL)

    @pytest.mark.asyncio
    async def test_a_real_ddl_failure_is_not_swallowed(self) -> None:
        client = _client(command_side_effect=Exception("syntax error"))
        with pytest.raises(Exception, match="syntax error"):
            await store.ensure_schema(client, "mem_x")

    @pytest.mark.asyncio
    async def test_schema_creates_the_database_and_does_not_drop_it_by_default(self) -> None:
        client = _client()
        await store.ensure_schema(client, "mem_x")
        client.create_database.assert_awaited_once_with("mem_x")
        client.drop_database.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reset_drops_the_database_first(self) -> None:
        # A partial re-import leaves nodes from a source file that no longer
        # exists, so "one namespace, one corpus" needs the drop.
        client = _client()
        await store.ensure_schema(client, "mem_x", reset=True)
        client.drop_database.assert_awaited_once_with("mem_x")
        client.create_database.assert_awaited_once_with("mem_x")

    def test_ddl_declares_both_layers(self) -> None:
        ddl = "\n".join(store.DDL)
        for statement in (
            "CREATE VERTEX TYPE ONODE IF NOT EXISTS",
            "CREATE EDGE TYPE RDF_REL IF NOT EXISTS",
            "CREATE VERTEX TYPE ONTOLOGY_CLASS IF NOT EXISTS",
            "CREATE EDGE TYPE BROADER IF NOT EXISTS",
            # uri is the package's join key: write_instance_of and
            # write_class_matches both MATCH ONTOLOGY_CLASS on it.
            "CREATE PROPERTY ONTOLOGY_CLASS.uri IF NOT EXISTS STRING",
        ):
            assert statement in ddl

    def test_ddl_stays_out_of_the_core_schema(self) -> None:
        # CREATE INDEX ON ONTOLOGY_CLASS(key) UNIQUE against a pre-existing
        # namespace whose rows have no `key` is a migration hazard, so only this
        # tool applies it.
        from graphknows.storage.arcadedb._schema import _CORE_DDL

        core = "\n".join(stmt for _, stmt in _CORE_DDL)
        assert "ONODE" not in core
        assert "ONTOLOGY_CLASS(key)" not in core
