"""MCP tools: taxonomy — list available taxonomies and configuration help."""

from __future__ import annotations

from graphknows.mcp._app import app


@app.tool()
async def list_taxonomies() -> dict:
    """List available taxonomy files for ontology-guided extraction.

    Taxonomies control which entity types GLiNER2 recognises and which
    relation types the DSPy relation extractor uses during corpus ingestion.

    Returns a list of available taxonomy names plus instructions on how to
    use them with ``corpus_ingest``.

    Usage example::

        # 1. Discover what is available
        await list_taxonomies()

        # 2. Pass the name to corpus_ingest
        await corpus_ingest(
            documents=[...],
            taxonomy_file="taxonomy.eu-politics.yaml",
        )

    You can also supply an absolute path to your own YAML file::

        await corpus_ingest(
            documents=[...],
            taxonomy_file="/data/my-custom-taxonomy.yaml",
        )
    """
    from graphknows.ingestion.domain.taxonomy import list_bundled_taxonomies

    bundled = list_bundled_taxonomies()

    return {
        "bundled_taxonomies": [
            {
                "name": t["name"],
                "path": t["path"],
                "usage": f'corpus_ingest(..., taxonomy_file="{t["name"]}")',
            }
            for t in bundled
        ],
        "custom_taxonomy": {
            "description": (
                "Supply any absolute path to a YAML file that follows the "
                "TaxonomyConfig schema (entity_groups + relation_types). "
                "See the bundled taxonomy files for format examples."
            ),
            "usage": 'corpus_ingest(..., taxonomy_file="/absolute/path/to/my.yaml")',
        },
        "schema_reference": {
            "entity_groups": (
                "Dict of entity labels → {children, aliases, description, "
                "classification_tokens, classification_patterns, min_ner_confidence}"
            ),
            "relation_types": "List of relation type strings (e.g. [SUPPORTS, OPPOSES, CAUSES])",
        },
    }
