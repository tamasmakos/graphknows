"""Migration 001 — LTM-8 schema: Document, Chunk, Dialog, Utterance."""

from __future__ import annotations

from graphknows.config import GraphKnowsSettings
from graphknows.schema import apply_schema


def run() -> None:
    settings = GraphKnowsSettings()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        with driver.session() as s:
            result = apply_schema(s)
            print(f"Applied {result['constraints']} constraints, {result['indexes']} indexes")
            if result["errors"]:
                print("Errors:", result["errors"])
    finally:
        driver.close()


if __name__ == "__main__":
    run()
