---
name: neo4j-info
description: Provides guidance on querying the project's Neo4j graph database, including schema conventions and Cypher patterns used in this codebase
---

## What I do

- Describe the graph schema (Entity, Document, Community nodes and their relationships)
- Suggest Cypher query patterns for common operations
- Remind you to use the `neo4j-query` tool to run live queries against the database

## Schema overview

- `Entity` nodes have: `name`, `type`, `community_id`, `embedding`
- `Document` nodes have: `source`, `text`, `chunk_id`
- Relationships: `(Entity)-[:COOCCURS]->(Entity)`, `(Document)-[:MENTIONS]->(Entity)`

## When to use me

Use this skill when writing or debugging Cypher queries, exploring the graph structure,
or building new ingestion/retrieval pipelines.

> Note: This skill contains instructions only. To run actual queries, use the `neo4j-query` custom tool.
