---
title: Project Administration
type: concept
tags: [admin, pipeline, deployment]
confidence: high
---

# Project Administration

## Repository

- GitHub: tamasmakos/graphgen
- Local: ~/graphgen
- Package: graphgen/ (Python package with pyproject.toml)
- Tests: tests/ (gitignored - stage with: git add -f tests/)

## Branch/PR Workflow

- Use classic GitHub tokens (repo scope) - fine-grained PATs return 403
- Token stored in ~/.git-credentials
- PR workflow: feature branch -> push -> gh pr create -> CI -> merge
- Always merge latest main before final merge

## Config Management

- config.yaml: application logic, schema, extraction settings
- .env in ~/graphgen/: secrets (GROQ_API_KEY), infra overrides (NEO4J_HOST, NEO4J_PORT)
- Never commit .env to git
- config.full-improved.yaml and config.medium.yaml: alternative configs for different run sizes

## Key Commands

- graphgen: runs standard KnowledgePipeline
- python run_with_config.py: alternative entry point
- docker compose up: start Neo4j + pgvector containers
- pytest tests/ (or git add -f tests/ first if gitignored)

## Infrastructure

- Neo4j: bolt://172.18.0.1:7687, user=neo4j, pass=testpassword123
- pgvector: postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb
- Use 172.18.0.1 (not localhost) inside the hermes container
- docker-compose.yaml manages both services

## Environment Setup

- pip install -e . in ~/graphgen
- GROQ_API_KEY in ~/graphgen/.env
- EMBED_MODEL=BAAI/bge-small-en-v1.5 (constant in code)

## Links

[[graphgen-project]] [[project-roadmap]] [[technical-debt]]
