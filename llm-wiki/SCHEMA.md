# Wiki Schema

## Domain
Political Intelligence & Knowledge Graph Research — the GraphGen project and its surrounding ecosystem: NLP pipeline, graph construction, community detection, academic thesis, and business development.

## Conventions
- File names: lowercase, hyphens, no spaces
- Every wiki page starts with YAML frontmatter
- Use [[wikilinks]] to link between pages (minimum 2 outbound links per page)
- Bump updated date on every edit
- Every new page must be added to index.md
- Every action must be appended to log.md

## Frontmatter
title, created, updated, type (entity|concept|comparison|query|summary), tags, sources, confidence (high|medium|low)

## Tag Taxonomy
- Architecture: pipeline, architecture, graph, extraction, embedding, community
- Research: thesis, topic-modeling, nlp, evaluation, corpus
- Entities: project, person, organization, technology
- Infrastructure: neo4j, pgvector, docker, deployment
- Business: business, roadmap, planning, market
- Meta: comparison, admin

## Page Thresholds
- Create when entity/concept appears in 2+ sources OR is central to one source
- Split pages over 200 lines
