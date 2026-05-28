---
title: Business Plan
type: concept
tags: [business, roadmap, planning]
created: 2026-05-03
updated: 2026-05-03
confidence: medium
---

# Business Plan

## Problem

Political intelligence is fragmented, expensive, and opaque. Existing tools aggregate news and monitor legislation, but none show HOW politicians are connected to each other and to ideas, or WHY specific positions persist and spread across debates over time. Analysts spend hours manually tracing influence and cross-referencing sources with no systematic reasoning trail.

## Solution

A graph-structured political intelligence platform that builds a living knowledge graph of political discourse -- entities, relationships, communities, and temporal patterns -- with a transparent reasoning audit trail. Users can query the graph conversationally and trace exactly which source segments justify each answer.

## Target Customers

- Think tanks (policy analysis, research briefs)
- EU affairs lobbying firms (position tracking, stakeholder mapping)
- Political consultancies (opposition research, campaign strategy)
- Investigative journalists (influence mapping, follow-the-money style graph analysis)
- Compliance teams at multinationals (regulatory monitoring, EU legislative risk)
- Academic researchers (computational political science, discourse analysis)

## Value Proposition

Know not just WHAT politicians say, but HOW they are connected and WHY it matters. GraphGen provides:
- Entity-level relationship graphs across speakers and debates
- Topic communities that show where ideas cluster and how they evolve
- Full source traceability: every insight links back to the original debate segment
- Reasoning audit trail: the path from question to answer is inspectable, not a black box

## Revenue Model

| Tier          | Price             | Target                                  |
|---------------|-------------------|-----------------------------------------|
| Researcher    | $99/month         | Academic researchers, independent analysts |
| Professional  | $299/month        | Policy analysts, journalists, consultants |
| Enterprise    | Custom            | Lobbying firms, multinationals, institutions |
| API Access    | Usage-based       | Developers, media platforms             |
| White-label   | Custom            | Data pipelines for third-party products |

## Competitive Landscape

| Competitor    | Strength                        | Gap vs GraphGen                          |
|---------------|---------------------------------|------------------------------------------|
| Dods          | UK/EU legislative monitoring    | No graph structure, no reasoning trail   |
| Politico Pro  | High-quality journalism + data  | Editorial, not analytical infrastructure |
| Capitol.ai    | AI-native policy intelligence   | US-focused, no KG                        |
| Quorum        | US legislative tracking         | US only, no discourse analysis           |
| FiscalNote    | Enterprise regulatory           | No graph reasoning, black-box AI         |

None of these competitors offer graph-structured reasoning transparency or entity relationship mapping across debate corpora.

## Moat

- Proprietary KG construction pipeline (GraphGen)
- Ontology-guided extraction methodology (hard to replicate without domain expertise)
- Multi-country parliamentary coverage as it scales
- Reasoning audit trail as a trust differentiator (especially for regulated industries)
- Network effects: more corpora ingested = richer graph = better answers

## Go-to-Market

Start in the EU affairs niche (Brussels bubble: lobbyists, think tanks, EP staff, EU institutions). This is a concentrated, high-value, underserved market with strong word-of-mouth dynamics. Expand to national parliaments and regulatory bodies in subsequent phases.

See [[gtm-strategy]] for phased rollout plan.

## Key Risks

- LLM extraction quality variance: precision depends on model quality; degradation is hard to detect without continuous evaluation
- Data freshness requirements: political intelligence has a short shelf life; ingest pipeline must be near-real-time
- Multi-language support complexity: EU alone has 24 official languages; non-English corpora require multilingual models
- Regulatory data access terms: parliamentary transcript licensing and scraping terms of service vary by country
- Commoditization: LLM capabilities are rapidly advancing; the moat must be in the data and the graph, not just the AI layer

## Funding Plan

- MVP (6 months): ~150,000 EUR -- solo/small team, cloud infrastructure, model costs
- Seed round target: 1-2M EUR -- hire core team, scale infrastructure, first enterprise contracts
- Use of funds: engineering team (ML + backend), EU affairs domain expert, sales/BD hire, infrastructure

## Team (Target)

- ML Engineer: KG pipeline, embedding, evaluation
- Backend Engineer: API, Neo4j/pgvector infrastructure, ingestion
- EU Affairs Domain Expert: ontology curation, customer validation, sales support
- Sales / Business Development: EU affairs network, enterprise outreach

---

Links: [[graphgen-project]] [[market-analysis]] [[gtm-strategy]]
