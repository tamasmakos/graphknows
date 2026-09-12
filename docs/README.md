# GraphKnows documentation

GraphKnows is agentic memory for LLM applications: it reads a conversation or
document stream, builds a knowledge graph over ArcadeDB, and serves hybrid
retrieval (vector ANN + graph traversal + BM25, fused with RRF) back to an
agent.

New here? Start with the [project README](../README.md) — install, quickstart
and benchmarks. This folder is the reference behind it.

## Understand the system

| Page | Answers |
| --- | --- |
| [services.md](services.md) | What are the main pieces, what does each own, and how do they stack? |
| [architecture.md](architecture.md) | How does a chunk actually get written, consolidated and retrieved? |

## Use it

| Page | Answers |
| --- | --- |
| [modes.md](modes.md) | `llm_free` vs `llm_assisted` — what the preset really switches, and how to mix the knobs. |
| [configuration.md](configuration.md) | Every `GRAPHKNOWS_*` setting, and how to pick an LLM provider. |
| [integrations.md](integrations.md) | Wiring GraphKnows into an agent framework (LangGraph is the shipped reference) or into Claude Code's hooks. |

## Optional layers

| Page | Answers |
| --- | --- |
| [ontology.md](ontology.md) | Injecting an RDF/OWL ontology so the graph speaks your domain's vocabulary. |
| [packs.md](packs.md) | Domain packs: the shipped code and agent vocabularies, and what a pack must offer. |

## Operate it

| Page | Answers |
| --- | --- |
| [deployment.md](deployment.md) | Running the release stack with no network — what the build still needs, what falls outside the guarantee, and how to upgrade. |
| [multi-tenancy.md](multi-tenancy.md) | Namespaces as the isolation boundary, and how far they scale. |

## Project

| Page | Answers |
| --- | --- |
| [versioning.md](versioning.md) | What counts as public surface, and the deprecation window. |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Development setup, extension points, code style. |
| [../CHANGELOG.md](../CHANGELOG.md) | Release history. |

---

`agents/` is configuration consumed by Claude Code skills (issue-tracker
conventions, triage labels, domain-doc rules). It is not product documentation
and is not part of this index.
