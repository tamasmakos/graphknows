---
name: tech-writing
description: Write and maintain this repo's user-facing documentation — which page owns which question, when NOT to write, and the house voice. Use when editing anything under docs/ or README.md, or when a change alters something a user of the package can observe.
---

# Technical writing

The job is to leave the documentation **true** again after a change — not to describe the change.

## First decide: can a user observe this?

A user-visible change is a new or changed public function, argument or return value; a changed default; a new or renamed `GRAPHKNOWS_*` setting; a different retrieval, ingestion or consolidation behaviour; a new dependency or a changed minimum Python; or a documented behaviour that was wrong and is now right.

If it is none of those — an internal refactor, a test, a type fix, a private helper — **write nothing** and say why. Documenting an invisible change adds noise every future reader has to filter. Not writing is the correct output more often than not.

## The documentation tree

`docs/README.md` is the index **and the nav**: themed tables (Understand the system / Use it / Optional layers / Operate it / Project), each row a page and the question it answers. A page that is not in a table there is invisible, whether or not the file exists.

**Not everything under `docs/` is yours.** This skill covers the pages a person installing the package reads. Three neighbours are off-limits and are deliberately absent from the table below:

| Not yours | Whose it is |
|---|---|
| `docs/agents/` | Rules written for agents — the Definition of Done, the label and severity tables. Owned by the workflows and skills that read them. |
| `docs/agents/runs/` | Machine output from previous runs. Gitignored: it is in the tree, not in the repository. Never edit it, never index it. |
| `docs/adr/` | Decision records. Immutable by design — a superseded ADR changes status and keeps its text. |

If one of them is genuinely wrong, say so and leave it. A writer told to "leave the documentation true again" over all of `docs/` will otherwise tidy last week's run report.

| Page | The question it answers |
|---|---|
| `README.md` | The shop window: install, quickstart, benchmarks. Also the wheel `long_description`, so it must render on PyPI. Keep it short; depth belongs in `docs/`. |
| `docs/README.md` | Which page answers which question. |
| `docs/services.md` | What the pieces are, what each owns, how they stack. |
| `docs/architecture.md` | How a chunk is written, consolidated and retrieved. |
| `docs/modes.md` | `llm_free` vs `llm_assisted`, and the knobs behind the preset. |
| `docs/configuration.md` | Every `GRAPHKNOWS_*` setting, and how to pick a provider. |
| `docs/integrations.md` | Wiring GraphKnows into an agent framework. |
| `docs/ontology.md` | Injecting an RDF/OWL ontology. |
| `docs/topics.md` | The re-runnable topic stage. |
| `docs/multi-tenancy.md` | Namespaces as the isolation boundary. |
| `docs/versioning.md` | What counts as public surface, and the deprecation window. |
| `CONTRIBUTING.md` | Development setup, extension points, code style. Contributor-facing, not user-facing. |

**Out of bounds:** `docs/agents/` is configuration consumed by Claude Code skills, not product documentation — it is gitignored and deliberately absent from the index. Never edit it, never link a product page to it. `CHANGELOG.md` belongs to whoever cuts the release.

Prefer editing an existing page to adding one. A new page is justified only by a genuinely new topic, and in the same commit it must be added as a row in the right table in `docs/README.md`, phrased as the question it answers — matching the voice of the rows already there.

## Make the smallest true edit

1. **Fix what is now wrong first.** Grep the docs for the symbol, the setting name, the old behaviour, and correct every page that now lies. A stale sentence is worse than a missing one — a reader trusts it.
2. **Then add what is missing**, on the page that already owns that topic.
3. **Show a runnable example only where the surrounding page already does**, and only one. Copy its style, its imports, its async idiom.
4. **Every code example, setting name, default value and signature must match the code you just read.** Never carry an example over from another page without re-reading the code behind it.
5. **Write for someone who has the package installed and does not have the repository.** Never point at a test file, an internal module path, or anything under `.claude/`.

## House voice

Plain declarative sentences, present tense, second person for instructions. State what the software does; do not sell it. No "simply", no "just", no "powerful", no "seamless", no marketing adjectives, no emoji. Match the spelling convention (British or American) of the page you are editing.

**Site-ready** means: plain CommonMark, no generator front matter and no shortcodes, one H1 per page, relative links between pages (`services.md` from inside `docs/`, `../CONTRIBUTING.md` to reach the root), and no link that only resolves on a local checkout.

## Report what you did not fix

Documentation that was **already wrong** before your change is not yours to fix here — a drive-by rewrite makes the diff unreviewable. Report it so it becomes its own issue. Same for any link you followed that does not resolve.
