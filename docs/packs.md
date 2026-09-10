# Domain packs

A **domain pack** is domain knowledge as data: the concepts, predicates,
extraction labels and identity rules of one domain. Packs sit *above* the core —
`graphknows.packs` imports the core, the core never imports a pack — so adding a
domain adds no branch to the pipeline.

## The protocol

A pack is anything that satisfies `graphknows.packs.DomainPack` (a
`runtime_checkable` `Protocol`, so there is no base class to inherit):

| Member | What it supplies |
| --- | --- |
| `name` | The pack's name, reported when two packs claim the same key. |
| `concepts()` | The pack's concepts, lazily — `ConceptRef` per entry. |
| `predicates()` | The pack's predicates, lazily — `PredicateRef` per entry. |
| `entity_labels()` | The extraction label set. It **replaces**, it never unions. |
| `prompt_addendum()` | Domain guidance appended to the extraction prompt. |
| `hygiene()` | `str -> bool`: may this surface become an entity? |
| `thresholds()` | The `mention` and `fact` confidence floors. |
| `veto(a, b)` | True when the pack forbids merging these two entity ids. |

## Loading is all-or-nothing

`load_packs(packs)` merges every pack's symbols into one `LoadedPacks`
namespace — concepts keyed by uri, predicates keyed by id:

```python
from graphknows.packs import load_packs
from graphknows.packs.code import CodePack

loaded = load_packs([CodePack()])
```

If two packs claim one concept uri or predicate id, `load_packs` raises
`PackConflictError` naming the key and both packs, and returns nothing — so a
contested key leaves **no** pack loaded rather than half of them. Resolve the
clash in the pack's data, not by ordering the list.

## Shipped packs

| Pack | Domain |
| --- | --- |
| `graphknows.packs.code.CodePack` | A repository's source: symbols, files, imports, calls. Its `CodeExtractor` reads facts off the syntax tree — no model. |
| `graphknows.packs.agent.AgentPack` | The agent loop: sessions, prompts, decisions, files and tools. `AgentExtractor` reads a transcript deterministically. |

Both keep their vocabulary in `graphknows/packs/data/*.json`, so extending a
domain is an edit to data, not to code.
