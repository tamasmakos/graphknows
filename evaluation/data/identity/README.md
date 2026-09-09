# Identity ground truth (FR-040)

`labelled.jsonl` is the hand-labelled identity sample scored by
[`evaluation/identity.py`](../../identity.py). One JSON object per line:

```json
{"referent": "graphknows.symbolic.index", "kind": "module",
 "mentions": ["graphknows.symbolic.index", "graphknows/symbolic/index.py", "symbolic/index.py"]}
```

A **cluster** is the set of mentions — names, paths, symbols — that refer to one real
referent. `referent` names it for a reader; only `mentions` is scored, since pairwise
precision/recall ignores cluster names.

## Corpus

This repository's own source tree and the Claude Code transcripts written against it
(`docs/`, `.claude/specs/`, `docs/agents/runs/`). Deliberately **not** LoCoMo: the panel
rows must not double as identity ground truth.

## Labelling procedure

1. Candidate referents enumerated from the tracked tree at commit `b214bc0`: every
   non-`__init__` module under `graphknows/` and `evaluation/`, every module-level class
   whose name occurs in exactly one module, and every document under `docs/`.
2. Mention forms per kind, as they actually appear in the transcripts and source:
   - module — dotted import path, repo-relative path, package-relative path, bare file name;
   - class — bare name, dotted path, `path.py::Name`;
   - document — repo-relative path and the `../` / `./` link forms used from README files.
3. Reviewed by hand, which is where the judgement calls live and the two rules that came
   out of them:
   - a bare name too generic to attribute to one referent in prose (`base.py`, `core.py`,
     `Memory`, `Decoder`, `Channel`, …) is dropped from the cluster, keeping the qualified
     forms — a labeller reading "the decoder" cannot say which referent is meant, so
     neither should the ground truth;
   - a name shared by two modules or two classes is not a referent at all and its
     candidate is dropped, so no mention belongs to two clusters.
4. Invariants checked by `tests/evaluation/test_identity_sample.py`: at least 200 clusters,
   at least two mentions each, and every mention in exactly one cluster.

Re-labelling after a rename is a hand edit of this file, not a regeneration: the point of
the sample is that a human decided each cluster.
