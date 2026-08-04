# Test-fixture ontologies

Small, fully-annotated ontologies used ONLY as test fixtures. They are not
shipped: the single bundled ontology is the CCO digest at
`graphknows/ontology/assets/cco/cco.json`.

They live here because the behaviours they exercise — SKOS definition parsing,
the domain/range conformance gate, injection quality, predicate mapping — are
ontology-agnostic, and asserting them against CCO's 1401 classes would be both
slow and brittle. `personal.ttl` is 11 classes with 100% definition coverage and
a hand-checked domain/range table, which is what a controlled fixture wants.

- `personal.ttl` — 11 classes / 25 properties, fully annotated.
- `personal-summary.jsonld` — its label-list digest, for the summary-parser path.
- `personal-profile.json` — the curated relation profile that CCO replaced;
  kept because the domain/range selection tests are written against its shape.
