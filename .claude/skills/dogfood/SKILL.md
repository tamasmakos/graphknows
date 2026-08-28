---
name: dogfood
description: Read the data your change actually produced before anyone reviews it — run the pipeline for real, read the records whole, and refine once if the output is ugly.
disable-model-invocation: true
---

# Dogfood

A green suite proves the kitchen did not burn down. It does not prove the food is edible.

You just changed code that produces data — nodes, edges, embeddings, chunks, retrieval hits, a JSON payload, a report. **Eat it.** Run the thing for real, put the output in front of you, and read it the way the user who receives it will. A test asserts one field of one record; you are here to look at the whole plate.

## 1. Cook

Run the changed path end to end on real input, with the smallest command that produces the real artifact.

Not a unit test — the actual pipeline: the CLI, the evaluation harness, a short script against a live store, whatever this repo already uses to make the thing. Use real input (a fixture conversation, a document from `examples/`, a handful of turns), never a mock that hands back what you were hoping to see.

**Done when:** an artifact exists that a user of this package would receive.

## 2. Plate

Get the output into a form you can read whole: dump the records, print the graph slice, pretty-print the JSON, write it to a scratch file.

Counts are not the plate. `wrote 1437 nodes` tells you nothing about whether any of them is any good.

**Done when:** you have at least 20 real records in front of you (or all of them, if there are fewer) — the actual field values, not a summary, a count, or a schema.

## 3. Taste

Read every record you plated. You are looking for output that passes every assertion and a human would still call garbage.

The smells:

- **Degenerate values** — empty strings, `None` where a value was promised, `0.0` scores, `"unknown"` labels, a default that quietly filled in for a failure.
- **Identical rows** — the same score, embedding, label or timestamp repeating across records that should differ. This is the classic: computed once at the wrong level and broadcast down.
- **Textual rubbish** — mojibake, stray escapes, markdown fences left inside extracted text, half-sentences, a prompt template leaking into stored content.
- **Degenerate distribution** — everything in one cluster, one label on 90% of rows, every similarity above 0.98, a long tail that is entirely noise.
- **Lost provenance** — a record you cannot trace back to the turn, chunk or document it came from.
- **Write-only volume** — thousands of records that nothing ever reads back. A channel that only writes is dead weight; say so plainly if you find one.
- **Wrong order, wrong time** — timestamps that do not match the source, ordering that is not stable across runs.

For each smell, quote the actual record as evidence. A smell without a record you can paste is a hunch, and hunches are not findings.

**Done when:** every plated record is accounted for as clean or smelly, and each smell carries a quoted record.

## 4. Refine

Decide which smells are **yours** — caused by this change — and which were already there.

Fix yours, in **one** pass. Rerun from step 1 and taste again to confirm the fix reached the data. If it is still ugly after that round, stop grinding: report what is still wrong and let a human read it. Two failed rounds means the problem is in the design, not the code.

Everything not caused by this change is a **discovery**: report it with its evidence — it becomes its own issue. Do not fix it here — a drive-by data fix makes the change unreviewable, and the smell has been there long enough to survive one more day.

Never reach for the test to make the taste better: weakening, retargeting or skipping an assertion turns bad data into invisible bad data.

**Done when:** every smell is fixed, filed as a discovery, or reported as still wrong — and the verdict is one of:

- **edible** — the data is what the change promised.
- **needs-work** — a human should read the output before this merges.
- **structurally-wrong** — the design produces bad data no local fix reaches. Stop and say so.
- **not-applicable** — the change produces no data anyone reads (a pure type fix, a docstring, build config). Say why and change nothing.
