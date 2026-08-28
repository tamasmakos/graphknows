# Local-files tracker

When the configured tracker is local files (no real issue tracker), publish by writing one file per ticket under `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order (blockers first). One ticket per file, never a single combined file. Each file's "Blocked by" lists the numbers/titles it depends on.

<local-ticket-template>

# <NN> — <Ticket title>

**What to build:** the end-to-end behaviour this ticket makes work, from the user's perspective — not a layer-by-layer implementation list.

**Blocked by:** the numbers/titles of the tickets that gate this one, or "None — can start immediately".

**Status:** the triage label the tracker config names

- [ ] Acceptance criterion 1
- [ ] Acceptance criterion 2

</local-ticket-template>
