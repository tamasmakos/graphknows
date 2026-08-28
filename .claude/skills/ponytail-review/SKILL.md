---
name: ponytail-review
description: >
  Code review focused exclusively on over-engineering. Finds what to delete:
  reinvented standard library, unneeded dependencies, speculative abstractions,
  dead flexibility. One line per finding: location, what to cut, what replaces
  it. Use when the user asks for an over-engineering review or invokes
  /ponytail-review. Complements correctness-focused review, this one only
  hunts complexity.
---

Review diffs for unnecessary complexity. One line per finding: location, what
to cut, what replaces it. The diff's best outcome is getting shorter.

## Format

`L<line>: <tag> <what>. <replacement>.`, or `<file>:L<line>: ...` for
multi-file diffs.

When the caller supplies an output schema, the schema wins: put `<tag>: <what>`
in the summary field, the replacement plus the maintenance cost of keeping the
code in the failure/consequence field. Severity, where the schema asks for one:
over-engineering defaults to the lowest tier; raise it only when the excess code
plausibly hides or causes defects; it is never a blocker on its own.

Tags:

- `delete:` dead code, unused flexibility, speculative feature. Replacement: nothing.
- `stdlib:` hand-rolled thing the standard library ships. Name the function.
- `native:` dependency or code doing what the platform already does. Name the feature.
- `yagni:` abstraction with one implementation, config nobody sets, layer with one caller.
- `shrink:` same logic, fewer lines. Show the shorter form.

## Examples

❌ "This EmailValidator class might be more complex than necessary, have you
considered whether all these validation rules are needed at this stage?"

✅ `L12-38: stdlib: 27-line validator class. "@" in email, 1 line, real validation is the confirmation mail.`

✅ `L4: native: moment.js imported for one format call. Intl.DateTimeFormat, 0 deps.`

✅ `repo.py:L88: yagni: AbstractRepository with one implementation. Inline it until a second one exists.`

✅ `L52-71: delete: retry wrapper around an idempotent local call. Nothing replaces it.`

✅ `L30-44: shrink: manual loop builds dict. dict(zip(keys, values)), 1 line.`

## Scoring

End with the only metric that matters: `net: -<N> lines possible.` (fold it
into the verdict field when a schema owns the output).

If there is nothing to cut, say `Lean already. Ship.` — and if the caller
assigned you other review axes too, report zero over-engineering findings and
carry on with those; this line ends the over-engineering pass, not the review.

## Boundaries

Scope: this skill covers only the over-engineering axis. It does not make
correctness bugs, security holes, or performance out of bounds for you — any
other axes the caller assigned still stand; this skill just isn't where they
come from. A single smoke test or `assert`-based self-check is the ponytail
minimum, not bloat, never flag it for deletion.
Does not apply the fixes, only lists them.
