# Definition of Done

Every ticket in this repo inherits this. It is stated **once, here**, and no ticket body
repeats it.

That is not tidiness. The implementer is a language model, and repeated boilerplate in the
document it must act on is measurably harmful: [Lost in the
Middle](https://arxiv.org/abs/2307.03172) shows retrieval degrades for material buried in a
long context, and Chroma's [Context Rot](https://www.trychroma.com/research/context-rot)
found a single distractor degrades performance across all 18 models tested. A rule restated
in every ticket is a distractor in every ticket.

A note on vocabulary: the [Scrum Guide
2020](https://scrumguides.org/scrum-guide.html) defines a Definition of Done and
deliberately **dropped** "Definition of Ready" in the 2020 revision. What makes a ticket
ready here is our own rule — see [triage-labels.md](triage-labels.md) — and is not Scrum.

## A change is done when

1. **The ticket's `## Verify` command passes**, producing the output the ticket said it
   would. This is the acceptance. A ticket with no runnable check is not done, it is
   *claimed* done.
2. **The full suite is green**, and no pre-existing failure was silenced to get there.
3. **No existing test was weakened, deleted, retargeted, or had an assertion loosened.**
   If a test genuinely encodes wrong behaviour, leave it failing and say so. This is the one
   unforgivable move.
4. **The diff stays inside the ticket's stated scope** — the files it named, plus their
   tests. Anything else is a separate ticket.
5. **Nothing was built that the ticket did not ask for.** No abstraction with one caller,
   no wrapper around a framework, no configuration for a value that never changes, no
   dependency a few lines of stdlib would cover. Ask in order: does it need to exist at
   all; is it already here; does the stdlib do it; can it be one line. Stop at the first
   answer that holds.
6. **Deletion earns no new tests.** When the work is removal, the existing suite still
   passing *is* the acceptance. A test asserting that a file is absent is the ticket's
   acceptance criteria copied into Python.
7. **Distribution rules hold** — every third-party import declared with a lower bound, no
   version bump, non-`.py` files declared to the build backend. This repo is a package
   installed by strangers, not an application.
8. **Documentation is true again** — but only where the change made it false. A change with
   no user-visible impact documents nothing.
9. **Pre-commit hooks pass unmodified.** Never `--no-verify`.

## What the change leaves behind

The minimum honest record, and no more. Each line below exists because something actually
consumes it; nothing here is present to satisfy a compliance requirement, because none of the
usual ones reach this project. SOC 2 scopes to a service organisation's system and a library
on PyPI is not one; the CRA's obligations attach to commercial activity; and the traceability
"shall" clauses everyone attributes to ISO/IEC/IEEE 29148 could not be verified from the
standard's own text, which is paywalled. The one public traceability mandate that does say
what it requires is [NASA NPR
7150.2](https://swehb.nasa.gov/display/7150/SWE-052+-+Bidirectional+Traceability+Between+Higher+Level+Requirements+and+Software+Requirements),
and it governs Class A–C flight software.

| Artifact | Why it exists |
|---|---|
| `Closes #N` **in the PR description** | The ticket→code edge. [GitHub's docs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue): a keyword in a *commit message* closes the issue but does **not** register the PR as linked. |
| A conventional-commit type prefix, and `BREAKING CHANGE:` where it applies | The only [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) rules that are normative, and the only ones SemVer consumes. |
| `Assisted-by: <agent>:<model>` on machine-authored commits | The [Linux kernel's policy](https://docs.kernel.org/process/coding-assistants.html); Fedora converged on the same token independently. |
| `Signed-off-by:` from a **human** | The [DCO](https://developercertificate.org/) is a legal certification only a person can make. The kernel is explicit that agents must not add this trailer, and it rejected `Co-authored-by` for models. |
| One CHANGELOG line per release, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)'s six categories | The user-visible difference, for a human reading a release. |
| An ADR — **only** when the decision is architecturally significant | See [docs/adr](../adr/). Structure, interfaces, dependencies, non-functional characteristics. Not per implementation choice. |

Not required, and deliberately not done: a traceability matrix, an issue reference in every
commit message, issue links in every changelog line, a design document per ticket.
