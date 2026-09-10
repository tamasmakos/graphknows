---
name: graphknows-memory
description: When to record something into GraphKnows memory and when to retrieve from it. Use before answering from assumption about this project's decisions, people, systems or history, and after anything durable is decided or learned.
---

# Record versus retrieve

The hooks already do the routine half: a session start pulls `context`, a prompt pulls
`recall`, a tool call pulls `preview`, a stop or compaction records the transcript with
`observe` and `catchup`. What is left to you is the deliberate half.

## Retrieve

Retrieve — ask memory before you answer — when the question turns on something the
repository cannot tell you by itself:

- a past decision, its rationale, or who made it;
- who or what a name refers to (a person, a service, an environment, an acronym);
- what was already tried, and how it went;
- anything the user speaks of as known ("the migration", "the usual gate").

Do not retrieve what the working tree answers: file contents, symbol definitions, the
current test result. Read those instead — memory is for what is not in the code.

## Record

Record — `remember` — only what will still be true and still be wanted next week:

- a decision and the reason behind it, not the deliberation;
- a durable fact about a person, system or convention;
- a correction the user makes to something memory holds.

Do not record: transient state, generated code, anything already committed to the
repository, or your own summary of a conversation the transcript already carries.

When both apply, retrieve first: a record that contradicts memory is worth flagging to
the user rather than storing beside it.
