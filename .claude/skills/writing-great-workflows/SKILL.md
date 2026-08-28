---
name: writing-great-workflows
description: Use when writing, editing or debugging a dynamic workflow script — the plain-JS scripts that orchestrate subagents via the Workflow tool, `ultracode`, or `.claude/workflows/`. Also use when deciding whether a task wants a workflow at all rather than a subagent, skill or agent team, even when the user never says the word workflow.
---

A workflow exists to buy **determinism** at scale: the script owns control flow, agents own the thinking, and intermediate results live in script variables instead of your context. Everything below serves that split.

Operational material — availability, triggering, permissions, the `/workflows` TUI, saving and distribution, settings, cost, the SDK type — is in [`REFERENCE.md`](REFERENCE.md). Read it when you need to run, watch, resume or ship a workflow rather than write one.

## Reach for a workflow when

The job needs **breadth** (sweep more than one context holds), **independent verification** (findings that must survive a skeptic), or **scale** (dozens of items, one shape) — and you want the orchestration itself version-controlled.

Anything else is a subagent or a plain turn. A single-agent task wrapped in a script is pure overhead, and so is any job that needs your judgment between stages: the script cannot stop and ask you. Split it into two workflows instead.

The shape that loses is work whose parts depend on each other — agents that would each need to see the others' intermediate state, or one change that has to stay coherent across every file it touches. Most coding is that shape. A refactor is one agent's job; an audit of a hundred files is a workflow's.

## Anatomy

```javascript
export const meta = {
  name: 'audit-routes',
  description: 'Audit route handlers for missing auth checks',
  phases: [
    { title: 'Discover', detail: 'list files' },
    { title: 'Audit',    detail: 'one agent per file' },
  ],
}

const found = await agent('List every .ts file under src/routes/.', {
  schema: { type: 'object', required: ['files'], additionalProperties: false,
            properties: { files: { type: 'array', items: { type: 'string' } } } },
})

const audits = await pipeline(found.files, file =>
  agent(`Audit ${file} for missing authentication checks.`, { label: file, phase: 'Audit' }),
)

return audits.filter(Boolean)
```

`meta` is a pure literal and the first statement — no variables, calls, spreads or interpolation, or the syntax check kills the run before it starts. The body is plain JS with top-level `await`. What you `return` is what reaches the session.

## Primitives

| Call | Behavior |
|---|---|
| `agent(prompt, opts?)` | One subagent. Returns its final text, or a validated object when `schema` is set. Returns `null` if stopped mid-run or killed by an API error |
| `pipeline(items, ...stages)` | Each item flows through every stage independently — no barrier. Stage signature `(prevResult, originalItem, index)`. A throwing stage drops that item to `null` |
| `parallel(thunks)` | Runs `() => …` thunks concurrently and **waits for all** — a barrier. A throwing thunk resolves to `null`; the batch never rejects |
| `workflow(ref, args?)` | Runs a saved workflow or `{ scriptPath }` inline, sharing this run's concurrency cap, agent cap, budget and journal. One level deep — nesting inside a child throws |
| `phase(title)` / `log(msg)` | Progress group; narrator line |

Globals: `args` (whatever JSON was passed at launch, `undefined` if none) and `budget` (token target; `budget.total` is `null` when none was set — guard every budget loop on it, or the loop runs to the 1,000-agent cap).

Always `.filter(Boolean)` what `pipeline()` and `parallel()` hand back.

### `agent()` options

`schema` forces a structured-output tool call validated at the tool layer, so the model retries its own mistakes; set `required` and `additionalProperties: false` to make those retries rare, and reach for it whenever a downstream stage consumes the result — free-text parsing is the top failure source. `label` and `phase` name the agent in the progress UI: set both on every agent, since inside `pipeline()`/`parallel()` the `phase` option is the only way to group without racing the global `phase()`. `model`, `effort` and `agentType` override per call — omit them to inherit the session and keep fan-out siblings identical, so they share one cached prompt prefix. `isolation: 'worktree'` is for parallel agents that write files that would otherwise collide, and costs disk and setup per agent.

### What goes in an agent prompt

Four things, and the one that fails is almost always a missing one: the **objective**, the **output format** (a `schema`, or the exact shape of text you expect back), the **boundaries** — what this agent owns and what a sibling owns — and whatever **tool or source guidance** it needs to pick well. A thin brief is the top cause of a fan-out going wrong: siblings duplicate each other, leave gaps between their territories, and drift off the task they were given. Write strategies rather than absolutes — *prefer primary sources, go breadth-first before depth* survives contact with a case the rule did not anticipate, where *always do X* does not.

## Barriers

**Default to `pipeline()`.** A barrier is real idle time: the fastest item waits on the slowest.

A `parallel()` barrier earns its place only when the next stage needs the whole prior set at once — dedupe or merge across every result, an early exit on the total, or a prompt that compares findings against each other. It does not earn its place because you want to flatten, map or filter first (do that inside a stage), because the stages feel conceptually separate (separate ≠ synchronized), or because it reads more cleanly.

Smell test: `parallel` → transform → `parallel`, where the transform has no cross-item dependency, was a pipeline.

Reduce in plain JS. Flattening, deduping, ranking and filtering are code, not an agent.

## Patterns

| Pattern | Shape |
|---|---|
| Fan out → reduce → synthesize | The base skeleton for audits, sweeps, research, reviews |
| Adversarial verify | Per finding, N skeptics each prompted to *refute* it; keep the majority survivors. One skeptic that can re-read the source or re-run the test is worth several that cannot |
| Perspective-diverse verify | Give each verifier a distinct lens — correctness, security, perf, does-it-reproduce — instead of N identical skeptics |
| Judge panel | N attempts from different angles → parallel judges each score in one call, 0–1 against a named rubric → synthesize from the winner, grafting the best runner-up bits |
| Loop-until-dry | Keep spawning finders until K consecutive rounds surface nothing new. Dedupe against everything **seen**, not just what was confirmed, or rejected items reappear every round and it never converges. Name the stopping condition inside the finder prompt too — an agent that does not know what done looks like keeps going |
| Fix-until-green | Run the checker → fix failures → repeat until it passes or two rounds make no progress |
| Isolated migration | One worktree agent per file, then verify each result |
| Completeness critic | A final agent asked what is missing — a modality not run, a claim unverified, a source unread |

Compose freely, and size the run to the question. A fact with one answer is one agent. A comparison across a handful of things is two to four. A sweep that has to be exhaustive is ten or more, each with its territory named. Fifty agents on a question one could have answered is the classic overshoot, and the cost is real — a fan-out burns on the order of fifteen times what the same question costs in conversation.

Spend a verification budget on evidence before you spend it on votes. Repeat skeptics agree with each other for reasons that have nothing to do with the claim — shared priors, position bias — so the third identical vote buys far less than the first verifier with a tool in its hand. And keep verification as the last line rather than the plan: it catches a bad finding, it cannot rescue a badly split fan-out.

Whenever a script bounds its own coverage — top-N, sampling, no retry — `log()` what was dropped. Silent truncation reads as "covered everything".

## What breaks a script

- `import()` anywhere fails the syntax check; there is no module loading.
- `Date.now()`, `Math.random()` and argless `new Date()` throw — they would invalidate the resume journal. Pass timestamps through `args`; vary siblings by index or prompt instead of RNG.
- The script itself has no filesystem, shell or network. Agents do every bit of I/O.
- Concurrency tops out at 16 (fewer on fewer CPUs), one `parallel`/`pipeline` call takes at most 4,096 items, and a run takes at most 1,000 agents. Larger fan-outs queue rather than fail.

Prefer many small agents to a few long ones: better resume economics, better cache sharing, finer progress. And slice first — one directory, one narrow question — read the token totals, then scale.
