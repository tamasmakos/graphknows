export const meta = {
  name: 'triage-issues',
  description: 'Sweep open GitHub issues: one screen per issue against the current tree, labels decided in code, then grouped into candidate specs for /speckit-specify. Applies labels only with {apply:true}; never applies ready-for-agent',
  phases: [
    { title: 'List', detail: 'open issues' },
    { title: 'Screen', detail: 'one agent per issue' },
    { title: 'Label', detail: 'decided in code; applied only with apply:true' },
    { title: 'Group', detail: 'candidate specs, single-task fixes, closures' },
  ],
}

const opt = args && typeof args === 'object' && !Array.isArray(args) ? args : {}
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
const LIMIT = Number.isInteger(opt.limit) && opt.limit > 0 ? Math.min(opt.limit, 100) : 40
// Writing to the tracker is opt IN, in code. ready-for-agent is never applied here: that label is
// what a human adds after reading this report, and a script that adds it has removed the human.
const APPLY = opt.apply === true

const HARNESS = ['.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/gate.sh', '.pre-commit-config.yaml']
const norm = (f) => String(f).replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase()
const hits = (files, list) => (files || []).map(norm).filter((f) => list.includes(f))
const LABELS = { triage: 'needs-triage', info: 'needs-info', agent: 'ready-for-agent', human: 'ready-for-human' } // docs/agents/triage-labels.md

// ---- List ----
phase('List')
const listed = await agent(
  `In ${MAIN_REPO}: \`gh issue list --state open --limit ${LIMIT} --json number,title,labels\`. Return the issues, newest first. Read nothing else, change nothing.`,
  {
    label: 'list',
    phase: 'List',
    schema: {
      type: 'object',
      required: ['issues'],
      additionalProperties: false,
      properties: {
        issues: {
          type: 'array',
          items: {
            type: 'object',
            required: ['number', 'title', 'labels'],
            additionalProperties: false,
            properties: { number: { type: 'integer' }, title: { type: 'string' }, labels: { type: 'array', items: { type: 'string' } } },
          },
        },
      },
    },
  },
)
const issues = listed?.issues || []
if (!issues.length) return { issues: [], groups: [], note: 'no open issues' }
log(`${issues.length} open issue(s)${issues.length === LIMIT ? ` -- capped at ${LIMIT}, pass {limit} for more` : ''}`)

// ---- Screen ----
phase('Screen')
const SCREEN_SCHEMA = {
  type: 'object',
  required: ['number', 'kind', 'still_valid', 'duplicate_of', 'size', 'files_named', 'verify', 'needs_info', 'summary'],
  additionalProperties: false,
  properties: {
    number: { type: 'integer' },
    kind: { type: 'string', enum: ['bug', 'enhancement', 'question', 'chore'] },
    still_valid: { type: 'boolean', description: 'false if the tree already does what the issue asks, or the code it names is gone' },
    duplicate_of: { type: 'integer', description: 'another open issue number this one duplicates, else 0' },
    size: { type: 'string', enum: ['small', 'medium', 'large'], description: 'small: one or two files, under ~50 lines' },
    files_named: { type: 'array', items: { type: 'string' }, description: 'every repository path the issue names or the fix would plainly touch' },
    verify: { type: 'string', description: 'the ONE runnable command from a "## Verify" section, verbatim; empty if absent or prose. Do not invent one' },
    needs_info: { type: 'string', description: 'what the reporter must add before anyone can act; empty if nothing' },
    summary: { type: 'string', description: 'one line: what it asks for, in terms of this codebase' },
  },
}
const screened = (
  await pipeline(issues, (issue) =>
    agent(
      [
        `Screen GitHub issue #${issue.number} ("${issue.title}") against the CURRENT tree of ${MAIN_REPO}. \`gh issue view ${issue.number} --json body,comments\`.`,
        'Check the code the issue names: does the file still exist, does the tree already do what it asks, is another open issue asking the same thing',
        `(the open list: ${issues.map((i) => `#${i.number} ${i.title}`).join('; ')}).`,
        'Read only. Change nothing, comment nowhere, run no tests. Report what you observed, not what the issue claims.',
      ].join('\n'),
      { label: `screen#${issue.number}`, phase: 'Screen', schema: SCREEN_SCHEMA },
    ).then((s) => (s ? { ...s, title: issue.title, labels: issue.labels } : null)),
  )
).filter(Boolean)
if (screened.length < issues.length) log(`${issues.length - screened.length} screen(s) returned nothing -- see journal.jsonl`)

// ---- Label: every decision here is code, from fields the screen reported. The agent never picks a label.
phase('Label')
const decide = (s) => {
  if (s.duplicate_of) return { label: null, why: `duplicate of #${s.duplicate_of} -- close by hand` }
  if (!s.still_valid) return { label: null, why: 'the tree already does this, or the code it names is gone -- close by hand' }
  if (s.needs_info) return { label: LABELS.info, why: s.needs_info }
  const harness = hits(s.files_named, HARNESS)
  if (harness.length) return { label: LABELS.human, why: `edits the harness (${harness.join(', ')}) -- no script may land it` }
  if (!s.verify) return { label: LABELS.triage, why: 'no runnable "## Verify" -- give it one, then a human may mark it ready-for-agent' }
  return { label: LABELS.triage, why: `has a Verify; recommend ${LABELS.agent} if a human agrees with the scope` }
}
const verdicts = screened.map((s) => ({ ...s, ...decide(s) }))
const toApply = verdicts.filter((v) => v.label && v.label !== LABELS.triage && !v.labels.includes(v.label))
if (APPLY && toApply.length) {
  await agent(
    [
      `In ${MAIN_REPO}, apply exactly these labels and no other. Create a label only if it does not exist (see docs/agents/triage-labels.md). Comment nowhere, close nothing.`,
      ...toApply.map((v) => `  gh issue edit ${v.number} --add-label "${v.label}"   # ${v.why}`),
    ].join('\n'),
    { label: 'apply', phase: 'Label', schema: { type: 'object', required: ['applied'], additionalProperties: false, properties: { applied: { type: 'array', items: { type: 'integer' } } } } },
  )
} else if (toApply.length) {
  log(`${toApply.length} label(s) would be applied -- pass {apply:true} to write them`)
}

// ---- Group ----
phase('Group')
const actionable = verdicts.filter((v) => v.label && v.label !== LABELS.info)
const grouped = await agent(
  [
    'Group these screened GitHub issues into work a person can start on. Each issue goes in exactly one group:',
    '  - "spec": several issues that are one feature -- the input to one /speckit-specify run;',
    '  - "single-task": one small issue with a Verify, fixable as one task without a spec;',
    '  - "human": must be done by a person (harness, or too vague to scope).',
    'Judge by what the issues ask of THIS codebase (the summaries below), not by their titles. Do not read the repository.',
    ...actionable.map((v) => `#${v.number} [${v.kind}, ${v.size}${v.verify ? ', has Verify' : ''}] ${v.summary} -- ${v.why}`),
  ].join('\n'),
  {
    label: 'group',
    phase: 'Group',
    schema: {
      type: 'object',
      required: ['groups'],
      additionalProperties: false,
      properties: {
        groups: {
          type: 'array',
          items: {
            type: 'object',
            required: ['kind', 'title', 'issues', 'specify'],
            additionalProperties: false,
            properties: {
              kind: { type: 'string', enum: ['spec', 'single-task', 'human'] },
              title: { type: 'string' },
              issues: { type: 'array', items: { type: 'integer' } },
              specify: { type: 'string', description: 'for kind=spec: the one-paragraph feature description to hand /speckit-specify; else empty' },
            },
          },
        },
      },
    },
  },
)

return {
  issues: verdicts.map((v) => ({ number: v.number, title: v.title, label: v.label, why: v.why, verify: v.verify })),
  close: verdicts.filter((v) => !v.label).map((v) => ({ number: v.number, why: v.why })),
  groups: grouped?.groups || [],
  applied: APPLY,
}
