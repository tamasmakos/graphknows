export const meta = {
  name: 'ready-tasks',
  description: 'Spec Kit tasks.md with one runnable Verify per task; probe each on the tip (must FAIL); speckit-analyze report',
  phases: [
    { title: 'Tasks', detail: '/speckit-tasks plus the Verify rule' },
    { title: 'Probe', detail: 'every Verify must fail on the tip' },
    { title: 'Repair', detail: 'one round for tasks that broke the rule' },
    { title: 'Analyze', detail: '/speckit-analyze, read-only' },
  ],
}

// Human-driven before this: /speckit-specify, /speckit-clarify, /speckit-plan. Human-read after
// this: the analyze report. This script owns the one rule Spec Kit lacks -- a task is a runnable
// check that fails now -- and enforces it in code, not in a prompt.

const opt = args && typeof args === 'object' && !Array.isArray(args) ? args : {}
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
const FEATURE = opt.feature || ''

const HARNESS = ['.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/gate.sh', '.pre-commit-config.yaml']
const norm = (f) => String(f).replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase()
const hits = (files, list) => (files || []).map(norm).filter((f) => list.includes(f))
const skill = (path, why) => `Read ${MAIN_REPO}/.claude/skills/${path} and follow it: ${why}`

const FEATURE_HINT = FEATURE
  ? `The feature is ${FEATURE} (specs/${FEATURE}/).`
  : 'Resolve the feature directory with `powershell -ExecutionPolicy Bypass -File .specify/scripts/powershell/check-prerequisites.ps1 -Json -PathsOnly`.'

const VERIFY_RULE = [
  'ONE ADDITIONAL RULE, on top of the skill: every task line is followed by exactly one indented sub-bullet',
  '`  - Verify: `<command>``, holding ONE runnable shell command in backticks, on one line, no prose, no && chains.',
  'It must FAIL on the tree as it is now and PASS once the task is done; a task whose Verify already passes is not',
  'a task. If it names a test that does not exist yet, writing that test is part of the task. Python commands run',
  "in the workspace container, so write them as they would run at /app (e.g. `pytest -q tests/x/test_y.py`)",
  '-- the runner adds the docker exec. Sub-bullets are invisible to the other speckit skills, which is the point.',
].join(' ')

const IN_DOCKER = (dir) =>
  [
    'Run anything Python (pytest, ruff, mypy, python) inside the running workspace container, which mounts this',
    `checkout at /app: \`docker exec graphknows-workspace sh -c 'cd /app && <command>'\`. Never on the host.`,
    `First run \`docker exec graphknows-workspace test -f /app/${dir}/tasks.md\`; if that fails the container is`,
    'down or mounts another tree: report ran=false and stop. Do not start it, do not fall back to host python,',
    'and never report a result you did not observe.',
  ].join(' ')

const probe = (dir, cmd) =>
  [
    'Run one command and report what it did. Implement NOTHING, fix NOTHING, commit NOTHING -- you are measuring, not building.',
    `The command, verbatim: ${cmd}`,
    `Run it from the repository root (${MAIN_REPO}). ${IN_DOCKER(dir)}`,
    'Be exact about the difference between these two, because they mean opposite things:',
    '  - It RAN and FAILED -- an assertion failed, a symbol does not exist, an import errors because the module is not written, pytest exits 4 because the test file is not there. That is ran=true, passes=false, and it is the EXPECTED answer for a task with work left in it.',
    '  - It could not be run AT ALL -- malformed, prose rather than a command, docker unavailable. That is ran=false.',
    'Only a command that ran to a successful conclusion is passes=true. Do not adjust the command to make it run, and do not fix what it reveals.',
  ].join('\n')

const PROBE_SCHEMA = {
  type: 'object',
  required: ['ran', 'passes', 'excerpt'],
  additionalProperties: false,
  properties: {
    ran: { type: 'boolean' },
    passes: { type: 'boolean' },
    excerpt: { type: 'string', description: 'the last ~5 lines of output, or why it could not run' },
  },
}

const TASK_SCHEMA = {
  type: 'object',
  required: ['id', 'description', 'verify', 'files'],
  additionalProperties: false,
  properties: {
    id: { type: 'string', description: 'T001' },
    description: { type: 'string' },
    verify: { type: 'string', minLength: 1, description: 'the command from the Verify sub-bullet, without backticks' },
    files: { type: 'array', items: { type: 'string' }, description: 'every repository path the task line or its Verify names' },
  },
}

const TASKS_SCHEMA = {
  type: 'object',
  required: ['feature_dir', 'tasks'],
  additionalProperties: false,
  properties: {
    feature_dir: { type: 'string', description: 'e.g. specs/001-foo' },
    tasks: { type: 'array', items: TASK_SCHEMA, description: 'every OPEN ([ ]) task, read back from the file you wrote' },
  },
}

// ---- Tasks ----
phase('Tasks')
const written = await agent(
  [
    skill('speckit-tasks/SKILL.md', 'generate tasks.md for the current feature.'),
    FEATURE_HINT,
    VERIFY_RULE,
    'If tasks.md ALREADY EXISTS, do not regenerate it: keep every line, and add a Verify sub-bullet to each open task that lacks one.',
    'When done, re-read tasks.md and return every open task exactly as the file has it.',
  ].join('\n\n'),
  { label: 'tasks', phase: 'Tasks', schema: TASKS_SCHEMA },
)
if (!written?.tasks?.length) return { error: 'the tasks stage returned nothing -- see journal.jsonl' }
const DIR = written.feature_dir.replace(/\\/g, '/').replace(/\/$/, '')
log(`${DIR}: ${written.tasks.length} open task(s)`)

// A task that edits the harness cannot be verified by the harness. It is filed, never implemented here.
const gated = written.tasks
  .map((t) => ({ id: t.id, why: hits(t.files, HARNESS) }))
  .filter((g) => g.why.length)
  .map((g) => ({ id: g.id, why: `edits the harness (${g.why.join(', ')}) -- land it by hand` }))
const gatedIds = gated.map((g) => g.id)

// ---- Probe ----
phase('Probe')
const runProbes = (tasks) =>
  pipeline(tasks, (t) =>
    agent(probe(DIR, t.verify), { label: `probe:${t.id}`, phase: 'Probe', schema: PROBE_SCHEMA }).then((p) => ({ id: t.id, verify: t.verify, ...(p || {}) })),
  )
const verdict = (p) =>
  p.passes === true ? 'already satisfied -- its Verify passes on the tip before any work was done' : p.ran === false ? `not runnable -- ${p.excerpt || 'no detail'}` : ''
const probed = (await runProbes(written.tasks.filter((t) => !gatedIds.includes(t.id)))).filter(Boolean)
let rejected = probed.map((p) => ({ id: p.id, why: verdict(p) })).filter((r) => r.why)

// ---- Repair ----
if (rejected.length) {
  phase('Repair')
  const fixed = await agent(
    [
      `In ${MAIN_REPO}/${DIR}/tasks.md these tasks broke the Verify rule. ${VERIFY_RULE}`,
      ...rejected.map((r) => `- ${r.id}: ${r.why}`),
      'For each: if the work is genuinely already done, DELETE the task line and its sub-bullet. Otherwise REWRITE its',
      'Verify so it observes the missing work and can run. Touch nothing else in the file. Return the tasks you rewrote,',
      'read back from the file; omit the ones you deleted.',
    ].join('\n'),
    { label: 'repair', phase: 'Repair', schema: { ...TASKS_SCHEMA, required: ['tasks'] } },
  )
  const rewritten = (fixed?.tasks || []).filter((t) => rejected.some((r) => r.id === t.id))
  const deleted = rejected.filter((r) => !rewritten.some((t) => t.id === r.id)).map((r) => ({ id: r.id, why: `dropped: ${r.why}` }))
  const reprobed = (await runProbes(rewritten)).filter(Boolean)
  rejected = [...deleted, ...reprobed.map((p) => ({ id: p.id, why: verdict(p) })).filter((r) => r.why)]
  for (const r of rejected) log(`${r.id}: ${r.why}`)
}

const ready = written.tasks.map((t) => t.id).filter((id) => !gatedIds.includes(id) && !rejected.some((r) => r.id === id))

// ---- Analyze ----
phase('Analyze')
const analyze = await agent(
  [
    skill('speckit-analyze/SKILL.md', 'run the cross-artifact analysis for the current feature and return the full report as text.'),
    FEATURE_HINT,
    'Indented `- Verify:` sub-bullets under task lines are acceptance commands, not scope: do not flag the paths they name as unreferenced by the spec.',
    'Read-only. Edit nothing.',
  ].join('\n\n'),
  { label: 'analyze', phase: 'Analyze' },
)

return { feature_dir: DIR, ready, rejected, gated, analyze }
