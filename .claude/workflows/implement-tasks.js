export const meta = {
  name: 'implement-tasks',
  description: 'Agentless loop over tasks.md: probe (Verify must FAIL) -> N sampled patches -> validate/select -> commit; draft PR; merge only with {merge:true}, CI green, harness untouched. Not resumable mid-task: rerun instead',
  phases: [
    { title: 'Roster', detail: 'open tasks with a Verify; branch and tree must be clean' },
    { title: 'Probe', detail: 'per task, before any work' },
    { title: 'Sample', detail: 'N independent patches, each in its own detached worktree of HEAD' },
    { title: 'Validate', detail: 'apply, run Verify + regression, rank, commit the winner' },
    { title: 'Ship', detail: 'push, draft PR, optional CI watch and merge' },
  ],
}

const opt = args && typeof args === 'object' && !Array.isArray(args) ? args : {}
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
const FEATURE = opt.feature || ''
const ONLY = Array.isArray(opt.tasks) ? opt.tasks.map(String) : null
const SAMPLES = Number.isInteger(opt.samples) && opt.samples > 0 ? Math.min(opt.samples, 5) : 3
// Merging is the only step that changes main. Opt IN, in code: a forgotten argument must not write to main.
const DO_MERGE = opt.merge === true

const HARNESS = ['.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/gate.sh', '.pre-commit-config.yaml']
const DEPS = ['pyproject.toml', 'uv.lock']
const norm = (f) => String(f).replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase()
const hits = (files, list) => (files || []).map(norm).filter((f) => list.includes(f))

const FEATURE_HINT = FEATURE
  ? `The feature is ${FEATURE} (specs/${FEATURE}/).`
  : 'Resolve the feature directory with `powershell -ExecutionPolicy Bypass -File .specify/scripts/powershell/check-prerequisites.ps1 -Json -PathsOnly`.'

const HOUSE =
  'Repo rules (CLAUDE.md) are binding: pythonic OOP, type hints, one thing per function, small diffs, no speculative ' +
  'abstraction. NEVER weaken, delete or retarget an existing test to make a check pass; leave it failing and report it. ' +
  'A check that fails because the ENVIRONMENT is broken (missing corpus, model, service, credential) is not a code ' +
  'defect: library code under graphknows/ never downloads, provisions or reaches the network to make a check pass.'

const IN_DOCKER = (dir) =>
  [
    'Run anything Python (pytest, ruff, mypy, python) inside the running workspace container, which mounts this',
    `checkout at /app: \`docker exec graphknows-workspace sh -c 'cd /app && <command>'\`. Never on the host.`,
    `First run \`docker exec graphknows-workspace test -f /app/${dir}/tasks.md\`; if that fails the container is`,
    'down or mounts another tree: report ran=false and stop. Do not start it, do not fall back to host python,',
    'and never report a result you did not observe.',
  ].join(' ')

const PREPUSH =
  'Run the push in the FOREGROUND, in ONE Bash call, with an explicit timeout of 900000 ms: the pre-push hook runs ' +
  'the full container gate (~5 minutes) and the default timeout kills it mid-gate. A slow push is not an auth ' +
  'failure and not a hang: do not retry, do not re-authenticate, NEVER add --no-verify. If it is killed anyway, ' +
  'check `docker ps` for a `gk-gate-*` survivor and remove it before anything else.'

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
  properties: { ran: { type: 'boolean' }, passes: { type: 'boolean' }, excerpt: { type: 'string' } },
}

const ROSTER_SCHEMA = {
  type: 'object',
  required: ['feature_dir', 'branch', 'dirty', 'tasks', 'without_verify'],
  additionalProperties: false,
  properties: {
    feature_dir: { type: 'string', description: 'e.g. specs/001-foo' },
    branch: { type: 'string', description: 'output of git branch --show-current' },
    dirty: { type: 'boolean', description: 'true if git status --porcelain printed anything' },
    tasks: {
      type: 'array',
      description: 'every OPEN ([ ]) task that has a Verify sub-bullet, in file order',
      items: {
        type: 'object',
        required: ['id', 'line', 'verify'],
        additionalProperties: false,
        properties: {
          id: { type: 'string' },
          line: { type: 'string', description: 'the full task line, verbatim' },
          verify: { type: 'string', description: 'the command from the Verify sub-bullet, without backticks' },
        },
      },
    },
    without_verify: { type: 'array', items: { type: 'string' }, description: 'ids of open tasks that have no Verify sub-bullet' },
  },
}

const SAMPLE_SCHEMA = {
  type: 'object',
  required: ['patch', 'files', 'lines'],
  additionalProperties: false,
  properties: {
    patch: { type: 'string', description: 'absolute path of the .patch file you wrote; empty if you produced nothing' },
    files: { type: 'array', items: { type: 'string' }, description: 'git diff --cached --name-only, verbatim' },
    lines: { type: 'integer', description: 'insertions + deletions from git diff --cached --shortstat' },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object',
  required: ['winner', 'committed', 'sha', 'results'],
  additionalProperties: false,
  properties: {
    winner: { type: 'integer', description: 'index into the candidate list, or -1 when none passed its Verify' },
    committed: { type: 'boolean' },
    sha: { type: 'string' },
    results: {
      type: 'array',
      items: {
        type: 'object',
        required: ['index', 'applies', 'verify_passes', 'regression_passes', 'lint_ok', 'note'],
        additionalProperties: false,
        properties: {
          index: { type: 'integer' },
          applies: { type: 'boolean' },
          verify_passes: { type: 'boolean' },
          regression_passes: { type: 'boolean' },
          lint_ok: { type: 'boolean' },
          note: { type: 'string' },
        },
      },
    },
  },
}

// ---- Roster ----
phase('Roster')
const roster = await agent(
  [
    `In ${MAIN_REPO}: ${FEATURE_HINT}`,
    'Report `git branch --show-current` and whether `git status --porcelain` prints anything. Then read the feature\'s',
    'tasks.md and list every OPEN task (`- [ ]`) that is followed by an indented `- Verify:` sub-bullet, in file order.',
    'Edit nothing, run nothing else.',
  ].join('\n'),
  { label: 'roster', phase: 'Roster', schema: ROSTER_SCHEMA },
)
if (!roster) return { error: 'roster returned nothing -- see journal.jsonl' }
const DIR = roster.feature_dir.replace(/\\/g, '/').replace(/\/$/, '')
const featureNumber = DIR.split('/').pop().split('-')[0]
if (!roster.branch.startsWith(featureNumber)) {
  return { error: `checkout is on "${roster.branch}", not the ${DIR} feature branch -- \`git checkout ${DIR.split('/').pop()}\` first` }
}
if (roster.dirty) return { error: 'the tree is dirty -- commit or stash first; every sample forks HEAD and every candidate is applied on a clean tree' }
for (const id of roster.without_verify) log(`${id}: no Verify sub-bullet -- run /ready-tasks; skipped`)
const todo = roster.tasks.filter((t) => !ONLY || ONLY.includes(t.id))
if (ONLY) for (const id of ONLY) if (!todo.some((t) => t.id === id)) log(`${id}: not an open task with a Verify; skipped`)
log(`${todo.length} task(s) to implement on ${roster.branch}, ${SAMPLES} sample(s) each`)

const PATCH_DIR = `${MAIN_REPO}/docs/agents/runs/${DIR.split('/').pop()}`
const WT = `${MAIN_REPO}/.claude/worktrees` // gitignored; where the runtime keeps its own
const STANCES = [
  'the SMALLEST diff that makes the Verify pass',
  'the most THOROUGH change: handle the edge cases the task implies, not just the happy path',
  'a DIFFERENT approach from the obvious one -- a different seam, helper or data shape',
  'the change a maintainer would find easiest to read six months from now',
  'the change that reuses the most existing code in this repository',
]

const states = []
let commits = 0

for (const task of todo) {
  // ---- Probe: the precondition. A task whose Verify passes has already been done, by a sibling or by main.
  const before = await agent(probe(DIR, task.verify), { label: `probe:${task.id}`, phase: 'Probe', schema: PROBE_SCHEMA })
  if (!before || before.ran === false) {
    states.push({ id: task.id, state: 'needs-human', why: `its Verify could not run: ${before?.excerpt || 'no detail'}` })
    continue
  }
  if (before.passes === true) {
    const closed = await agent(
      `In ${MAIN_REPO}/${DIR}/tasks.md flip the task line \`${task.line}\` from \`- [ ]\` to \`- [X]\` and append " (already satisfied on ${roster.branch}: its Verify passed before any work)". Then \`git add -A && git commit -m "${task.id}: already satisfied"\`. Change nothing else.`,
      { label: `close:${task.id}`, phase: 'Probe', schema: { type: 'object', required: ['committed'], additionalProperties: false, properties: { committed: { type: 'boolean' } } } },
    )
    states.push({ id: task.id, state: 'already-satisfied' })
    if (!closed?.committed) { log(`${task.id}: could not commit the [X] flip -- stopping, the tree may be dirty`); break }
    continue
  }

  // ---- Sample: N independent attempts. No tests, no iteration -- selection happens in Validate.
  const raw = await parallel(
    Array.from({ length: SAMPLES }, (_, i) => () =>
      agent(
        [
          `You are attempt ${i + 1} of ${SAMPLES} at one task. Prefer ${STANCES[i % STANCES.length]}.`,
          // Hand-rolled on purpose: the runtime's isolation:'worktree' forks from main (measured 2026-09-02),
          // not from this branch, so a sample there would never see the commits earlier tasks made.
          `First, from ${MAIN_REPO}: \`git worktree add --detach "${WT}/${task.id}-${i + 1}" HEAD\`, and do ALL your work inside that directory.`,
          `Task: ${task.line}`,
          `Acceptance, which must pass when you are done and which the task's file path is a hint for: \`${task.verify}\`. If it names a test that does not exist, write that test.`,
          `Context: ${DIR}/spec.md and ${DIR}/plan.md in that worktree. ${HOUSE}`,
          'Do NOT run tests, linters or the Verify -- a separate step does that for every attempt. Edit, then:',
          `\`mkdir -p "${PATCH_DIR}" && git add -A && git diff --cached > "${PATCH_DIR}/${task.id}-${i + 1}.patch"\`, report \`git diff --cached --name-only\` and \`--shortstat\`,`,
          `then remove the worktree: from ${MAIN_REPO}, \`git worktree remove --force "${WT}/${task.id}-${i + 1}"\`.`,
        ].join('\n'),
        { label: `sample:${task.id}#${i + 1}`, phase: 'Sample', schema: SAMPLE_SCHEMA },
      ),
    ),
  )
  // Gate 7/8, in code, before anything touches the checkout: a patch that edits the harness cannot be
  // vouched for by the harness, and a new dependency is a supply-chain decision for a person.
  const candidates = raw.filter((s) => s && s.patch)
  const admitted = candidates.filter((s) => {
    const bad = [...hits(s.files, HARNESS), ...hits(s.files, DEPS)]
    if (bad.length) log(`${task.id}: dropped ${s.patch} -- it edits ${bad.join(', ')}`)
    return !bad.length
  })
  if (!admitted.length) {
    states.push({ id: task.id, state: 'needs-human', why: candidates.length ? 'every candidate edits the harness or a dependency file' : 'no sample produced a patch' })
    continue
  }

  // ---- Validate: tests decide. Apply each, run the Verify, then regression on the ones that pass; commit the winner.
  const verdict = await agent(
    [
      `In ${MAIN_REPO} (clean tree on ${roster.branch}), select and commit one of ${admitted.length} candidate patch(es) for task ${task.id}: ${task.line}`,
      ...admitted.map((s, i) => `  [${i}] ${s.patch} (${s.lines} lines: ${s.files.join(', ')})`),
      `For each candidate in order: \`git apply --check <patch>\` then \`git apply <patch>\`; run the Verify \`${task.verify}\`; ONLY where it passes also run`,
      '`pytest -q -m "not integration"`, `ruff check graphknows tests` and `mypy graphknows`. Between candidates: `git reset --hard && git clean -fd`.',
      IN_DOCKER(DIR),
      'Rank: Verify passes > regression passes > lint and types clean > fewer lines. Re-apply the winner. Flip its task line in',
      `${DIR}/tasks.md from \`- [ ]\` to \`- [X]\`. \`git add -A\`; if \`git diff --cached --name-only\` names any of ${[...HARNESS, ...DEPS].join(', ')}, STOP, reset, and report committed=false.`,
      `Otherwise \`git commit -m "${task.id}: <the task description>"\`. If a pre-commit hook rewrites files, \`git add -A\` and commit once more; never --no-verify.`,
      'If NO candidate passes its Verify: reset the tree, winner=-1, committed=false. Do not write code of your own.',
      HOUSE,
    ].join('\n'),
    { label: `validate:${task.id}`, phase: 'Validate', schema: VALIDATE_SCHEMA },
  )
  if (!verdict) { states.push({ id: task.id, state: 'failed', why: 'validate returned nothing' }); break }
  const win = verdict.results.find((r) => r.index === verdict.winner)
  if (verdict.winner < 0 || !win?.verify_passes) {
    states.push({ id: task.id, state: 'failed', why: verdict.results.map((r) => `[${r.index}] ${r.note}`).join('; ') })
    continue
  }
  if (!verdict.committed) {
    states.push({ id: task.id, state: 'failed', why: 'winner passed but was not committed -- the tree may be dirty' })
    break // a dirty tree poisons every later task: samples fork HEAD, candidates apply on a clean tree
  }
  commits++
  states.push({ id: task.id, state: 'done', sha: verdict.sha, files: admitted[verdict.winner].files, lines: admitted[verdict.winner].lines })
  log(`${task.id}: done (${verdict.sha}, ${admitted[verdict.winner].lines} lines)`)
}

// ---- Ship ----
if (!commits) return { branch: roster.branch, tasks: states, pr_url: null, merged: false }
phase('Ship')
const shipped = await agent(
  [
    `In ${MAIN_REPO} on ${roster.branch}: \`git push -u origin ${roster.branch}\`. ${PREPUSH}`,
    'Then open a DRAFT pull request against main (`gh pr create --draft --base main`), or reuse the open one for this branch. Body: a table of tasks with state, changed files and line counts, from this list:',
    ...states.map((s) => `  - ${s.id}: ${s.state}${s.why ? ` -- ${s.why}` : ''}${s.files ? ` -- ${s.files.join(', ')} (${s.lines} lines)` : ''}`),
    'Report the PR URL and `git diff --name-only main...HEAD`.',
  ].join('\n'),
  {
    label: 'ship',
    phase: 'Ship',
    schema: {
      type: 'object',
      required: ['pr_url', 'files'],
      additionalProperties: false,
      properties: { pr_url: { type: 'string' }, files: { type: 'array', items: { type: 'string' } } },
    },
  },
)
if (!shipped?.pr_url) return { branch: roster.branch, tasks: states, pr_url: null, merged: false, error: 'push or PR creation failed' }

let merged = false
if (DO_MERGE) {
  const harness = hits(shipped.files, HARNESS)
  if (harness.length) {
    log(`NOT merging -- the branch rewrites the harness (${harness.join(', ')}), so its own green checks prove nothing. A person lands it.`)
  } else {
    const ci = await agent(
      `Watch the CI run for ${shipped.pr_url} once (\`gh pr checks --watch\`). Report outcome "green", "red" or "unknown". Do NOT re-run, re-trigger or retry anything, and do not merge.`,
      { label: 'ci', phase: 'Ship', schema: { type: 'object', required: ['outcome'], additionalProperties: false, properties: { outcome: { type: 'string', enum: ['green', 'red', 'unknown'] } } } },
    )
    // Every condition for writing to main held in code: asked for, harness untouched, CI green on this tree.
    if (ci?.outcome === 'green') {
      const m = await agent(
        `\`gh pr ready ${shipped.pr_url}\` then \`gh pr merge ${shipped.pr_url} --squash\`. Report whether it merged.`,
        { label: 'merge', phase: 'Ship', schema: { type: 'object', required: ['merged'], additionalProperties: false, properties: { merged: { type: 'boolean' } } } },
      )
      merged = m?.merged === true
    } else {
      log(`NOT merging -- CI is ${ci?.outcome || 'unknown'}`)
    }
  }
}

return { branch: roster.branch, tasks: states, pr_url: shipped.pr_url, merged }
