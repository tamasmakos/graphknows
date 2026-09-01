export const meta = {
  name: 'express-lane',
  description: 'Land several tiny tickets as one branch: a commit each, then one gate, one PR, one CI run, one merge',
  whenToUse:
    'For a handful of tickets that are each a few lines. The full per-issue pipeline spends about thirty-one minutes per ticket on the pre-push gate, the PR CI run and the post-merge main watch, and none of that shrinks with the diff -- so N tiny tickets cost N times thirty-one minutes for work that takes seconds. This lane shares all three across the batch: one worktree, one commit per ticket so history and issue-closing stay per-ticket, then a single gate, PR, CI watch and rebase-merge. run-merge-train calls it for the express cars it rosters; it can also be run directly on a list of small issues.',
  phases: [
    { title: 'Cut', detail: 'one worktree for the whole batch, from the current tip', model: 'sonnet' },
    { title: 'Probe', detail: "run each ticket's own \"## Verify\" against the tip; one already passing has nothing to commit", model: 'sonnet' },
    { title: 'Commit', detail: 'one agent per ticket, sequential, one commit each', model: 'sonnet' },
    { title: 'Ship', detail: 'one gate run, one PR listing every ticket', model: 'sonnet' },
    { title: 'Land', detail: 'watch CI once, rebase-merge so each ticket keeps its own commit', model: 'sonnet' },
  ],
}

// args: { issues: [201, 204, 207], merge?: false, base?: 'main', branch?: 'express/...',
//         maxLines?: 50, watchMain?: false, ship?: true, report?: true }
//
// WHY THIS EXISTS, since a second implementation path is a cost and needs to earn it:
// the expensive part of landing a change is not the thinking, it is the waiting. Measured
// on this repo: ~9 minutes of pre-push gate, ~11 of PR CI, ~11 of post-merge main watch.
// That ~31 minutes is per PR and does not care whether the diff is one line or four
// hundred. Five one-line tickets therefore cost about two and a half hours through the
// normal lane, and there is no stage to skip that changes it -- you cannot merge to main
// without CI, and a train cannot stop watching main because a red main is its halt
// condition. Sharing ONE of each across the batch is the only lever, so that is all this
// script does.
//
// WHAT IT DELIBERATELY GIVES UP: there is no per-ticket adversarial review, no dogfood, no
// wheel gate, no docs stage. One review reads the whole batch diff. That is only defensible
// while the changes really are tiny, which is why the caller's size rule is strict and why
// the commit stage below EJECTS a ticket that turns out bigger than advertised rather than
// quietly carrying on. A mis-sized ticket must fall back to the full lane, not receive a
// discount for having been mis-estimated.
const opt = args || {}
const ISSUES = Array.isArray(opt.issues) ? opt.issues.filter(Number.isInteger) : []
if (ISSUES.length < 1) {
  return { error: 'express-lane needs issue numbers. Pass args as an OBJECT: { issues: [201, 204] }.' }
}
if (ISSUES.length > 8) {
  return { error: `express-lane took ${ISSUES.length} issues; the cap is 8. Past that, one red CI run blocks too much work at once and bisecting the batch costs more than the lane saved.` }
}
const BASE = opt.base || 'main'
const DO_MERGE = opt.merge === true
const DO_SHIP = opt.ship !== false
const DO_REPORT = opt.report !== false
// The ejection threshold, and the far side of THE REVIEW BOUNDARY OF THE WHOLE SYSTEM:
// run-merge-train's EXPRESS_MAX_LINES admits a car at <=30 expected lines, this ejects one
// at >50 actual. Keep the pair in mind when moving either number -- the 20-line slack is
// deliberate (a wrong estimate should cost a restart, not a badly-reviewed merge) and it
// means a ticket rostered at 30 can legitimately land at 49 having had only batch-level
// review: no adversarial review, no dogfood, no wheel gate, no docs.
const MAX_LINES = opt.maxLines || 50
// Only a train needs this: post_merge_green is its halt condition. A standalone run has
// nothing that acts on the answer, so it does not spend eleven minutes measuring it.
const WATCH_MAIN = opt.watchMain === true
// Whether to run each ticket's own "## Verify" against the tip before committing anything.
// 'shadow' reports the verdict and commits anyway; 'on' closes a ticket already satisfied on
// main; 'off' skips it. On by default, like the caller's flag of the same name -- see the
// note there for why shadow was demoted (it proved #247 was already done, then paid for a
// full car anyway).
const PROBES = ['shadow', 'on', 'off'].includes(opt.probes) ? opt.probes : 'on'
const RUN = opt.runLabel || 'an express-lane run'

const GATE_LABEL = 'needs-triage'
const READY_LABEL = 'ready-for-agent'
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
const RUNS_DIR = 'docs/agents/runs'
const GH = `Use \`gh\`. If it is not on PATH, fall back to "${opt.ghPath || '/c/Program Files/GitHub CLI/gh.exe'}" (quote it).`

// Stopping the stack a car started is part of finishing, not an afterthought for a human.
// `docker compose` names its project after the worktree directory, so an issue's ArcadeDB
// comes up as gk-issue-<n>-arcadedb-1 and NOTHING stops it: the workflows never start it
// (a stage does, when it needs a live database) and so never brought it down. Measured
// 2026-08-30: five leaked containers from issues closed days earlier held ~14.7GB of a
// 31.2GB Docker allocation, and the pre-push gate died with `Killed` at 51% of the suite --
// on every branch, for every car, until they were stopped by hand. A car that cannot push
// still costs its full pipeline first, so this leak bills the whole batch.
//
// Only on success: a parked car keeps its stack AND its worktree, because that is the state
// a human needs to debug it.
const teardown = (worktree) =>
  `Then tear down what this car started, in this order: \`cd ${worktree} && docker compose down --remove-orphans\` (its ArcadeDB stack is named after that directory and nothing else ever stops it -- leaked stacks OOM-kill the pre-push gate for every branch, not just this one), then \`git worktree remove ${worktree}\` from the repo root. If either refuses, say so in notes and leave it -- do not force past a refusal you have not read.`

// Deterministic, so a resumed run reuses the same tree instead of stranding the first one.
// Hand-rolled rather than the runtime's `isolation: 'worktree'` on purpose: that one is
// per-agent and ephemeral, and this tree must be named, outlive any single agent, survive a
// killed session, and be written by N sequential Commit agents plus a Ship agent.
const BRANCH = opt.branch || `express/${ISSUES.join('-')}`
const WORKTREE = opt.worktree || `${MAIN_REPO.replace(/\/[^/]+$/, '')}/gk-express-${ISSUES[0]}`

const RUN_ON = { model: 'sonnet', effort: 'medium' }
const call = (prompt, opts = {}) => agent(prompt, { ...RUN_ON, ...opts })

const PREPUSH = [
  'Run it in the FOREGROUND, in ONE Bash call, with an explicit timeout of 900000 ms (15 minutes).',
  'The pre-push hook runs the full container gate -- ~5 minutes measured 2026-08-28, longer on a cold cache -- so the DEFAULT',
  '2-minute timeout kills it mid-gate every time. A slow push is NOT an auth failure and NOT a hang: do not retry',
  'with a different remote, do not re-authenticate, and NEVER add --no-verify. Run it once -- killing and retrying',
  'leaves an orphaned container-side pytest behind, and they starve each other of CPU.',
].join(' ')

// Same command process-issues uses, and for the same reasons: never on the host, never via
// scripts/gate.sh from a worktree (that script execs into the shared container, whose /app
// is the MAIN repo -- it would test the wrong code and report it as yours).
const CHECKS = [
  "Run the repo's checks INSIDE a container mounted on the worktree, from anywhere:",
  '```',
  `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${WORKTREE.replace(/\//g, '\\')}:/app" -w /app graphknows-workspace \\`,
  "  sh -c 'ruff check --no-fix graphknows evaluation tests \\",
  '   && ruff format --check graphknows evaluation tests \\',
  '   && mypy graphknows --strict --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores \\',
  '   && lint-imports \\',
  '   && python -m pytest -q -m "not integration"\'',
  '```',
  'Both env vars are load-bearing. If docker is unavailable, STOP and report it -- never fall back to host pytest,',
  'and never report a check you did not run.',
].join('\n')

const HOUSE = [
  'Repo rules (CLAUDE.md) are binding: pythonic OOP, type hints, one thing per function, small diffs,',
  'no speculative abstraction. NEVER weaken, delete or retarget an existing test to make the suite pass.',
  `The standing rules every ticket inherits are in ${MAIN_REPO}/docs/agents/definition-of-done.md -- read it once, follow it, and do not restate it anywhere.`,
].join(' ')

// -- Cut -----------------------------------------------------------
// One tree for the batch. Every ticket commits into it in turn, so two tickets touching the
// same file is a non-event here -- which is the one thing this lane does BETTER than the
// normal one, where the same overlap is a cross-car conflict.
phase('Cut')
const cut = await call(
  [
    `Prepare one worktree for a batch of ${ISSUES.length} small tickets in the GraphKnows repository at ${MAIN_REPO}. ${GH}`,
    `From the repo root: \`git fetch origin\`, then \`git worktree add ${WORKTREE} -b ${BRANCH} origin/${BASE}\`.`,
    `If ${WORKTREE} already exists from an earlier run, reuse it: cd there, \`git fetch origin\`, and confirm it is on ${BRANCH}. Report reused=true and DO NOT reset it -- an earlier run may have commits in it that this one is about to add to.`,
    `Then read each ticket: ${ISSUES.map((n) => `\`gh issue view ${n} --json number,title,state,labels,body\``).join(', ')}.`,
    `For each, report its number, title, state, labels, and the body VERBATIM -- the whole thing, not a summary. The next stage sees only what you report, and the body is the contract.`,
    'Report facts. Do not implement anything, do not commit, do not judge whether a ticket belongs here.',
  ].join('\n'),
  {
    label: 'express:cut',
    phase: 'Cut',
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['ready', 'tickets'],
      properties: {
        ready: { type: 'boolean', description: 'true once the worktree exists and is on the branch' },
        reused: { type: 'boolean' },
        base_sha: { type: 'string' },
        tickets: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['number', 'title', 'state', 'labels', 'body'],
            properties: {
              number: { type: 'integer' },
              title: { type: 'string' },
              state: { type: 'string' },
              labels: { type: 'array', items: { type: 'string' } },
              body: { type: 'string' },
            },
          },
        },
        notes: { type: 'string' },
      },
    },
  },
)

if (!cut?.ready) {
  return { error: `the worktree could not be prepared: ${cut?.notes || 'see the run journal'}`, issues: ISSUES, branch: BRANCH }
}

// The same admission rule the train applies, re-checked here because express-lane can also
// be run directly. A gated ticket must never ride a lane with less scrutiny than the normal
// one -- that would turn the express lane into a way around the gate.
const boarded = []
const refused = []
for (const t of cut.tickets || []) {
  const labels = t.labels || []
  const why =
    t.state !== 'OPEN'
      ? 'not open'
      : labels.includes(GATE_LABEL)
        ? `carries "${GATE_LABEL}" -- a person owes it a decision, and this lane reviews less, not more`
        : !labels.includes(READY_LABEL)
          ? `no "${READY_LABEL}" label`
          : ''
  if (why) refused.push({ number: t.number, title: t.title, reason: why })
  else boarded.push(t)
}
for (const r of refused) log(`#${r.number} refused from the express lane: ${r.reason}`)
if (!boarded.length) {
  return { error: 'no ticket was eligible for the express lane', issues: ISSUES, refused, branch: BRANCH }
}

// -- Probe ---------------------------------------------------------
// Every ticket carries a falsifiable acceptance test in "## Verify" and nothing has ever run
// it. A ticket already satisfied on the tip this batch was cut from has nothing to commit,
// and a Commit agent handed one either invents a diff to justify itself or reports "already
// done" as a note nobody counts. One agent for the whole batch, not one per ticket: they
// share this worktree and would share a container anyway.
//
// This is not the ejection net. Ejection catches a ticket BIGGER than rostered; this catches
// one with nothing left in it. An ejected ticket goes back to the full pipeline, a satisfied
// one is closed -- there is no work in it for the long road either.
const satisfied = []
const probeVerdicts = []
if (PROBES !== 'off' && boarded.length) {
  phase('Probe')
  const probe = await call(
    [
      `cd ${WORKTREE}. It is a clean checkout of the current tip of ${BASE} and nothing has been committed to it yet. ${GH}`,
      `Run each of these ${boarded.length} tickets' own acceptance check against that tree, and report one verdict each. Implement NOTHING, fix NOTHING, commit NOTHING -- you are measuring, not building.`,
      '',
      ...boarded.map((t) => [`TICKET #${t.number}: ${t.title}`, t.body, ''].join('\n')),
      `For each ticket: find its "## Verify" section, which holds one runnable command, and run exactly that command in this worktree. If it names the test suite, run it in a container mounted on the tree:`,
      '```',
      `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${WORKTREE.replace(/\//g, '\\')}:/app" -w /app graphknows-workspace \\`,
      `  sh -c '<the command from that ticket\'s "## Verify">'`,
      '```',
      'Both env vars are load-bearing. Never run the suite on the host, and never report a result you did not observe.',
      'Be exact about the difference between these two, because they mean opposite things:',
      '  - The command RAN and FAILED -- an assertion failed, a symbol does not exist yet, an import errors because the module is not written. That is ran=true, passes=false, and it is the NORMAL answer for a ticket with work left in it.',
      '  - The command could not be run AT ALL -- there is no "## Verify" section, it holds prose rather than a command, or docker is unavailable. That is ran=false. Do not invent a command for a ticket that does not carry one.',
      'Only a command that ran to a successful conclusion is passes=true, and that means an earlier merge already did this ticket\'s work.',
      PROBES === 'on'
        ? `Then, for each ticket with passes=true AND ONLY those: \`gh issue close <n> --comment "Its \\"## Verify\\" already passes on the tip this batch was cut from, before any work was done on it, so an earlier merge already satisfied it. Closed by ${RUN} rather than committing a no-op. Reopen if the check is simply too weak to see the missing work."\`. Close nothing else and comment nowhere else.`
        : 'Close nothing and comment nowhere. This run is only measuring what the probe would have decided.',
    ].join('\n'),
    {
      label: 'express:probe',
      phase: 'Probe',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['verdicts'],
        properties: {
          verdicts: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['number', 'ran', 'passes'],
              properties: {
                number: { type: 'integer' },
                ran: { type: 'boolean', description: 'true only if you executed the command and read its result. A command that runs and fails RAN.' },
                passes: { type: 'boolean', description: 'true only if the command completed successfully on this tree' },
                excerpt: { type: 'string', description: 'the last few lines of its output, or why it could not be run at all' },
                closed: { type: 'boolean' },
              },
            },
          },
          notes: { type: 'string' },
        },
      },
    },
  )
  for (const v of probe?.verdicts || []) probeVerdicts.push({ number: v.number, ran: v.ran === true, passes: v.passes === true })
  const passing = new Set(probeVerdicts.filter((v) => v.passes).map((v) => v.number))
  for (const v of probe?.verdicts || []) {
    if (v.passes !== true) continue
    log(`#${v.number}: its "## Verify" already passes on this tip -- ${PROBES === 'on' ? 'closed, nothing to commit' : 'WOULD close it (probes: shadow)'}`)
    if (PROBES === 'on') satisfied.push({ number: v.number, title: (boarded.find((t) => t.number === v.number) || {}).title || '', excerpt: v.excerpt || '' })
  }
  if (PROBES === 'on' && passing.size) {
    for (let i = boarded.length - 1; i >= 0; i--) if (passing.has(boarded[i].number)) boarded.splice(i, 1)
  }
  if (!boarded.length) {
    // Cut made a worktree and the probe may have left a stack in it. Nothing downstream will
    // reach the Land stage that normally cleans both up, so this path has to do it itself --
    // a tree nobody committed to is exactly the one nobody remembers to remove, and a leaked
    // stack OOM-kills the pre-push gate for every branch, not just this one.
    await call([`cd ${WORKTREE}. Nothing was committed here and nothing is going to be. ${teardown(WORKTREE)}`].join('\n'), {
      label: 'express:teardown',
      phase: 'Probe',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['removed'],
        properties: {
          removed: { type: 'boolean', description: 'true if both the compose stack and the worktree are gone' },
          notes: { type: 'string', description: 'what refused, if either did' },
        },
      },
    })
    return {
      lane: 'express',
      branch: BRANCH,
      worktree: WORKTREE,
      landed: [],
      satisfied,
      ejected: [],
      refused: refused.map((r) => ({ number: r.number, reason: r.reason })),
      probes: PROBES,
      probe_verdicts: probeVerdicts,
      next_step: `Every ticket in this batch already passed its own "## Verify" on the tip and was closed. Nothing was committed, and the worktree and its stack have been torn down.`,
    }
  }
}

// -- Commit --------------------------------------------------------
// Serial, on purpose: one worktree, and parallel writers to one tree collide. This is a
// loop, not a fan-out -- there is nothing here to parallelise and pretending otherwise
// would buy a conflict for no wall-clock.
phase('Commit')
const committed = []
const ejected = []
// Distinct from `satisfied`, which the Probe proved by running the ticket's own "## Verify".
// This one is only a Commit agent's unverified claim, so it is reported, never acted on.
const claimedSatisfied = []
for (const t of boarded) {
  const out = await call(
    [
      `cd ${WORKTREE} and work only there. This is a SHARED worktree: ${boarded.length} small tickets are landing on one branch, one commit each. Earlier commits from this batch may already be present -- leave them alone.`,
      `YOUR TICKET -- #${t.number}: ${t.title}`,
      '',
      t.body,
      '',
      `This ticket was rostered as a few lines of work. Read its "## Verify" section: that command, passing, is the acceptance. Read ${MAIN_REPO}/docs/agents/definition-of-done.md once for the standing rules.`,
      // The whole justification for this lane's reduced review is that the changes are
      // genuinely small. An agent that discovers otherwise and carries on has removed
      // that justification without telling anyone, so it is told to stop instead.
      `STOP AND EJECT if any of these turns out to be true, and commit NOTHING when it does: the change needs more than about ${MAX_LINES} changed lines; it needs a file the ticket does not name; it needs a new dependency; it touches \`.github/workflows/\`, \`scripts/gate.sh\` or \`.pre-commit-config.yaml\`; or the ticket turns out to be ambiguous about what to build.`,
      'Ejecting is the right call for an oversized or ambiguous ticket, but it is NOT free: the ticket goes back to the full pipeline, which is 15 agent stages. Measured 2026-08-31 on #278, a four-line Dockerfile change: 11 agents, 54 minutes, ~1.6M tokens. Eject when the ticket really is too big for this lane, and say so precisely -- but do not eject to avoid a judgement call, and never eject a ticket whose work is simply already done (see below). Carrying on regardless is the other failure: this lane reviews the whole batch once, so an oversized change here gets a fraction of the scrutiny it would get on its own.',
      '',
      // An ejected ticket is re-queued as a full car; a satisfied one is closed. Reporting
      // "already done" as an ejection buys the most expensive outcome for the cheapest
      // finding, so the two need separate channels -- see the Probe note above.
      `If you find this ticket's work is ALREADY DONE -- an earlier commit in this batch, or an earlier merge, already did it -- set satisfied=true with the evidence in eject_reason (the commit sha or the code that already does it) and commit NOTHING. Do NOT set ejected=true for this: an ejected ticket is re-run through the whole 15-stage pipeline, and re-doing work that is already finished is the single most wasteful thing this lane can do. A satisfied ticket is just closed.`,
      '',
      'If it IS small, implement it fully, then:',
      `  1. Run the ticket's "## Verify" command. ${CHECKS}`,
      '     Run the full check chain, not just the ticket\'s own command -- your commit shares a branch with the others, and a lint failure you introduce blocks all of them.',
      '  2. `git add -A` and commit JUST your change. Check `git status --short` first for `dist/`, `build/` or `*.egg-info/` and delete any that appear.',
      `  3. Commit message: a concise lower-case imperative subject, then a blank line, then exactly \`Closes #${t.number}\`. That trailer is what closes the ticket when the branch lands, and it is per-commit because this branch rebase-merges -- each ticket keeps its own commit on main. NEVER add Co-Authored-By or any author trailer.`,
      '     Let the pre-commit hooks run. If one fails, fix the cause and recommit; never --no-verify.',
      'Do not push. Do not open a PR. Do not touch another ticket in this batch, even if you can see it is wrong -- say so in notes instead.',
      HOUSE,
    ].join('\n'),
    {
      label: `express:commit#${t.number}`,
      phase: 'Commit',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['ejected', 'committed'],
        properties: {
          ejected: { type: 'boolean' },
          satisfied: { type: 'boolean', description: 'the work this ticket asks for is ALREADY on this branch -- nothing to commit, and nothing for the full pipeline to do either. Not the same as ejected: satisfied closes the ticket, ejected re-runs it through all 15 stages.' },
          eject_reason: { type: 'string' },
          committed: { type: 'boolean' },
          sha: { type: 'string' },
          subject: { type: 'string' },
          changed_lines: { type: 'integer' },
          files: { type: 'array', items: { type: 'string' } },
          checks_green: { type: 'boolean' },
          notes: { type: 'string' },
        },
      },
    },
  )

  // A claim of "already done" is NOT auto-closed. Measured 2026-08-31: #278's commit agent
  // claimed the work was already on the branch, and was wrong -- the full car then found
  // real missing transitive deps and shipped them. So a satisfied claim buys the cheap
  // outcome (no 15-stage re-run) but not the destructive one (closing the ticket): it is
  // recorded with its evidence for a human, who confirms or re-queues it.
  if (out && !out.committed && out.satisfied && !out.ejected) {
    const evidence = out.eject_reason || out.notes || 'the agent gave no evidence -- see the run journal'
    claimedSatisfied.push({ number: t.number, title: t.title, evidence })
    log(`#${t.number} claimed already-satisfied (UNVERIFIED, not closed, not re-queued): ${evidence}`)
    continue
  }

  if (!out || out.ejected || !out.committed) {
    const reason = out?.eject_reason || out?.notes || 'the agent returned nothing -- see the run journal'
    ejected.push({ number: t.number, title: t.title, reason })
    log(`#${t.number} ejected from the express lane: ${reason}`)
    continue
  }
  committed.push({ number: t.number, title: t.title, sha: out.sha || '', subject: out.subject || '', lines: out.changed_lines || 0, files: out.files || [], green: out.checks_green === true })
  log(`#${t.number} committed: ${out.subject || out.sha || 'no subject reported'} (~${out.changed_lines || '?'} lines)`)
}

if (!committed.length) {
  return {
    error: 'every ticket ejected or failed -- nothing was committed',
    issues: ISSUES,
    branch: BRANCH,
    worktree: WORKTREE,
    ejected,
    claimed_satisfied: claimedSatisfied,
    refused,
    next_step: `Run the ejected ones through process-issues as normal cars; the express lane found none of them small enough.${
      claimedSatisfied.length
        ? ` ${claimedSatisfied.map((c) => '#' + c.number).join(', ')} were claimed already-done and are NOT re-queued: confirm the evidence and close them, or re-queue them by hand. Do not take the claim on trust -- it is unverified.`
        : ''
    }`,
  }
}
const notGreen = committed.filter((c) => !c.green)
if (notGreen.length) log(`${notGreen.length} ticket(s) committed without reporting green checks: ${notGreen.map((c) => '#' + c.number).join(', ')}`)

// -- Ship ----------------------------------------------------------
// One gate run and one PR for the batch. Every ticket is named in the body with its own
// "Closes" line, so the PR is readable as a list of small changes rather than one opaque
// diff -- and so a reviewer can see what they are being asked to take on faith.
let ship = null
if (DO_SHIP) {
  phase('Ship')
  ship = await call(
    [
      `cd ${WORKTREE}. ${ISSUES.length ? '' : ''}${committed.length} small ticket(s) are committed on ${BRANCH}, one commit each. Push them and open ONE pull request. ${GH}`,
      `1. \`git push -u origin ${BRANCH}\`. ${PREPUSH}`,
      `   Then confirm with \`git ls-remote origin ${BRANCH}\`.`,
      `2. Check for an existing PR first: \`gh pr list --head ${BRANCH} --state open --json number,url\`. If one is open, \`gh pr edit <n> --body "<body>"\` -- never open a second PR for the same branch.`,
      `   Otherwise: \`gh pr create --base ${BASE} --head ${BRANCH} --title "<concise, naming the batch not one ticket>" --body "<body>"\``,
      '',
      'PR body, exactly this shape:',
      `   - One opening line: batched by ${RUN} because each change is a few lines; they share one CI run.`,
      '   - Then one bullet per ticket, in commit order:',
      ...committed.map((c) => `       - Closes #${c.number} — ${c.title} (~${c.lines} lines, ${(c.files || []).join(', ') || 'files not reported'})`),
      ejected.length ? `   - A "Not in this batch" line naming ${ejected.map((e) => '#' + e.number).join(', ')} -- ejected as bigger than rostered, going through the full pipeline instead.` : '',
      `   - A "How this was checked" line: the full check chain ran in a container per commit, and CI runs once over the batch.`,
      '   Every "Closes #N" must appear in the PR BODY as well as in its commit. GitHub links a PR to an issue from the body; a keyword only in a commit message closes the issue but leaves the PR unlinked.',
      '',
      'Do not merge. Do not mark it ready or draft either way beyond the default. Report the PR number and URL.',
    ]
      .filter(Boolean)
      .join('\n'),
    {
      label: 'express:ship',
      phase: 'Ship',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['pushed', 'pr_number'],
        properties: {
          pushed: { type: 'boolean' },
          pr_number: { type: 'integer' },
          pr_url: { type: 'string' },
          gate_passed: { type: 'boolean', description: 'true if the pre-push gate ran and passed' },
          notes: { type: 'string' },
        },
      },
    },
  )
  log(ship?.pushed ? `pushed ${BRANCH}, PR #${ship.pr_number || '?'}` : `push failed: ${ship?.notes || 'see the run journal'}`)
}

// -- Land ----------------------------------------------------------
// One CI watch and one merge for the whole batch -- the entire point of the lane. Rebase,
// not squash: squashing would collapse N tickets into one commit on main and lose the
// per-ticket history that the per-commit "Closes" trailers exist to give. Rebase-merge
// lands each ticket as its own commit, which is what the repo's usual squash-per-issue
// convention is actually after.
let land = null
if (DO_SHIP && ship?.pr_number && DO_MERGE) {
  phase('Land')
  land = await call(
    [
      `PR #${ship.pr_number} carries ${committed.length} small ticket(s) as ${committed.length} commit(s). Land it. ${GH}`,
      `1. Watch CI to a conclusion: \`gh pr checks ${ship.pr_number} --watch\`. If it goes red, STOP: set ci="red", merged=false and name the failing check. Do NOT try to fix it -- guessing which of ${committed.length} commits broke it is exactly the cost this lane was trading away.`,
      `   Then close the PR so the batch stops claiming these tickets: \`gh pr close ${ship.pr_number} --comment "batch CI went red; these tickets go back through the full pipeline one at a time"\`. Do NOT pass --delete-branch: the caller re-queues each ticket as a full car, and leaving ${BRANCH} on the remote is what lets a human bisect the batch afterwards.`,
      `2. Green: \`gh pr ready ${ship.pr_number}\` if it is a draft, then \`gh pr merge ${ship.pr_number} --rebase --delete-branch\`.`,
      '   REBASE, not squash. Each ticket must land as its own commit on main so its "Closes #N" trailer closes it and `git log` stays readable per ticket. If the repository refuses rebase merges, set merged=false and say so -- do NOT fall back to squash, which would close every ticket from one commit and destroy the per-ticket history.',
      `3. Confirm each issue closed itself: ${committed.map((c) => `\`gh issue view ${c.number} --json state\``).join(', ')}. Report any still open; do not close one by hand, that hides a broken link.`,
      WATCH_MAIN
        ? '4. The merge triggers CI on main. Poll `gh run list --branch main --limit 3 --json databaseId,status,conclusion,url` until it completes and report post_merge_green. If it went red, say so plainly and do NOT try to fix main -- revert or roll forward is a human decision.'
        : '4. Do NOT wait for main\'s CI. Leave post_merge_green unset: nothing in this run acts on it, and the PR CI already went green on this exact tree.',
      `5. Only if the merge SUCCEEDED: ${teardown(WORKTREE)} If CI went red, leave both in place -- the branch and its tree are what a human bisects.`,
      'Do not tag, do not release, do not bump a version.',
    ].join('\n'),
    {
      label: 'express:land',
      phase: 'Land',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['ci', 'merged'],
        properties: {
          ci: { type: 'string', enum: ['green', 'red', 'timed-out', 'unknown'] },
          merged: { type: 'boolean' },
          merge_commit: { type: 'string' },
          closed: { type: 'array', items: { type: 'integer' } },
          still_open: { type: 'array', items: { type: 'integer' } },
          post_merge_green: { type: 'boolean' },
          failing_check: { type: 'string' },
          notes: { type: 'string' },
        },
      },
    },
  )
  log(`express batch: CI ${land?.ci || 'unknown'}, ${land?.merged ? 'merged' : 'NOT merged'}${land?.notes ? ` -- ${land.notes}` : ''}`)
} else if (DO_SHIP && ship?.pr_number) {
  log(`PR #${ship.pr_number} is open and NOT merged (merge:false) -- ${committed.length} ticket(s) waiting on a human`)
}

// -- Report --------------------------------------------------------
const REPORT_PATH = opt.reportPath || `${RUNS_DIR}/express-lane-${ISSUES[0]}.md`
if (DO_REPORT) {
  const body = [
    `# express-lane -- ${committed.length} ticket(s) on one branch`,
    '',
    `${RUN}. Branch \`${BRANCH}\`${ship?.pr_url ? `, PR ${ship.pr_url}` : ''}. ${land?.merged ? `Merged as ${land.merge_commit || 'a rebase'}.` : 'Not merged.'}`,
    '',
    'One gate run, one CI run and one merge were shared across this batch. Through the normal lane each ticket',
    `would have paid its own -- roughly thirty-one minutes each, or about ${committed.length * 31} minutes for these ${committed.length}.`,
    '',
    land?.merged ? '## Landed' : '## Committed, but NOT landed',
    '',
    '| # | Ticket | ~lines | Files |',
    '| --- | --- | --- | --- |',
    ...committed.map((c) => `| #${c.number} | ${c.title} | ${c.lines || '?'} | ${(c.files || []).join(', ')} |`),
    '',
    DO_MERGE && !land?.merged
      ? `## Why nothing landed\n\nCI came back ${land?.ci || 'unknown'}${land?.failing_check ? ` (${land.failing_check})` : ''} and the PR was closed, so nothing above reached ${BASE}. Every ticket in the table is still open and goes back through process-issues one at a time. The commits are still on \`${BRANCH}\` on the remote if you want to bisect the batch first.`
      : null,
    ejected.length
      ? `## Ejected\n\nBigger than rostered, or not buildable as described. These did NOT land and are still open — run them through process-issues as normal cars.\n\n${ejected
          .map((e) => `- #${e.number} ${e.title} — ${e.reason}`)
          .join('\n')}`
      : null,
    satisfied.length
      ? `## Already satisfied\n\nTheir "## Verify" passed on the tip this batch was cut from, before anything was committed — an earlier merge had already done the work. Closed, not committed, and NOT re-queued as full cars.\n\n${satisfied
          .map((s) => `- #${s.number} ${s.title}${s.excerpt ? ` — ${s.excerpt}` : ''}`)
          .join('\n')}`
      : null,
    claimedSatisfied.length
      ? `## Claimed already done — needs a human\n\nTheir Commit agent reported the work was already on the branch and committed nothing. **This claim is unverified**: nothing ran their "## Verify". It has been wrong before — on 2026-08-31 #278's commit agent claimed already-fixed, and the full car then found real missing transitive dependencies and shipped them.\n\nSo these are neither closed nor re-queued. Confirm the evidence and close, or re-queue as a full car.\n\n${claimedSatisfied
          .map((c) => `- #${c.number} ${c.title} — claimed: ${c.evidence}`)
          .join('\n')}`
      : null,
    refused.length
      ? `## Refused at the door\n\n${refused.map((r) => `- #${r.number} ${r.title} — ${r.reason}`).join('\n')}`
      : null,
    land?.still_open?.length ? `\n## Still open after the merge\n\n${land.still_open.map((n) => `- #${n} — its "Closes" trailer did not fire; check the commit message`).join('\n')}` : null,
  ]
    .filter((x) => x !== null)
    .join('\n')
  await call(
    [
      'Write a file. That is the whole task: no analysis, no summarising, no edits to the text.',
      `Path: ${MAIN_REPO}/${REPORT_PATH} -- create the directory if it is missing, overwrite the file if it exists.`,
      'Content: everything between the two marker lines below, byte for byte. The markers are NOT part of the file.',
      '<<<REPORT-BEGINS>>>',
      body,
      '<<<REPORT-ENDS>>>',
      'Then confirm the file exists and report its path.',
    ].join('\n'),
    { label: 'express:report', phase: 'Land' },
  )
  log(`run report: ${REPORT_PATH}`)
}

return {
  lane: 'express',
  branch: BRANCH,
  worktree: WORKTREE,
  pr: ship?.pr_number || 0,
  pr_url: ship?.pr_url || '',
  ci: land?.ci || (DO_MERGE ? 'not run' : 'merge:false'),
  merged: land?.merged === true,
  merge_commit: land?.merge_commit || '',
  post_merge_green: land?.post_merge_green,
  // Committed is NOT landed. Nothing reaches main until the rebase-merge succeeds, so a red
  // CI, a refused merge or a run with merge:false leaves this empty -- reporting the batch as
  // landed there told the train to record N merged tickets that do not exist.
  landed: land?.merged ? committed.map((c) => c.number) : [],
  // The caller re-queues these as normal cars. Naming them here is the whole handover:
  // an ejected ticket that nobody picks up is a ticket this lane silently dropped. Same for
  // a batch that committed and then failed to land -- its PR is closed, so nothing else is
  // claiming these tickets and they need the lane that reviews one at a time.
  red_batch: DO_MERGE && !land?.merged ? committed.map((c) => c.number) : [],
  ejected: ejected.map((e) => ({ number: e.number, reason: e.reason })),
  // Closed without a commit: their acceptance already passed on the tip this batch was cut
  // from. Unlike an ejected ticket these are NOT re-queued as full cars -- there is no work
  // left in them for the long road either, and the caller must not send them down it.
  satisfied,
  // A Commit agent's UNVERIFIED claim that the work was already done. Not closed (the claim
  // has been wrong before) and not re-queued (re-running 15 stages on finished work is the
  // waste this channel exists to stop). A human confirms or re-queues each one.
  claimed_satisfied: claimedSatisfied,
  refused: refused.map((r) => ({ number: r.number, reason: r.reason })),
  probes: PROBES,
  probe_verdicts: probeVerdicts,
  report: DO_REPORT ? REPORT_PATH : 'not written',
  next_step: DO_MERGE && !land?.merged
    ? `The batch did NOT land (CI ${land?.ci || 'unknown'}${land?.failing_check ? `: ${land.failing_check}` : ''}). Its PR is closed and ${committed.map((c) => '#' + c.number).join(', ')} are still open -- run them through process-issues as normal cars. The commits are still on ${BRANCH} if you want to bisect the batch.`
    : ejected.length
      ? `${ejected.map((e) => '#' + e.number).join(', ')} were too big for this lane and are still open -- run them through process-issues as normal cars.`
      : land?.merged
      ? `All ${committed.length} landed as separate commits on ${BASE}.`
      : `PR ${ship?.pr_url || BRANCH} is open with ${committed.length} commit(s); merge it or re-run with { merge: true }.`,
}
