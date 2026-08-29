export const meta = {
  name: 'express-lane',
  description: 'Land several tiny tickets as one branch: a commit each, then one gate, one PR, one CI run, one merge',
  whenToUse:
    'For a handful of tickets that are each a few lines. The full per-issue pipeline spends about thirty-one minutes per ticket on the pre-push gate, the PR CI run and the post-merge main watch, and none of that shrinks with the diff -- so N tiny tickets cost N times thirty-one minutes for work that takes seconds. This lane shares all three across the batch: one worktree, one commit per ticket so history and issue-closing stay per-ticket, then a single gate, PR, CI watch and rebase-merge. run-merge-train calls it for the express cars it rosters; it can also be run directly on a list of small issues.',
  phases: [
    { title: 'Cut', detail: 'one worktree for the whole batch, from the current tip', model: 'sonnet' },
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
// The ejection threshold. A ticket claiming ~25 lines that turns into 200 is not a small
// ticket, it is a wrong estimate, and the whole justification for the reduced scrutiny
// above rests on the claim being true. Generous enough to absorb an honest miss.
const MAX_LINES = opt.maxLines || 50
// Only a train needs this: post_merge_green is its halt condition. A standalone run has
// nothing that acts on the answer, so it does not spend eleven minutes measuring it.
const WATCH_MAIN = opt.watchMain === true
const RUN = opt.runLabel || 'an express-lane run'

const GATE_LABEL = 'needs-triage'
const READY_LABEL = 'ready-for-agent'
const MAIN_REPO = 'C:/Users/User/Documents/dev/graphknows'
const RUNS_DIR = 'docs/agents/runs'
const GH = 'Use `gh`. If it is not on PATH, fall back to "/c/Program Files/GitHub CLI/gh.exe" (quote it).'
// Deterministic, so a resumed run reuses the same tree instead of stranding the first one.
const BRANCH = opt.branch || `express/${ISSUES.join('-')}`
const WORKTREE = opt.worktree || `C:/Users/User/Documents/dev/gk-express-${ISSUES[0]}`

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

// -- Commit --------------------------------------------------------
// Serial, on purpose: one worktree, and parallel writers to one tree collide. This is a
// loop, not a fan-out -- there is nothing here to parallelise and pretending otherwise
// would buy a conflict for no wall-clock.
phase('Commit')
const committed = []
const ejected = []
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
      'Ejecting is a normal outcome and costs nothing -- the ticket goes back to the full pipeline, which reviews it properly. Set ejected=true with the reason. Carrying on regardless is the failure: this lane reviews the whole batch once, so an oversized change here gets a fraction of the scrutiny it would get on its own.',
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
    refused,
    next_step: 'Run these through process-issues as normal cars; the express lane found none of them small enough.',
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
      `1. Watch CI to a conclusion: \`gh pr checks ${ship.pr_number} --watch\`. If it goes red, STOP: set merged=false, name the failing check, and do not try to fix it here. A red batch is bisected by a human or re-run through the full pipeline per ticket -- guessing which of ${committed.length} commits broke it is exactly the cost this lane was trading away.`,
      `2. Green: \`gh pr ready ${ship.pr_number}\` if it is a draft, then \`gh pr merge ${ship.pr_number} --rebase --delete-branch\`.`,
      '   REBASE, not squash. Each ticket must land as its own commit on main so its "Closes #N" trailer closes it and `git log` stays readable per ticket. If the repository refuses rebase merges, set merged=false and say so -- do NOT fall back to squash, which would close every ticket from one commit and destroy the per-ticket history.',
      `3. Confirm each issue closed itself: ${committed.map((c) => `\`gh issue view ${c.number} --json state\``).join(', ')}. Report any still open; do not close one by hand, that hides a broken link.`,
      WATCH_MAIN
        ? '4. The merge triggers CI on main. Poll `gh run list --branch main --limit 3 --json databaseId,status,conclusion,url` until it completes and report post_merge_green. If it went red, say so plainly and do NOT try to fix main -- revert or roll forward is a human decision.'
        : '4. Do NOT wait for main\'s CI. Leave post_merge_green unset: nothing in this run acts on it, and the PR CI already went green on this exact tree.',
      `5. Remove the worktree: \`git worktree remove ${WORKTREE}\` from the repo root (add --force only if it refuses over untracked files you have confirmed are build output).`,
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
    '## Landed',
    '',
    '| # | Ticket | ~lines | Files |',
    '| --- | --- | --- | --- |',
    ...committed.map((c) => `| #${c.number} | ${c.title} | ${c.lines || '?'} | ${(c.files || []).join(', ')} |`),
    '',
    ejected.length
      ? `## Ejected\n\nBigger than rostered, or not buildable as described. These did NOT land and are still open — run them through process-issues as normal cars.\n\n${ejected
          .map((e) => `- #${e.number} ${e.title} — ${e.reason}`)
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
  landed: committed.map((c) => c.number),
  // The caller re-queues these as normal cars. Naming them here is the whole handover:
  // an ejected ticket that nobody picks up is a ticket this lane silently dropped.
  ejected: ejected.map((e) => ({ number: e.number, reason: e.reason })),
  refused: refused.map((r) => ({ number: r.number, reason: r.reason })),
  report: DO_REPORT ? REPORT_PATH : 'not written',
  next_step: ejected.length
    ? `${ejected.map((e) => '#' + e.number).join(', ')} were too big for this lane and are still open -- run them through process-issues as normal cars.`
    : land?.merged
      ? `All ${committed.length} landed as separate commits on ${BASE}.`
      : `PR ${ship?.pr_url || BRANCH} is open with ${committed.length} commit(s); merge it or re-run with { merge: true }.`,
}
