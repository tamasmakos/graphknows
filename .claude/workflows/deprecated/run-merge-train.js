export const meta = {
  name: 'run-merge-train',
  description: 'Drain one spec issue: roster its sub-issue tickets, topo-order them, land each serially on the fresh tip via process-issues',
  whenToUse:
    'The drain half of the merge-train skill. Given a spec issue filed by prepare-tickets, it reads the spec\'s sub-issues, admits only tickets a human has approved (ready-for-agent present, needs-triage absent), orders them by their "## Blocked by" edges, and lands them one at a time -- each car cut from main as it stands after the previous merge, so sibling conflicts cannot happen. A failed car is parked with its dependents and the train continues; a red main after a merge halts everything. Departure, parked cars and halts are commented on the spec issue as they happen and the wrap-up posts the full ledger, so a killed run resumes by relaunching -- position rebuilds from GitHub state alone.',
  phases: [
    { title: 'Roster', detail: 'read the spec\'s sub-issues, admit only human-approved tickets, order them', model: 'sonnet' },
    { title: 'Express', detail: 'the tiny tickets first, batched onto one branch with one CI run', model: 'sonnet' },
    { title: 'Drain', detail: "per car: run its own \"## Verify\" against the tip, dispatch process-issues with merge, classify the outcome", model: 'sonnet' },
    { title: 'Recover', detail: 'one rebase-and-resolve attempt when a green PR failed to merge', model: 'sonnet' },
    { title: 'Ledger', detail: 'departure, parked and halt comments post immediately; landed cars batch into the wrap-up', model: 'sonnet' },
    { title: 'Report', detail: 'one wrap-up agent: single main flake-check, the ledger comment, the train report', model: 'sonnet' },
  ],
}

// args: { spec: <issue number>, merge: true, maxCars?: 8, runLabel?: 'nightly',
//         car?: { <extra args merged into every process-issues dispatch> },
//         express?: true, report?: true, reportPath?: 'docs/agents/runs/x.md' }
//
// TWO LANES. Cars whose bodies claim a few lines of work ride the EXPRESS lane first: one
// branch, one commit each, one gate run, one PR, one CI run, one merge. Everything else
// drains one car at a time as before. The split exists because the normal lane's cost is
// almost entirely per-PR waiting -- about thirty-one minutes of gate, CI and post-merge
// watch that does not shrink with the diff -- so five one-line tickets cost two and a half
// hours to change five lines. { express: false } sends everything the long way.
//
// The launch is deliberately two words: `spec` names the train, `merge: true` authorizes
// writing to main N times. Neither implies the other, and the run refuses to start with
// only one of them. `force` is refused outright: it exists so a HUMAN can override the
// triage gate on one named issue, and combined with a train it would delete the only
// per-car approval in an N-merge run.
//
// The human's controls, all durable GitHub state the train re-reads at each dispatch:
//   - apply "ready-for-agent" to a ticket   -> it may board
//   - re-apply "needs-triage" mid-train      -> that car parks at its dispatch check
//   - close a ticket                         -> it is treated as landed/irrelevant
// There is no pause button because there is no pause: a workflow cannot wait for a person.
// Parking and halting are the brakes, and both leave a comment saying why.
const opt = args || {}
const SPEC = Number.isInteger(opt.spec) ? opt.spec : 0
if (!SPEC) {
  return { error: 'run-merge-train needs a spec issue number. Pass args as an OBJECT: { spec: 201, merge: true }.' }
}
if (opt.merge !== true) {
  return { error: 'run-merge-train writes to main once per car, so the launch must say so: { spec: ' + SPEC + ', merge: true }.' }
}
if (opt.force) {
  return { error: 'force is refused in a train: it bypasses the per-ticket human gate, which is the only approval standing behind ' + 'N merges. Override a single gated issue with process-issues { issues: [n], force: true } instead.' }
}
if (opt.issues || opt.from) {
  return { error: 'a train rosters from its spec\'s sub-issues; issues/from would be a second roster. Hand-pick an order by running process-issues directly.' }
}
const MAX_CARS = opt.maxCars || 8
const RUN = opt.runLabel || 'a merge-train run'
const DO_REPORT = opt.report !== false
// Whether to run a car's own "## Verify" against the tip before cutting it a worktree.
// 'shadow' reports the verdict and dispatches anyway; 'on' acts on it; 'off' skips it.
//
// Shadow WAS the default, on the theory that a new predicate deciding whether work happens
// should be watched before it is trusted. The watching is done and the verdict is in.
// Measured 2026-08-31 on the #307 train: the probe proved #247's "## Verify" already passed
// (`pytest tests/test_repo_hygiene.py::TestNoRuntimeDownloads`, 2 passed), and shadow mode
// dispatched a full 15-stage car for it anyway -- roughly 1.5M tokens to redo finished work.
// Paying for a measurement and then ignoring it is worse than not measuring.
//
// So: act on it. The risk this accepts is a verify too weak to see missing work, which
// closes a ticket early; the close comment says exactly that and invites a reopen. That is
// a cheaper mistake than the one shadow mode made, and a visible one.
const PROBES = ['shadow', 'on', 'off'].includes(opt.probes) ? opt.probes : 'on'
// Extra args merged into every dispatch (e.g. { baseline: false } to skip the per-car
// baseline once the tip is trusted). merge/issues/force stay ours.
const CAR_ARGS = { ...(opt.car || {}) }
delete CAR_ARGS.merge
delete CAR_ARGS.issues
delete CAR_ARGS.from
delete CAR_ARGS.force
delete CAR_ARGS.watchMain

// Same vocabulary as prepare-tickets.js and process-issues.js; gates.test.mjs asserts they agree.
const GATE_LABEL = 'needs-triage'
const READY_LABEL = 'ready-for-agent'
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
// Where an agent's own output goes. One directory, one ignore rule, and it sits beside the
// rules agents READ (definition-of-done.md, triage-labels.md) rather than off in a scratch
// tree of its own -- so "what does an agent read here, and what did it write" is one folder.
// docs/agents/runs/ is gitignored; the files beside it are not, deliberately: a rule an
// agent cannot open from a fresh clone is a rule that gets restated into every ticket
// instead, which is the bloat this whole pipeline was rebuilt to stop.
const RUNS_DIR = 'docs/agents/runs'
const GH = `Use \`gh\`. If it is not on PATH, fall back to "${opt.ghPath || '/c/Program Files/GitHub CLI/gh.exe'}" (quote it).`

const RUN_ON = { model: 'sonnet', effort: 'medium' }
const call = (prompt, opts = {}) => agent(prompt, { ...RUN_ON, ...opts })

// Stopping the stack a car started is part of finishing, not an afterthought for a human.
// Third copy of this, after process-issues.js and express-lane.js -- workflow scripts cannot
// import, and the train needs its own because the one path where IT merges is the one place
// neither of the others can reach: a car recovered below merged AFTER its child had already
// returned "not merged", so the child's own teardown-on-success never fired.
//
// `docker compose` names its project after the worktree directory, so a car's ArcadeDB comes
// up as gk-issue-<n>-arcadedb-1 and NOTHING stops it. Measured 2026-08-30: five leaked
// stacks from issues closed days earlier held ~14.7GB of a 31.2GB Docker allocation and the
// pre-push gate died with `Killed` at 51% of the suite -- on every branch, for every car,
// until they were stopped by hand. A recovered car is the likeliest one to have a stack up:
// it went through the whole pipeline before its merge was refused.
//
// Only on success: a parked car keeps its stack AND its worktree, because that is the state
// a human needs to debug it.
const teardown = (worktree) =>
  `Then tear down what this car started, in this order: \`cd ${worktree} && docker compose down --remove-orphans\` (its ArcadeDB stack is named after that directory and nothing else ever stops it -- leaked stacks OOM-kill the pre-push gate for every branch, not just this one), then \`git worktree remove ${worktree}\` from the repo root. If either refuses, say so in notes and leave it -- do not force past a refusal you have not read.`

// -- the dispatch probe --------------------------------------------
//
// Every ticket carries a falsifiable acceptance test in its "## Verify" section and nothing
// in this line has ever run it, at either end -- so neither its passing nor its failing has
// meant anything anywhere. A ticket satisfied on main before it boards (by a sibling car, an
// express batch, an overlapping slice, a human PR) therefore pays the whole car: baseline,
// an opus plan, a red task, implement, review, verify, ship, ~31 minutes of gate and CI --
// to land nothing. Its own acceptance would have said so in about a minute.
//
// Probing NEVER touches the main checkout: a `git checkout origin/main` there would move a
// human's working tree out from under them. It gets a detached worktree of its own, cut once
// and reused, exactly as Baseline does in process-issues.
const SIBLING_DIR = MAIN_REPO.replace(/\/[^/]+$/, '')
const PROBE_TREE = `${SIBLING_DIR}/gk-probe`
const winPath = (p) =>
  String(p)
    .replace(/\\/g, '/')
    .replace(/^\/([a-zA-Z])\//, (_, drive) => `${drive.toUpperCase()}:/`)

const PROBE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ran', 'passes'],
  properties: {
    ran: { type: 'boolean', description: 'true only if you executed the command and read its result. A command that runs and fails RAN.' },
    passes: { type: 'boolean', description: 'true only if the command completed successfully on the current tip' },
    excerpt: { type: 'string', description: 'the last few lines of its output, or why it could not be run at all' },
    tip_sha: { type: 'string', description: 'the short SHA the probe ran against' },
    closed: { type: 'boolean', description: 'true if you closed the issue' },
  },
}

const probePrompt = (n, act) =>
  [
    `Run ONE issue's own acceptance check against the current tip of main. Change no code, fix nothing, implement nothing. ${GH}`,
    `1. \`gh issue view ${n} --json body\` and find its "## Verify" section. It holds one runnable command. If there is no such section, or it holds prose rather than a command, set ran=false and say which -- do not invent a command.`,
    `2. Get a clean tree at the tip, WITHOUT touching ${MAIN_REPO}'s own checkout -- a person may be working in it:`,
    `   from ${MAIN_REPO}: \`git fetch origin\`, then \`git worktree add ${PROBE_TREE} --detach origin/main\` (if ${PROBE_TREE} already exists, cd there and \`git checkout -q --detach origin/main\` instead).`,
    '   Report the short SHA under tip_sha.',
    `3. Run that command in ${PROBE_TREE}. If it names the test suite, run it in a container mounted on that tree:`,
    '```',
    `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${winPath(PROBE_TREE)}:/app" -w /app graphknows-workspace \\`,
    `  sh -c '<the command from "## Verify">'`,
    '```',
    '   Both env vars are load-bearing. Never run the suite on the host, and never report a result you did not observe.',
    '4. Report the outcome, and be exact about the difference between these two:',
    '   - The command RAN and FAILED -- an assertion failed, a symbol does not exist yet, an import errors because the module is not written. That is ran=true, passes=false, and it is the NORMAL answer for a ticket with work left in it.',
    '   - The command could not be run AT ALL -- there is no command, it is malformed, docker is unavailable. That is ran=false.',
    '   Only a command that ran to a successful conclusion is passes=true.',
    act
      ? `5. ONLY if passes=true: the work this ticket asks for is already on main. Close it: \`gh issue close ${n} --comment "Its \\"## Verify\\" already passes on main at <tip_sha> before any work was done on this ticket, so an earlier merge already satisfied it. Closed by ${RUN} at dispatch rather than cutting a car for a no-op. Reopen if the check is simply too weak to see the missing work."\` and set closed=true. If passes is anything else, close nothing.`
      : '5. Close nothing and comment nowhere. This run is only measuring what the probe would have decided.',
  ].join('\n')

// -- pure helpers, sliced out and exercised by gates.test.mjs ------
//
// Admission is the per-car half of the safety story: the roster admits only tickets a
// human said yes to. Requiring READY_LABEL's presence (not merely GATE_LABEL's absence)
// is what turns "nobody objected" into "somebody agreed" -- the merge-queue pattern of
// per-item approval, batch execution.
const admit = (car, rosterNumbers) => {
  if (car.state !== 'OPEN') return 'not open'
  const labels = car.labels || []
  if (labels.includes(GATE_LABEL)) return `still carries "${GATE_LABEL}" -- no human has agreed to it`
  if (!labels.includes(READY_LABEL)) return `no "${READY_LABEL}" label -- a human has not said yes to this car`
  if (car.open_pr) return `parked from an earlier run -- PR already open (${car.open_pr})`
  const harness = (car.files_named || []).filter((f) => HARNESS.some((h) => String(f).replace(/\\/g, '/').toLowerCase().endsWith(h)))
  if (harness.length) return `it rewrites the harness (${harness.join(', ')}) -- a car cannot change the gate it must pass or the CI it must green; land it by hand`
  const outside = (car.blocked_by || []).filter((b) => !rosterNumbers.includes(b) && car.blocker_open?.[b] === true)
  if (outside.length) return `blocked by ${outside.map((n) => '#' + n).join(', ')}, open and not on this train`
  return ''
}

// Which cars can share one branch, one CI run and one merge instead of paying their own.
//
// The cost this dodges is not thinking, it is waiting: ~9 minutes of pre-push gate, ~11 of
// PR CI and ~11 of post-merge main watch, per car, regardless of whether the diff is one
// line or four hundred. Five one-line tickets cost about two and a half hours through the
// normal lane and there is no stage to skip that changes it -- CI cannot be skipped on a
// merge to main, and a train cannot stop watching main because a red main is its halt
// condition. Sharing one of each across a batch is the only lever there is.
//
// Returns '' when the car may take the express lane, or the reason it may not -- same
// convention as admit(), which has already run and already refused anything touching the
// harness, so this does not re-check that.
// The near side of THE REVIEW BOUNDARY OF THE WHOLE SYSTEM: this admits a car at <=30
// expected lines, express-lane.js's MAX_LINES ejects one at >50 actual. The 20-line slack is
// deliberate -- see the note there -- and it means a ticket rostered at 30 can legitimately
// land at 49 having had only batch-level review. Move one number, look at the other.
const EXPRESS_MAX_LINES = 30
const EXPRESS_MAX_FILES = 2
const isExpress = (car, rosterNumbers) => {
  // Sizing comes from the body when the author committed to a number, and from the roster
  // reading the code when they did not. Trusting only the body was backwards: the body's
  // number is a guess made BEFORE anyone opened the files, and the roster's is one made
  // after. What actually protects the lane is not who guessed but the ejection net --
  // express-lane.js drops anything over 50 real lines, commits nothing, and re-queues it as
  // a full car. So a wrong estimate costs a restart, not a badly-reviewed merge, while a
  // ticket needlessly sent down the long road costs ~2h of gate, CI and main-watch.
  //
  // "unknown" still takes the long road: an unsized ticket fails closed, as before.
  if (car.sizing === 'unknown' || !car.sizing) return 'could not be sized'
  // A measured size has to be defended. Without the files it claims to have read, the
  // "measurement" is the invented number the old rule was right to refuse.
  if (car.sizing === 'measured' && !(car.sizing_evidence || '').trim()) return 'sized as measured but cites no evidence'
  if (car.sizing === 'measured' && !(car.files_named || []).length) return 'sized as measured but names no files to have read'
  if (!Number.isInteger(car.expected_lines) || car.expected_lines <= 0) return 'no usable size'
  if (car.expected_lines > EXPRESS_MAX_LINES) return `~${car.expected_lines} lines, over the ${EXPRESS_MAX_LINES} this lane takes`
  // files_named is every path the BODY MENTIONS, and it has to stay that way: the harness
  // veto in admit() can only refuse a car that rewrites the CI or the gate if it sees every
  // path, so that field fails closed by over-reporting. Counting it HERE asked it a
  // different question -- how many files does this change EDIT -- and punished a ticket for
  // citing its sources. Measured 2026-08-30: #265 (a two-line docs edit naming three files
  // to READ) and #188 (a one-line Makefile fix naming five) each took the ~14-stage lane,
  // about 28 wasted agents on one train, and a group-P roster put 10 of 14 cars on the long
  // road the same way. Two questions, two fields.
  const files = (car.files_to_edit || []).length ? car.files_to_edit : car.files_named || []
  if (files.length > EXPRESS_MAX_FILES) return `edits ${files.length} files, over ${EXPRESS_MAX_FILES}`
  // Express cars all land in one commit range, so anything waiting on another car on this
  // train cannot be in it -- its blocker has not landed yet when the batch is cut.
  if ((car.blocked_by || []).some((b) => rosterNumbers.includes(b))) return 'blocked by another car on this train'
  return ''
}

// The verification a train runs on lives in these files. A car that edits one of them
// invalidates its own evidence: the pre-push gate it runs is the gate it just rewrote, and
// the CI it must turn green is the CI it just rewrote. Neither result means anything, and
// nothing downstream of the merge can tell. So such a ticket never boards -- it is a
// human's commit, reviewed as a normal PR.
const HARNESS = ['.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/gate.sh', '.pre-commit-config.yaml']

// Kahn again -- same shape as prepare-tickets.js checkHandover(), which validated these edges before
// the tickets were ever filed. Duplicated because workflow scripts cannot import.
const topoOrder = (cars) => {
  const nums = cars.map((c) => c.number)
  const order = []
  let pending = cars.map((c) => ({ n: c.number, deps: (c.blocked_by || []).filter((b) => nums.includes(b) && b !== c.number) }))
  while (pending.length) {
    const ready = pending.filter((r) => r.deps.every((d) => order.includes(d)))
    if (!ready.length) return { order, cyclic: pending.map((r) => r.n) }
    for (const r of ready) order.push(r.n)
    pending = pending.filter((r) => !ready.includes(r))
  }
  return { order, cyclic: [] }
}

// Everything that transitively depends on a parked car parks with it -- reported at once,
// not rediscovered one skipped dispatch at a time.
const dependentsOf = (cars, n) => {
  const out = new Set()
  let grew = true
  while (grew) {
    grew = false
    for (const c of cars) {
      if (out.has(c.number)) continue
      if ((c.blocked_by || []).some((b) => b === n || out.has(b))) {
        out.add(c.number)
        grew = true
      }
    }
  }
  return [...out]
}

// -- Roster --------------------------------------------------------
phase('Roster')
const roster = await call(
  [
    `You are building the roster of a merge train for the GraphKnows repository. The train is spec issue #${SPEC}. ${GH}`,
    `1. Read the spec: \`gh issue view ${SPEC} --json number,title,state,labels\`. If it is not open, report it and list no children.`,
    `2. List its sub-issues: \`gh issue view ${SPEC} --json subIssues\`. These ARE the roster candidates -- do not search for children any other way; search is eventually consistent and this read is not.`,
    '3. For EACH sub-issue, read `gh issue view <n> --json number,title,state,labels,body` and extract:',
    '   - its labels and state, verbatim',
    '   - the issue numbers in its "## Blocked by" section (empty if it says it can start immediately)',
    '   - for each blocker that is NOT itself a sub-issue of the spec: whether that blocker is currently open (`gh issue view <b> --json state`)',
    '   - every repository file path its body names, verbatim, under files_named. Miss none: the script refuses to board a car that rewrites the CI or the gate, and it can only see the paths you report. Over-report here rather than under-report.',
    '   - under files_to_edit, the SUBSET of those the change actually MODIFIES. A well-written ticket cites more files than it touches: the module it reads for context, the file whose comment is the source of truth, the place the symptom was observed. Those belong in files_named and NOT here. Getting this wrong in the cautious direction is expensive, not safe -- an over-long files_to_edit sends a two-line fix down a lane that costs about two hours of gate, CI and main-watch it did not need.',
    '   - its size, under expected_lines + sizing, by ONE of these two routes:',
    '     * The body has an "Estimated diff: ~N lines" line -> expected_lines = N, sizing = "stated". The author committed to a number; take it.',
    '     * It does not -> SIZE IT YOURSELF FROM THE CODE, sizing = "measured". Open every path under files_named, find the functions the ticket actually changes, and count the lines that have to move. Put what you read in sizing_evidence: the symbols you found and why the change is that size. A number you did not get from reading the files is not measured -- report sizing = "unknown" instead, and say so.',
    '   Sizing is worth the legwork: a ticket wrongly sent down the long road costs about two hours of gate, CI and main-watch that a batched one shares. Guessing is not, so guessing has its own value that fails closed. Wrong-but-measured is cheap -- the express lane ejects anything over 50 real lines, commits nothing, and it goes back through the full pipeline.',
    // A ticket with no runnable "## Verify" cannot be probed, so it can never be shown to be
    // already done and always pays for a full car. On the #307 train every single candidate
    // was in this state, which is why the probe returned ran=false five times out of five.
    '   - under runnable_verify, whether its "## Verify" section names a command that could be run verbatim. Prose about how one WOULD check it is not a command, and neither is a checklist of acceptance criteria. Report what is there, do not judge whether it is a good check.',
    '4. Check work in flight: `gh pr list --state open --json number,headRefName,title,url` and `git ls-remote --heads origin "fix/issue-*"`. A candidate whose fix/issue-<n> branch has an open PR gets that PR URL under open_pr.',
    '5. From that same PR list, report every open PR that is NOT one of these candidates under other_prs, with its number, title and URL. A train lands N commits on main; a PR opened before it departed is the thing most likely to be made moot or unmergeable by them, and nobody finds out until the conflict.',
    'Sizing is the one place you measure rather than transcribe; everything else is facts as written. Do not decide who boards or which lane they take -- the script does that from what you report, and it will show its reasons.',
  ].join('\n'),
  {
    label: 'roster',
    phase: 'Roster',
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['spec_title', 'spec_open', 'children'],
      properties: {
        spec_title: { type: 'string' },
        spec_open: { type: 'boolean' },
        children: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['number', 'title', 'state', 'labels', 'blocked_by'],
            properties: {
              number: { type: 'integer' },
              title: { type: 'string' },
              state: { type: 'string', description: 'OPEN or CLOSED, as GitHub reports it' },
              labels: { type: 'array', items: { type: 'string' } },
              blocked_by: { type: 'array', items: { type: 'integer' } },
              blocker_open: {
                type: 'object',
                additionalProperties: { type: 'boolean' },
                description: 'for blockers outside the spec: issue number -> is it still open',
              },
              open_pr: { type: 'string', description: 'URL of an already-open PR for this ticket, if any' },
              files_named: { type: 'array', items: { type: 'string' }, description: 'every repository file path the ticket body names, including ones it only cites as context -- the harness veto reads this and must over-report, never under-report' },
              files_to_edit: {
                type: 'array',
                items: { type: 'string' },
                description: 'the SUBSET of files_named the change actually modifies, from the same reading that produced expected_lines. Omit a path the ticket only cites as a source to read, as the source of truth for the fix, or as the place a symptom was observed. Empty or absent falls back to files_named.',
              },
              expected_lines: { type: 'integer', description: 'lines the change is expected to move: the N from an "Estimated diff: ~N lines" line, or your own count from reading the files. 0 when you could not size it.' },
              sizing: {
                type: 'string',
                enum: ['stated', 'measured', 'unknown'],
                description: '"stated" = from the body\'s Estimated diff line; "measured" = you read the named files and counted; "unknown" = neither, so it takes the long road',
              },
              sizing_evidence: {
                type: 'string',
                description: 'required when sizing is "measured": the files and symbols you read, and why the change is that size',
              },
              runnable_verify: {
                type: 'boolean',
                description: 'true only if the body has a "## Verify" section naming a command that could be executed verbatim. Prose describing how one would check ("build the image and run an ingest"), an acceptance-criteria checklist, or no Verify section at all are all false.',
              },
            },
          },
        },
        other_prs: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['number', 'title'],
            properties: { number: { type: 'integer' }, title: { type: 'string' }, url: { type: 'string' } },
          },
          description: 'open PRs that are not cars on this train -- the ones the merges are most likely to break',
        },
        notes: { type: 'string' },
      },
    },
  },
)

if (!roster?.spec_open) {
  return { spec: SPEC, error: `spec issue #${SPEC} is not open (or could not be read) -- nothing to drain`, notes: roster?.notes || '' }
}
const children = roster.children || []
if (!children.length) {
  return { spec: SPEC, spec_title: roster.spec_title, error: 'the spec has no sub-issues. prepare-tickets files tickets with --parent; a hand-filed ticket joins the train via `gh issue edit ' + SPEC + ' --add-sub-issue <n>`.' }
}

const rosterNumbers = children.filter((c) => c.state === 'OPEN').map((c) => c.number)
const admitted = []
const excluded = []
for (const c of children) {
  const reason = admit(c, rosterNumbers)
  if (reason) excluded.push({ number: c.number, title: c.title, reason })
  else admitted.push(c)
}
for (const x of excluded) log(`#${x.number} not boarding: ${x.reason}`)

// An unprobeable car cannot be shown to be already done, so it always pays for a full car
// even when the work is finished. Not a refusal -- most of the existing backlog is in this
// state -- but it must be visible BEFORE the tokens are spent, not discovered afterwards in
// a wrap-up nobody reads. Measured 2026-08-31: 5 of 5 on the #307 train, and #247's car was
// pure waste as a result.
const unprobeable = admitted.filter((c) => c.runnable_verify === false)
if (unprobeable.length) {
  log(
    `${unprobeable.length} of ${admitted.length} admitted car(s) have no runnable "## Verify" and cannot be probed: ${unprobeable
      .map((c) => '#' + c.number)
      .join(', ')}. Each pays for a full car even if its work is already done -- add a runnable Verify to the bodies to make the cheap check possible.`,
  )
}

const { order, cyclic } = topoOrder(admitted)
if (cyclic.length) {
  // rnd's checkHandover refuses to file a cycle, so reaching this means hand-edited
  // bodies. Refuse the whole train: an order that ignores an edge merges work against
  // code that does not exist yet.
  return { spec: SPEC, error: `the "## Blocked by" edges contain a cycle: ${cyclic.map((n) => '#' + n).join(', ')}. Fix the bodies and relaunch.`, excluded }
}
if (!order.length) {
  return { spec: SPEC, spec_title: roster.spec_title, cars: [], excluded, note: 'no car is both human-approved and unblocked -- nothing to drain' }
}
const fullOrder = order.slice(0, MAX_CARS)
if (order.length > fullOrder.length) log(`maxCars=${MAX_CARS}: ${order.length - fullOrder.length} admitted car(s) will not be dispatched this run`)

const byNumber = new Map(admitted.map((c) => [c.number, c]))

// The express set: cars small enough to share one branch, one gate run, one CI run and one
// merge. They go FIRST and together, because the normal lane's cost is per-PR waiting that
// does not shrink with the diff -- see isExpress() above for the arithmetic. Everything
// else drains one car at a time exactly as before.
const EXPRESS = opt.express !== false
const expressSet = []
if (EXPRESS) {
  const notExpress = []
  for (const n of fullOrder) {
    const why = isExpress(byNumber.get(n), rosterNumbers)
    if (why) notExpress.push(`#${n} (${why})`)
    else expressSet.push(n)
  }
  // Say why each car missed the lane. Silence here reads as "nothing was small enough" when
  // the real answer can be "nothing was sized" -- a whole train took the long road on that
  // ambiguity, at roughly two hours a car.
  if (notExpress.length) log(`long road: ${notExpress.join(', ')}`)
  // One express car is not a batch: it would pay the same single gate-and-CI it pays as a
  // normal car, and get less review for it. Two is the smallest set where the lane wins.
  if (expressSet.length === 1) {
    log(`#${expressSet[0]} is small enough for the express lane but is the only one -- running it as a normal car, which costs the same and reviews more`)
    expressSet.length = 0
  }
}
const dispatchList = fullOrder.filter((n) => !expressSet.includes(n))
log(
  `train for #${SPEC} "${roster.spec_title}": ` +
    (expressSet.length ? `express batch [${expressSet.map((n) => '#' + n).join(' + ')}] -> ` : '') +
    (dispatchList.length ? dispatchList.map((n) => '#' + n).join(' -> ') : 'no full cars'),
)
const ledger = (text) =>
  call(
    [
      `Post ONE comment on GitHub issue #${SPEC} and do nothing else. ${GH}`,
      'The text is final -- do not rephrase it, extend it, or add a signature.',
      `\`gh issue comment ${SPEC} --body "<the text below, verbatim>"\``,
      '<<<COMMENT>>>',
      text,
      '<<<END COMMENT>>>',
    ].join('\n'),
    // Pure transcription -- the text is composed here, the agent only posts it. The
    // cheapest model cuts what each of a spawn's ~40k overhead tokens costs.
    { label: 'ledger', phase: 'Ledger', model: 'haiku', effort: 'low' },
  )

// Per-car outcomes accumulate here and go up as ONE comment in the wrap-up, not one
// ~40k-token agent per event -- the ledger agents alone cost more than any real stage
// across recent runs. The split is by what a KILLED run would lose: a landed car is
// durable without a comment (closed issue, merged PR, its own run report on disk), so it
// batches; a parked car with no PR leaves no other trace of WHY it parked, so parking
// still posts immediately -- as do departure and a halt, the moments a human waits on.
const ledgerLines = []

// The roster comment first: if the session dies mid-train, this plus the per-car comments
// is the whole recovery story -- a relaunch rebuilds position from GitHub state alone
// (closed child = landed, open PR = parked, open unblocked child = next car).
// Work in flight that is not on this train. Surfaced at departure, not at the conflict:
// N merges land on main, and a PR opened before any of them is the thing they quietly
// invalidate. The train does not touch it -- a human decides whether it is still wanted.
const otherPrs = roster.other_prs || []
for (const p of otherPrs) log(`open PR not on this train: #${p.number} "${p.title}" -- these merges may make it moot or unmergeable`)

await ledger(
  `${RUN}: train departing for this spec.` +
    (expressSet.length
      ? ` Express batch first — ${expressSet.map((n) => '#' + n).join(', ')} share one branch, one CI run and one merge because each is a few lines.`
      : '') +
    ` Order: ${dispatchList.length ? dispatchList.map((n) => '#' + n).join(' -> ') : 'no full cars'}.` +
    (excluded.length ? ` Not boarding: ${excluded.map((x) => `#${x.number} (${x.reason})`).join('; ')}.` : '') +
    (otherPrs.length ? ` Open PRs not on this train, check them after it lands: ${otherPrs.map((p) => '#' + p.number).join(', ')}.` : ''),
)

// -- Express -------------------------------------------------------
// One child workflow for the whole batch. It owns the worktree, the per-ticket commits and
// the single PR; this train only decides WHO rides and what to do with whoever is ejected.
const cars = []
// What the probe decided, per car and per express ticket. Shadow mode is worthless without
// this: a predicate that reports nothing countable can never be promoted, and the wrap-up
// comment below is where the count is read. Declared here rather than beside the drain loop
// because the express batch fills it first.
const probeVerdicts = []
// Whether a dispatch probe actually cut PROBE_TREE. Not derivable from probeVerdicts: the
// express lane probes in its own batch tree and contributes verdicts without touching this
// one, and a train halted by that batch never reaches a car, so nothing is created to remove.
let probedTip = false
let expressResult = null
if (expressSet.length) {
  phase('Express')
  expressResult = await workflow('express-lane', {
    issues: expressSet,
    merge: true,
    repo: opt.repo,
    ghPath: opt.ghPath,
    // Same reason the full cars get it: post_merge_green is this train's halt condition,
    // and a batch that poisons main must stop everything behind it.
    watchMain: true,
    // The express batch is cut before the drain loop runs, so nothing has probed these
    // tickets yet -- the lane does it once for the whole batch in its own worktree.
    probes: PROBES,
    runLabel: `${RUN} (express batch for #${SPEC})`,
  })

  const landed = expressResult?.landed || []
  const ejected = expressResult?.ejected || []
  // Tickets whose own "## Verify" already passed on the tip the batch was cut from. The lane
  // closed them and committed nothing. They are NOT re-queued -- unlike an ejected ticket,
  // there is no work left in them to send down the long road.
  const satisfied = expressResult?.satisfied || []
  for (const s of satisfied) {
    cars.push({ issue: s.number, title: byNumber.get(s.number)?.title || '', state: 'already-satisfied', reason: `its "## Verify" already passed before the batch was cut${s.excerpt ? ` -- ${s.excerpt}` : ''}` })
  }
  // A Commit agent's unverified claim that the work was already done. Deliberately neither
  // closed nor re-queued: closing on an unverified claim has been wrong before (#278,
  // 2026-08-31), and re-queueing runs 15 stages over finished work -- the exact waste that
  // measured 1.6M tokens. It parks for a human instead, which is the only honest state.
  const claimedSatisfied = expressResult?.claimed_satisfied || []
  for (const c of claimedSatisfied) {
    cars.push({
      issue: c.number,
      title: byNumber.get(c.number)?.title || '',
      state: 'parked',
      reason: `the express lane claimed this work is already done, but nothing verified it -- confirm and close, or re-queue as a full car. Claim: ${c.evidence}`,
    })
  }
  for (const v of expressResult?.probe_verdicts || []) probeVerdicts.push(v)
  // Committed but not merged: a red shared CI, or a refused merge. The lane closed its PR,
  // so these tickets have no open claim on them and are ours to re-dispatch.
  const redBatch = expressResult?.red_batch || []
  for (const n of landed) {
    cars.push({ issue: n, title: byNumber.get(n)?.title || '', state: 'landed', reason: 'express batch', pr_url: expressResult?.pr_url || '', merge_commit: expressResult?.merge_commit || '' })
  }
  // An ejected ticket turned out bigger than its estimate claimed; a red_batch ticket
  // committed fine and then the shared CI went red. Neither landed, and neither is a ticket
  // anyone should have to notice went missing -- both go back to the lane that reviews one at
  // a time. Appending to a topo-ordered list is safe: isExpress() refuses any car blocked by
  // another car on this train, so nothing here has a blocker still ahead of it.
  const reQueued = [...ejected.map((e) => ({ ...e, why: `ejected as bigger than rostered (${e.reason})` })), ...redBatch.map((n) => ({ number: n, why: 'the express batch went red' }))]
  for (const e of reQueued) {
    if (!dispatchList.includes(e.number) && byNumber.has(e.number)) {
      dispatchList.push(e.number)
      log(`#${e.number}: ${e.why} -- re-queued as a full car`)
    }
  }
  for (const r of expressResult?.refused || []) {
    cars.push({ issue: r.number, title: byNumber.get(r.number)?.title || '', state: 'parked', reason: `express lane refused it: ${r.reason}` })
  }

  ledgerLines.push(
    `express batch ${expressResult?.merged ? 'landed' : `did NOT land (CI ${expressResult?.ci || 'unknown'})`} — ${landed.length ? landed.map((n) => '#' + n).join(', ') : 'nothing'}${
      expressResult?.pr_url ? ` (${expressResult.pr_url})` : ''
    }.${ejected.length ? ` Ejected as bigger than rostered and re-queued as full cars: ${ejected.map((e) => '#' + e.number).join(', ')}.` : ''}${
      satisfied.length ? ` Closed without a commit, their "## Verify" already passing on the tip: ${satisfied.map((s) => '#' + s.number).join(', ')}.` : ''
    }${redBatch.length ? ` Its PR is closed and ${redBatch.map((n) => '#' + n).join(', ')} ${redBatch.length === 1 ? 'is' : 'are'} re-queued as full cars.` : ''}`,
  )
  log(`express batch: ${landed.length} landed, ${ejected.length} ejected, ${satisfied.length} already satisfied, ${claimedSatisfied.length} claimed-already-done (parked, unverified), ${redBatch.length} re-queued after a red CI, ${(expressResult?.refused || []).length} refused`)
}

// A poisoned main stops everything behind it, exactly as a full car's would.
const expressHalted = expressResult?.post_merge_green === false
if (expressHalted) {
  await ledger(`${RUN}: **HALTED.** Main went red after the express batch merged. Revert or roll forward before anything else lands — that is a human decision.`)
  log('express batch poisoned main -- halting the train')
}

// -- Drain ---------------------------------------------------------
// Serial on purpose: each car is cut from main as it stands after the previous merge,
// which is the entire conflict strategy. No pipeline() here -- overlap is the one thing
// a train must never do.
const parkedSet = new Set()
// A red main after the express batch stops the train before any full car is cut, for the
// same reason a red main after a car does: everything behind it would be built on a tip
// nobody has decided what to do about.
let halted = expressHalted
const releaseNotes = []
const landedSoFar = () => cars.filter((c) => c.state === 'landed').length

for (const n of dispatchList) {
  const car = byNumber.get(n)
  const finish = async (state, reason, extra = {}) => {
    cars.push({ issue: n, title: car.title, state, reason, ...extra })
    const line = `car #${n} ${state}${reason ? ` -- ${reason}` : ''}${extra.merge_commit ? ` (merged as ${extra.merge_commit})` : ''}${extra.pr_url ? ` ${extra.pr_url}` : ''}`
    ledgerLines.push(line)
    // A parked car's reason lives nowhere else if this run is killed before the wrap-up --
    // and neither does an already-satisfied one's: it leaves a closed issue with no PR and
    // no run report, so without this comment nothing says why it closed.
    if (state === 'parked' || state === 'already-satisfied') await ledger(`${RUN}: ${line}`)
  }

  if (halted) {
    cars.push({ issue: n, title: car.title, state: 'not-dispatched', reason: 'the train halted on a poisoned tip' })
    continue
  }
  if (parkedSet.has(n)) {
    await finish('parked-by-dependency', 'a car it is blocked by did not land')
    continue
  }

  phase('Drain')

  // The dispatch probe: this car's own acceptance, run against the tip it is about to be
  // cut from. One spawn, no fragments, no worktree of its own beyond the shared detached
  // one. A pass means an earlier merge already did this work, and the ~31 minutes of gate,
  // CI and main-watch behind this dispatch would land nothing.
  //
  // The asymmetry below is the point. A satisfied car does NOT park its dependents: its work
  // is on main, so everything blocked on it is now unblocked. A car whose acceptance cannot
  // be run does park them -- that ticket failed its handover at filing and should not have
  // boarded, and nothing downstream of it can be trusted to know what "done" means.
  if (PROBES !== 'off') {
    const probe = await call(probePrompt(n, PROBES === 'on'), { label: `probe#${n}`, phase: 'Drain', schema: PROBE_SCHEMA })
    probedTip = true
    probeVerdicts.push({ number: n, ran: probe?.ran === true, passes: probe?.passes === true })
    if (probe?.passes === true) {
      log(`#${n}: its "## Verify" already passes on ${probe.tip_sha || 'the tip'} -- ${PROBES === 'on' ? 'closed, no car cut' : 'WOULD close it and cut no car (probes: shadow)'}`)
      if (PROBES === 'on') {
        await finish('already-satisfied', `its "## Verify" already passes on ${probe.tip_sha || 'main'} -- an earlier merge did this work${probe.excerpt ? `: ${probe.excerpt}` : ''}`)
        continue
      }
    } else if (probe?.ran === false) {
      log(`#${n}: its "## Verify" could not be run -- ${probe.excerpt || 'no detail'}${PROBES === 'on' ? '' : ' (probes: shadow, dispatching anyway)'}`)
      if (PROBES === 'on') {
        await finish('parked', `its "## Verify" could not be run: ${probe.excerpt || 'no detail'}. A ticket whose acceptance does not exist cannot be shown to be done -- fix the body and re-triage it.`)
        for (const d of dependentsOf(admitted, n)) parkedSet.add(d)
        continue
      }
    }
  }

  // No separate dispatch-time LABEL recheck: process-issues' own Triage re-reads state and
  // labels on every named issue and refuses a gated or closed one whatever rosterChecked
  // says -- so the mid-train human brake (re-applying the gate label) still works, and the
  // refusal surfaces below through the child's `skipped` reason. A dedicated 40k-token
  // agent per car re-read the same two fields the child was about to read anyway. The probe
  // above is not that agent: it reads something nobody else reads, at the only moment the
  // answer is worth anything.

  // One car, end to end: worktree, 13 phases, merge. process-issues re-fetches and cuts
  // its worktree at its own Triage, so the fresh tip needs no help from here.
  // The child hand-rolls that worktree rather than using `isolation: 'worktree'`: the tree
  // must be named, durable and readable by a human after a parked car, which the per-agent
  // ephemeral one is not.
  //
  // Three things this train knows that a standalone run does not:
  //   - watchMain: post_merge_green is the halt condition below, so it must be measured.
  //     Not in CAR_ARGS and not overridable -- a train that stops watching main stops being
  //     able to stop.
  //   - baseline: the first car measures the pre-change state of the tip. Every later car is
  //     cut from a main that the previous car's CI just proved green, so re-running the full
  //     check chain to rediscover that costs a container per car and learns nothing new. A
  //     halt on a red main is what invalidates this, and a halt ends the train anyway.
  //   - rosterChecked: blocking edges were verified at Roster; Triage still re-reads the
  //     labels and state itself, which is the dispatch-time brake.
  //   - requireReady: Triage's re-read must also demand READY_LABEL's continued PRESENCE
  //     -- the train's admission rule, and the human's mid-train brake. Standalone runs do
  //     not require it, so the train has to say so.
  //   - watchMain 'auto': the merge agent compares trees and only watches main's CI when
  //     the squash landed something the PR's CI did not test. The per-train flake net is
  //     the single main check in the wrap-up below.
  const r = await workflow('process-issues', {
    issues: [n],
    merge: true,
    runLabel: `${RUN} (train for #${SPEC})`,
    baseline: landedSoFar() === 0,
    rosterChecked: true,
    ...CAR_ARGS,
    // The machine this train is running on, passed down rather than re-guessed per script.
    repo: opt.repo,
    ghPath: opt.ghPath,
    // Already probed at dispatch, a few lines above -- the child must not pay for it twice.
    probes: 'off',
    // Same principle, applied to the other thing already measured. The roster opened this
    // ticket's files, found the symbols it changes and wrote down why the change is that
    // size. Thrown away, the car's plan agent rediscovers all of it from scratch: measured
    // 2026-08-31 on #278, 51 turns to re-derive a four-line Dockerfile edit. Passed down, it
    // starts from what is already known. It is a STARTING POINT, never a cap -- the plan
    // stage is free to find the roster wrong, and says so when it does.
    rosterSizing: byNumber.get(n)
      ? {
          expected_lines: byNumber.get(n).expected_lines || 0,
          sizing: byNumber.get(n).sizing || 'unknown',
          evidence: byNumber.get(n).sizing_evidence || '',
          files_to_edit: byNumber.get(n).files_to_edit || byNumber.get(n).files_named || [],
        }
      : null,
    // After the spread on purpose: neither is the operator's to relax on a train.
    requireReady: true,
    watchMain: 'auto',
  })
  let s = r?.summary?.[0]

  if (!s) {
    const skip = (r?.skipped || []).find((x) => x.number === n)
    await finish('parked', skip ? `process-issues refused it: ${skip.reason}` : 'the run returned no summary -- see its journal')
    for (const d of dependentsOf(admitted, n)) parkedSet.add(d)
    continue
  }

  // The child can reach the same verdict the dispatch probe does, by a different route: a
  // reproduction that will not go red while the issue's acceptance passes. On a train the
  // child is launched with probes:'off' so this normally cannot fire -- it is here so that
  // changing that flag does not silently record a satisfied car as a parked one.
  if (s.state === 'already-satisfied') {
    await finish('already-satisfied', s.notes || 'the car found this work already on main')
    continue
  }

  // A green, shipped PR that did not merge is the classic behind-the-tip conflict.
  // One recovery attempt, mechanized from the skill's derail table; a second failure parks.
  if (!s.merged && s.shipped && s.ci === 'green') {
    phase('Recover')
    const rec = await call(
      [
        `A PR for issue #${n} is open, its CI is green, but the merge was refused -- almost always because the branch is behind main. One recovery attempt, then stop either way. ${GH}`,
        `The worktree is ${s.worktree} on branch ${s.branch}.${s.pr_url ? ` The PR: ${s.pr_url}` : ''}`,
        `cd there, \`git fetch origin\`, \`git rebase origin/main\`.`,
        `If the rebase conflicts: read ${MAIN_REPO}/.claude/skills/resolving-merge-conflicts/SKILL.md and follow it on the live conflict. Resolve honestly or abort -- never take "ours" wholesale to make it stop.`,
        'Then push with `git push --force-with-lease` in the FOREGROUND, one Bash call, explicit timeout of 900000 ms (15 minutes) -- the pre-push hook runs the full container gate, ~5 minutes measured 2026-08-28 and longer on a cold cache or a loaded host, and the default timeout kills it mid-gate. Never --no-verify.',
        'Watch the PR checks to a conclusion (`gh pr checks <pr> --watch`), and if green: `gh pr merge <pr> --squash --delete-branch`.',
        'Then find the CI run the merge triggered on main (`gh run list --branch main --limit 3`), wait for it, and report post_merge_green.',
        // The child already returned "not merged", so its own teardown-on-success never ran.
        // If nothing does it here, a recovered car's stack and tree outlive the whole train.
        `Only if the merge SUCCEEDED: ${teardown(s.worktree)} Set torn_down to what you actually stopped and removed. If the merge was refused or anything went red, change nothing -- the worktree and its stack are what a human debugs.`,
        'If anything fails -- a second conflict, red checks, a refused merge -- STOP and report it. Do not retry, do not improvise.',
      ].join('\n'),
      {
        label: `recover#${n}`,
        phase: 'Recover',
        schema: {
          type: 'object',
          additionalProperties: false,
          required: ['merged', 'reason'],
          properties: {
            merged: { type: 'boolean' },
            merge_commit: { type: 'string' },
            post_merge_green: { type: 'boolean' },
            torn_down: { type: 'string', description: 'what you stopped and removed after a successful merge -- the compose stack, the worktree, or why either refused. Empty when the merge did not succeed.' },
            reason: { type: 'string', description: 'what happened, in one or two sentences' },
          },
        },
      },
    )
    if (rec?.merged) {
      // A teardown that refused is the one thing here worth saying out loud: it is silent,
      // it survives the run, and the next car pays for it at the gate rather than this one.
      log(`#${n}: recovered and merged -- ${rec.torn_down ? `cleaned up: ${rec.torn_down}` : 'NOTHING was torn down; check for a leftover compose stack and worktree'}`)
      s = { ...s, merged: true, post_merge_green: rec.post_merge_green, notes: `recovered: ${rec.reason}` }
    }
    else {
      await finish('parked', `merge refused and one recovery attempt failed: ${rec?.reason || 'no detail'}`, { pr_url: s.pr_url })
      for (const d of dependentsOf(admitted, n)) parkedSet.add(d)
      continue
    }
  }

  if (s.merged && s.post_merge_green === false) {
    // The one halt. Every remaining car would be cut from the poisoned tip, and the
    // revert-or-roll-forward decision belongs to a person -- the report names the command.
    halted = true
    await finish('halted-main-red', 'main went red after this merge. HUMAN: decide revert vs roll-forward before anything else lands.', {
      pr_url: s.pr_url,
    })
    // A halt is the one per-car event a human may be waiting on -- it posts now, not at wrap-up.
    await ledger(`${RUN}: **HALTED at car #${n}.** Main went red after its merge${s.pr_url ? ` (${s.pr_url})` : ''}. Revert or roll forward before anything else lands.`)
    continue
  }
  if (s.merged) {
    if (s.changelog_line) releaseNotes.push(`${s.release_impact}: ${s.changelog_line} (#${n})`)
    await finish('landed', s.needs_human ? 'landed, but the run flagged needs_human -- read its report' : '', {
      pr_url: s.pr_url,
      merge_commit: s.merge_commit,
      filed: s.filed,
    })
    continue
  }
  await finish('parked', `ci: ${s.ci}${s.needs_human ? ', needs_human' : ''} -- ${s.notes || 'see the run report'}`, { pr_url: s.pr_url })
  for (const d of dependentsOf(admitted, n)) parkedSet.add(d)
}

const landed = cars.filter((c) => c.state === 'landed')
// needs_human is a wide OR in process-issues, and it can fire on a car that nonetheless
// merged. The merge cannot be undone, so the train keeps going -- but "6 landed" with five of
// them flagged and the flags visible only per row reads as a clean run. Count them.
const flagged = landed.filter((c) => c.reason).length
const flagLine = flagged ? `, ${flagged} of them flagged needs_human -- read those cars' run reports` : ''
// The promotion evidence for the probe, and the only thing that makes shadow mode worth
// running: how often the predicate fired, and on what. Under 'shadow' nothing acted on it,
// so this line is the whole record.
// By ticket, not by probe run: a car ejected from the express batch is probed again when it
// is re-queued, and counting it twice inflates the very number this line exists to justify.
// Last verdict wins -- it was taken against the later tip.
const probeByTicket = [...new Map(probeVerdicts.map((v) => [v.number, v])).values()]
const probePassed = probeByTicket.filter((v) => v.passes)
const probeUnrunnable = probeByTicket.filter((v) => !v.ran)
const probeLine = !probeVerdicts.length
  ? ''
  : `probe (${PROBES}): ${probeByTicket.length} ticket(s) had their "## Verify" run against the tip; ` +
    `${probePassed.length ? `${probePassed.length} already passed (${probePassed.map((v) => '#' + v.number).join(', ')})` : 'none already passed'}` +
    `${probeUnrunnable.length ? `, ${probeUnrunnable.length} could not be run (${probeUnrunnable.map((v) => '#' + v.number).join(', ')})` : ''}` +
    `${PROBES === 'shadow' && (probePassed.length || probeUnrunnable.length) ? ' — reported only; every car dispatched anyway' : ''}.`
if (probeLine) log(probeLine)
log(`train done: ${landed.length} landed${flagLine}, ${cars.filter((c) => c.state.startsWith('parked')).length} parked${halted ? ', HALTED on red main' : ''}`)

// -- Wrap-up -------------------------------------------------------
// One agent: the train's single main flake-check, the ledger comment, and the report file.
// It runs even with report:false -- the comment is how a killed session's human catches up.
const REPORT_PATH = opt.reportPath || `${RUNS_DIR}/run-merge-train-${SPEC}.md`
{
  const body = [
    `# merge-train -- spec #${SPEC}: ${roster.spec_title}`,
    '',
    `${RUN}. ${landed.length} of ${dispatchList.length} dispatched car(s) landed${flagLine}.${halted ? ' **The train halted: main went red after a merge.**' : ''}`,
    '',
    probeLine ? `${probeLine}\n` : null,
    '## Cars',
    '',
    expressResult ? '## Express batch' : null,
    expressResult ? '' : null,
    expressResult
      ? `${(expressResult.landed || []).length} ticket(s) shared one branch, one CI run and one merge` +
        (expressResult.pr_url ? ` (${expressResult.pr_url})` : '') +
        `: ${(expressResult.landed || []).map((n) => '#' + n).join(', ') || 'none landed'}.` +
        ((expressResult.ejected || []).length
          ? ` Ejected as bigger than rostered and re-queued as full cars: ${expressResult.ejected.map((e) => `#${e.number} (${e.reason})`).join('; ')}.`
          : '')
      : null,
    expressResult ? '' : null,
    '| # | Ticket | State | PR | Why |',
    '| --- | --- | --- | --- | --- |',
    ...cars.map((c) => `| #${c.issue} | ${c.title} | ${c.state} | ${c.pr_url || ''} | ${c.reason || ''} |`),
    ...excluded.map((x) => `| #${x.number} | ${x.title} | not-boarded | | ${x.reason} |`),
    '',
    halted
      ? '## HALT\n\nMain is red after the last landed merge. Revert it (`git revert <merge_commit>` from the Cars table) or roll forward -- a human decision. Nothing else should land until main is green.'
      : null,
    releaseNotes.length ? `## Release notes\n\n${releaseNotes.map((l) => `- ${l}`).join('\n')}` : null,
    otherPrs.length
      ? `## Open PRs this train did not touch\n\n${otherPrs.map((p) => `- #${p.number} ${p.title}${p.url ? ` -- ${p.url}` : ''}`).join('\n')}\n\nThese were open before the train departed. Check each one against the merges above: it may now be moot, or conflict.`
      : null,
    '',
    '## Next',
    '',
    `- A relaunch of { spec: ${SPEC}, merge: true } resumes from GitHub state alone (closed child = landed, open PR = parked).`,
    '- Parked cars keep their PRs open and their worktrees in place; clear the derail by hand or re-triage the ticket.',
    `- Per-car detail lives in each car's own run report under ${RUNS_DIR}/ (process-issues-<n>.md).`,
  ]
    .filter((x) => x !== null)
    .join('\n')

  // Every car skipped its own main watch when the squash landed the exact tree its PR CI
  // proved -- so one check here is the whole flake net for the train. A halt already
  // measured main red and posted, so it needs neither.
  const checkMain = landed.length > 0 && !halted
  await call(
    [
      `You are wrapping up a merge train for spec issue #${SPEC}. Do these in order and nothing else. ${GH}`,
      checkMain
        ? [
            '1. Check main ONCE: `gh run list --workflow CI --branch main --event push --limit 1 --json databaseId,status,conclusion,url`',
            '   -- the CI workflow specifically, not whatever ran on main last. If that run is still in progress, wait for it',
            '   with `gh run watch <id> --interval 30` (one foreground Bash call, explicit 1500000 ms timeout -- do not hand-roll',
            '   a sleep loop). Report its conclusion under main_conclusion.',
          ].join('\n')
        : '1. Do NOT check main -- nothing landed, or the train already halted on a measured red. Set main_conclusion to "not-checked".',
      `2. Post ONE comment on issue #${SPEC}. The text is final -- do not rephrase or extend it, except for the one RED line rule below.`,
      `\`gh issue comment ${SPEC} --body "<the text below>"\``,
      '<<<COMMENT>>>',
      `${RUN}: train finished -- ${landed.length} of ${dispatchList.length} dispatched car(s) landed${flagLine}${halted ? '; **HALTED on a red main**' : ''}.`,
      ...ledgerLines.map((l) => `- ${l}`),
      ...(probeLine ? [`- ${probeLine}`] : []),
      '<<<END COMMENT>>>',
      checkMain
        ? "Then ONE mechanical addition to that comment, decided by string comparison, not by judgement: if main_conclusion from step 1 is exactly \"success\", append nothing. For ANY other value, append this line verbatim: '**Main is RED after this train** -- revert or roll forward before anything else lands.'"
        : '',
      DO_REPORT
        ? [
            `3. Write a file at ${MAIN_REPO}/${REPORT_PATH} -- create the directory if missing, overwrite if it exists.`,
            'Content: everything between the two markers, byte for byte; the markers are NOT part of the file. The one allowed',
            'addition: append a final line `Main CI after the train: <conclusion from step 1>`.',
            '<<<REPORT-BEGINS>>>',
            body,
            '<<<REPORT-ENDS>>>',
            'Then confirm the file exists.',
          ].join('\n')
        : '3. No report file this run.',
      // The probe cut a detached tree at the tip and this is the last thing that will look at
      // it. Unlike a car's worktree there is nothing in it to debug -- it holds no work, only
      // a checkout of main -- so it goes whether the train landed everything or halted.
      // A "## Verify" command is copied out of a ticket body and run as given, so one that
      // brings up a compose stack left it named after this directory, and nothing else stops it.
      probedTip ? `4. Remove the probe worktree at ${PROBE_TREE}, which this train created to run each car's acceptance check against the tip. ${teardown(PROBE_TREE)} Report what you stopped and removed under notes.` : '',
    ]
      .filter(Boolean)
      .join('\n'),
    {
      label: 'wrapup',
      phase: 'Report',
      // Transcription plus two string comparisons -- no judgement left in it, so it runs on
      // the cheapest model like ledger() does. It is the one haiku agent here that watches a
      // CI run; if that turns out to be beyond it, put the model back and keep the wording.
      model: 'haiku',
      effort: 'low',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['posted', 'main_conclusion'],
        properties: {
          posted: { type: 'boolean' },
          main_conclusion: { type: 'string', description: "GitHub's conclusion string, or 'not-checked'" },
          notes: { type: 'string' },
        },
      },
    },
  )
  if (DO_REPORT) log(`run report: ${REPORT_PATH}`)
}

return {
  spec: SPEC,
  spec_title: roster.spec_title,
  order: dispatchList,
  cars,
  excluded,
  halted,
  probes: PROBES,
  probe_verdicts: probeVerdicts,
  landed: landed.map((c) => c.issue),
  release_notes: releaseNotes,
  report: DO_REPORT ? REPORT_PATH : 'not written',
  needs_human: halted || cars.some((c) => c.state.startsWith('parked')) || landed.some((c) => c.reason),
  next_step: halted
    ? 'main is red -- revert or roll forward before anything else lands'
    : cars.some((c) => c.state.startsWith('parked'))
      ? `clear the parked cars (see the comments on #${SPEC}), then relaunch { spec: ${SPEC}, merge: true } to drain the rest`
      : order.length > dispatchList.length
        ? `maxCars stopped the train early -- relaunch { spec: ${SPEC}, merge: true } for the remaining ${order.length - dispatchList.length} car(s)`
        : `every admitted car landed. If the spec is complete, close #${SPEC}.`,
}
