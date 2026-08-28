export const meta = {
  name: 'process-issues',
  description: 'Open GitHub issues -> baseline -> plan -> red test -> implement -> dogfood -> review -> verify -> docs -> wheel gate -> file follow-ups -> draft PR -> CI green -> merge',
  whenToUse:
    'Pick up open issues end to end in a distributed Python package. One git worktree, one branch and one draft PR per issue. Checks and the wheel gate run in an ephemeral container mounted on the worktree, because the host virtualenv cannot collect the suite and the shared container is bound to the main repo. Anything any stage discovers along the way -- bug or enhancement -- is filed as a new issue behind a human triage gate, so the loop feeds itself without running away.',
  phases: [
    { title: 'Baseline', detail: 'record failing checks + the distribution facts from pyproject.toml', model: 'sonnet' },
    { title: 'Triage', detail: 'gh issue list, pick candidates, create a worktree per issue', model: 'sonnet' },
    { title: 'Plan', detail: 'root cause + ordered tasks (failing test first) + test strategy', model: 'opus' },
    { title: 'Implement', detail: 'one subagent per task; task 1 must prove the bug red', model: 'sonnet' },
    { title: 'Dogfood', detail: 'run the change for real and read the data it produced; refine once if it is ugly', model: 'sonnet' },
    { title: 'Review', detail: 'adversarial review of the worktree diff + packaging impact', model: 'opus' },
    { title: 'Verify', detail: 'apply review fixes, run lint/typecheck/tests until green (bounded)', model: 'sonnet' },
    { title: 'Docs', detail: 'tech writing: bring docs/ back in step with the change, site-ready', model: 'sonnet' },
    { title: 'Package', detail: 'build sdist+wheel, install in a clean venv, prove the fix ships', model: 'sonnet' },
    { title: 'Followups', detail: 'file every stage discovery as a deduped, human-gated issue', model: 'sonnet' },
    { title: 'Ship', detail: 'commit, push, open or update a draft PR with the release impact', model: 'sonnet' },
    { title: 'CI', detail: 'watch the PR run, fix what it finds, iterate until green, then merge', model: 'sonnet' },
    { title: 'Report', detail: 'write the run report to docs/agents/runs/ so it outlives this session', model: 'sonnet' },
  ],
}

// args: { limit?: 1, base?: 'main', labels?: 'bug', issues?: [12, 34], from?: <prepare-tickets result>, force?: false,
//         draft?: true, maxTasks?: 4, maxNewIssues?: 3, verifyRounds?: 3, baseline?: true,
//         packageGate?: 'auto', dogfood?: true, docs?: true, report?: true, reportPath?: 'docs/agents/runs/x.md',
//         gates?: 'shadow' | 'on' | 'always',
//         runLabel?: 'nightly', ship?: true, merge?: false, ciRounds?: 3 }
//
// This is the back half of a two-workflow loop. The prepare-tickets workflow does the research, design
// and spec work and files tickets behind the "needs-triage" gate; a human removes that label
// from the ones worth doing; this workflow implements them. `from` takes prepare-tickets' whole return
// value, so the handover is { from: <that object> } -- but a still-gated ticket is refused
// even when named, because a machine naming an issue is not a human approving it.
//
// `args` is an OBJECT, not prose. Passing a sentence leaves every option at its default,
// so the run triages its own issues and ignores whatever the sentence asked for.
//
// Before trusting it on the backlog, prove it on one issue you already understand:
//   1. `node .claude/workflows/dryrun.mjs process-issues` -- renders every prompt, spends nothing.
//   2. { issues: [<n>], maxNewIssues: 0, ship: false } -- a real run that files nothing and
//      pushes nothing; the finished work sits in the worktree for you to read.
//   3. { issues: [<n>], maxNewIssues: 1 } -- the full loop, one issue, one draft PR.
const opt = args || {}
// The handover from prepare-tickets: pass its whole return value as `from` and this run
// works the tickets it filed. `issues` still wins, so a hand-picked list overrides it.
const FROM = Array.isArray(opt.from?.tickets) && opt.from.tickets.length ? opt.from.tickets : null
const ONLY = Array.isArray(opt.issues) && opt.issues.length ? opt.issues : FROM
// A named list is a list, not a suggestion: it used to be silently truncated to LIMIT (1),
// so handing over five tickets quietly worked one and dropped four.
// One issue by default otherwise: two put ~22 agents in flight, and every check funnels
// through the same docker daemon anyway, so the second mostly queues behind the first.
const LIMIT = opt.limit || (ONLY ? ONLY.length : 1)
const BASE = opt.base || 'main'
const DRAFT = opt.draft !== false
const MAX_TASKS = opt.maxTasks || 4
const MAX_NEW_ISSUES = opt.maxNewIssues === 0 ? 0 : opt.maxNewIssues || 3
const VERIFY_ROUNDS = opt.verifyRounds || 3
const DO_BASELINE = opt.baseline !== false
// Set by an orchestrator that has already read these issues' labels, state and blocking
// edges. It removes a duplicated GitHub read at Triage; it never relaxes the label gate.
const ROSTER_CHECKED = opt.rosterChecked === true
// The tech-writing stage. On by default: docs written in a follow-up PR never get written.
const DO_DOCS = opt.docs !== false
// Reading the produced data, not just the code. On by default: the suite asserts one field
// of one record, so nothing else in this run ever looks at the whole plate.
const DO_DOGFOOD = opt.dogfood !== false
// The run report. The return value lands in one session's context and dies with it; the
// report is the copy still there tomorrow. Composed in plain JS from the same data the
// return value carries -- a report an agent writes from memory disagrees with the run.
const DO_REPORT = opt.report !== false
const PKG_GATE = opt.packageGate || 'auto' // 'auto' = skip only a provably test-only change | 'always' | 'never'
// Adaptive stage gating -- see docs/research/adaptive-workflow-gating.md.
// 'on' (the default) acts on the decision. 'shadow' decides and LOGS what it would have
// skipped, then runs it anyway; 'always' never adaptively skips. A nightly forces 'always':
// that is TIA's "run everything at a configured periodicity", with the caller standing in
// for the clock, because a workflow script has no access to one.
//
// The default was 'shadow' for exactly the right reason -- a gate nobody has watched
// decline a check is how a defect ships -- and then never moved, so every skip this policy
// has ever decided was logged and overruled. A shadow that is never promoted is not caution,
// it is the cost of the gate with none of the saving. The predicates are pinned by
// gates.test.mjs, each one names the class of bug it is willing to miss, and Verify, CI and
// the wheel gate all still run behind them. Promote it; keep 'shadow' for a new predicate.
const GATES = opt.runLabel === 'nightly' ? 'always' : opt.gates || 'on'
// Merging is the only step here that changes main. It is gated on the PR's CI going green
// under its own steam -- never on an agent's opinion that the code is fine.
//
// Opt IN, not out. It used to default on, so a run launched without `merge:false` merged to
// main on its own; the reviewer found out afterwards. A forgotten argument must not be able
// to write to main, and `merge:true` is one word for the times you do want it.
const DO_MERGE = opt.merge === true
// Watching main's CI after the merge costs about as long as the PR's own CI, and for a
// one-off run nothing acts on the answer: it lands in a summary field and a report nobody
// blocks on. It is load-bearing in a TRAIN, where post_merge_green === false is the halt
// condition that stops the next car being cut from a poisoned tip -- so run-merge-train
// asks for it explicitly and cannot be talked out of it.
const WATCH_MAIN = opt.watchMain === true
const CI_ROUNDS = opt.ciRounds || 3
// The one escape hatch from the needs-triage gate on a named list. It exists so a human
// can say "yes, I know, work it anyway" -- never so an orchestrator can pass the gate on
// its own. An orchestrator that sets this has removed the only human in the loop.
const FORCE = opt.force === true
// Date.now()/new Date() throw inside a workflow (the journal must stay deterministic),
// so run provenance in issue bodies comes from args, not the clock.
const RUN = opt.runLabel || 'an automated process-issues run'

// The gate that keeps the self-filing loop from feeding itself. docs/agents/triage-labels.md
// owns the rules; what follows is why the constants live here and what they mean to THIS
// workflow.
//
// PROVENANCE DECIDES THE DEFAULT. The property GATE_LABEL protects is that an agent must
// not feed itself work it invented -- a claim about where the work came from, not about
// who types the label. This comment used to say the opposite ("what a HUMAN applies after
// triaging"), and read literally it made every ticket of every request a chore for one
// person to clear one at a time. A chore gets cleared without being read, which is worse
// than no gate at all.
//
// So the line is drawn at origin:
//   - Work tracing to a request issue a person filed  -> prepare-tickets files it ready,
//     gated only for the four reasons in triage-labels.md.
//   - Work THIS workflow found for itself (Followups, below) -> FOUND_LABEL + GATE_LABEL,
//     always, no exceptions. That is the self-feeding loop, and it never opens.
// Triage still refuses anything carrying GATE_LABEL, whichever way it got there.
const FOUND_LABEL = 'agent-found'
const GATE_LABEL = 'needs-triage'
// Mutually exclusive with GATE_LABEL. An issue carrying both is un-boardable -- every
// reader checks the gate label first -- so prepare-tickets re-reads what it filed and
// repairs the pair rather than trusting a prompt not to create it.
const READY_LABEL = 'ready-for-agent'

// .claude/ is gitignored, so skills and graphify-out do NOT exist inside a worktree.
// Anything under here is read from the main checkout, by absolute path.
const MAIN_REPO = 'C:/Users/User/Documents/dev/graphknows'
// Where an agent's own output goes. One directory, one ignore rule, and it sits beside the
// rules agents READ (definition-of-done.md, triage-labels.md) rather than off in a scratch
// tree of its own -- so "what does an agent read here, and what did it write" is one folder.
// docs/agents/runs/ is gitignored; the files beside it are not, deliberately: a rule an
// agent cannot open from a fresh clone is a rule that gets restated into every ticket
// instead, which is the bloat this whole pipeline was rebuilt to stop.
const RUNS_DIR = 'docs/agents/runs'

// Docker on Windows needs a drive-letter path; agents tend to report the Git Bash form.
const winPath = (p) =>
  p
    ? String(p)
        .replace(/\\/g, '/')
        .replace(/^\/([a-zA-Z])\//, (_, drive) => `${drive.toUpperCase()}:/`)
    : '<ABSOLUTE PATH OF YOUR WORKTREE, drive-letter form, e.g. C:/Users/User/Documents/dev/gk-baseline>'

// Every agent in this run is pinned, rather than inheriting whatever the launching
// session happened to be set to: a workflow that reads differently depending on who
// started it is not reproducible, and the journal would not say why.
const RUN_ON = { model: 'sonnet', effort: 'medium' }
// The two stages where a wrong answer is expensive and invisible. Planning picks the
// root cause -- get that wrong and every stage after it implements the wrong fix
// perfectly. Review is the only adversarial read of the diff; a reviewer that misses a
// defect makes the whole loop confidently wrong. Everything else is mechanical: run a
// command, read its output, fill a schema.
const DEEP = { model: 'opus' }
const call = (prompt, opts = {}) => agent(prompt, { ...RUN_ON, ...opts })

// gh landed on the machine PATH at install time; a session started before that
// still carries the stale PATH, so every gh-touching prompt gets the fallback.
const GH = 'Use `gh`. If it is not on PATH, fall back to "/c/Program Files/GitHub CLI/gh.exe" (quote it).'

// Every push in this run goes through the same hook, so it gets described once. Ship used to
// say "~5 minutes, allow 10" and CI "around 25 minutes" for the identical hook: an agent that
// believed the low number killed the gate mid-run, one that believed the high number idled.
//
// The "~3-4 minutes" this said before was stale, and it is what parked #137 and #160: the real
// cost was ~12 minutes against their 10-minute budget, so they did all the work and pushed
// nothing. Measured on 2026-08-28 after the frame-index fix, worktree (ephemeral runner, the
// slow path): pytest 1271 tests in 119s, whole gate ~5 minutes. 900000 is that with headroom
// for a cold model cache and a loaded host. Re-measure before lowering it.
const PREPUSH = [
  'Run it in the FOREGROUND, in ONE Bash call, with an explicit timeout of 900000 ms (15 minutes).',
  'Do NOT background it and do not poll for it: a backgrounded push cannot notify you before your turn ends,',
  'so you would wait forever and finish without reporting. One blocking call that returns is what you want.',
  'The pre-push hook runs the full container gate (lint, types, the whole suite, secret scan, dependency audit)',
  '-- around 5 minutes, longer on a cold model cache or a loaded host -- so the DEFAULT 2-minute timeout kills',
  'it mid-gate every time and you must pass the timeout explicitly. A slow push is NOT an auth failure and NOT',
  'a hang: do not retry with a different remote, do not re-authenticate, and NEVER add --no-verify. Run it once.',
  'If it is killed anyway, the gate now names and traps its own runner, so check `docker ps` for a `gk-gate-*`',
  'survivor and remove it before anything else -- a retry that races an orphan starves both.',
].join(' ')

const HOUSE =
  'Repo rules (CLAUDE.md) are binding: pythonic OOP, type hints, one thing per function, ' +
  'small diffs, no speculative abstraction. Pre-commit hooks are rules -- never pass --no-verify. ' +
  'NEVER weaken, delete or retarget an existing test to make the suite pass. If an existing test genuinely ' +
  'encodes the wrong behaviour, leave it failing and report it -- changing it silently is the one unforgivable move here. ' +
  // The pipeline pushes every agent toward a green check, so the rule that a green check is not the
  // goal has to be stated where that pressure is applied. #137 shipped `nltk.download()` into library
  // code because the corpus was missing from the image: the tests went green, the criterion was met,
  // and a package strangers pip-install now reaches the network mid-ingest. The repo already had
  // scripts/bake_models.py for exactly that corpus, and an open ticket (#208) to wire it up.
  'A check can fail because the ENVIRONMENT is broken -- a missing corpus, model, service, credential or ' +
  'volume -- and that is not a defect in the code under test. Fix it where it is provisioned: the Dockerfile, ' +
  'docker-compose.yaml, a setup script, a Makefile target. Library code under graphknows/ never downloads a ' +
  'corpus or model, provisions a service, or reaches the network to make a check pass. If you cannot fix the ' +
  'environment from inside this worktree, STOP: leave the check failing, say exactly what is missing and where ' +
  'it should be provisioned, and let the car park. A parked car costs one re-run; a workaround shipped to PyPI ' +
  'costs every user who installs it offline. Before writing any such fallback, grep scripts/ and the Makefile -- ' +
  'the provisioning step usually already exists and is simply not being run.'

// This repo is not an application: it is a distribution installed by strangers.
// Everything below is a failure mode that a green local test suite cannot see.
const PACKAGING = [
  'THIS REPO IS A DISTRIBUTED PYTHON PACKAGE. These rules are binding:',
  '  - NEVER change the version: not `[project] version` in pyproject.toml, not a single-sourced `__version__`, not a tag it is derived from. Versions are bumped once, at release. A version already published to PyPI can never be re-uploaded or replaced, so a stray bump in a fix branch burns a real version number.',
  '  - Every third-party module imported by package code must be declared in `[project.dependencies]` -- or in the right `[project.optional-dependencies]` extra if the feature is optional -- with a lower bound. An undeclared import installs fine for you and ImportErrors for every user.',
  '  - Test-only and dev-only imports go in the dev dependency group / extra, NEVER in `[project.dependencies]`.',
  '  - Do not use a stdlib API newer than `requires-python` permits, and do not change `requires-python` to make one legal.',
  '  - A new import package directory needs an `__init__.py`; without one it is a namespace package and may silently not ship.',
  '  - Non-.py files (data, ontologies, `py.typed`) do NOT ship automatically. They must be declared to the build backend or the wheel is missing them.',
  '  - This repo is FLAT layout: tests import the in-tree copy, so a wheel missing a file still tests green -- only the package gate catches it. Code that only works because of the current working directory works in the suite and breaks for installed users.',
  '  - Never add `dist/`, `build/` or `*.egg-info/` to the repository.',
].join('\n')


const EVIDENCE = [
  'Evidence bar -- two lanes, both strict:',
  '  BUG: report it only if you can state a concrete failure scenario -- real inputs producing a wrong result, a crash, a regression, or a broken contract.',
  '  ENHANCEMENT: report it only if you can name something a user cannot do today, or must do by hand, and why that matters now.',
  'Neither lane accepts style nits, speculative hardening, "could be more general", "add more tests", or a refactor wish. Those are not findings.',
].join('\n')

// The skills this repo already carries, by absolute path: a workflow subagent may not have
// the Skill tool, but it always has Read. Only the ones that pay for their tokens are wired
// in -- code-review genuinely spawns its own sub-agents (nested orchestration), while
// deep-code-review's interview loop needs a user to answer it, as do to-tickets / to-spec
// (both now carry agent-mode branches for drafting; deep-code-review's metrics scoreboard
// is wired into GRAPHIFY below without the interview).
const skill = (path, why) => `Read ${MAIN_REPO}/.claude/skills/${path} and follow it: ${why}`

// An orientation aid, not truth: the graph predates the last dozen commits and the package
// move, and graphify-out is gitignored so it exists only in the main checkout.
const GRAPHIFY = [
  `Orientation: this repo has a knowledge graph. Before grepping blind, run these from ${MAIN_REPO} (NOT your worktree -- graphify-out is gitignored):`,
  '  `graphify query "<your question>"`   -- broad context, BFS over the graph',
  '  `graphify path "A" "B"`              -- how two concepts connect',
  '  `graphify explain "<symbol>"`        -- plain-language explanation of a node',
  `  \`python ${MAIN_REPO}/.claude/skills/deep-code-review/metrics.py ${MAIN_REPO}/graphify-out/graph.json\`  -- risk scoreboard: hotspots (churn x fan-in), hidden coupling (files that co-change with your target but share no import edge -- callers grep cannot find), cycles, god nodes.`,
  'Use hidden-coupling rows to widen a root-cause search past the file the issue names, and blast-radius rows when assigning disjoint files to tasks.',
  'It was built 2026-08-09 and the package has moved since: use it to find where a thing lives and what touches it, then confirm against the actual file before acting on it. If the CLI or graph is missing, skip it and read the code.',
].join('\n')

// The checks CANNOT run on the host: its virtualenv is incomplete and host pytest dies with
// 77 collection errors unrelated to any change. They also cannot run via `scripts/gate.sh`
// from a worktree: that script execs into the long-lived graphknows-workspace container,
// whose /app is a bind mount of the MAIN repo -- it would lint and test the main working
// tree and report the verdict as the worktree's. The only honest check is an ephemeral
// container mounted on the worktree itself.
const checksIn = (worktree) =>
  [
    "Run this repo's checks INSIDE a container mounted on your worktree. Never on the host (its virtualenv cannot",
    'even collect the suite) and never via `scripts/gate.sh` from a worktree (that script execs into the shared',
    'container, whose /app is the MAIN repo, not yours -- it would test the wrong code and report it as yours).',
    'One command, from anywhere:',
    '```',
    `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${winPath(worktree)}:/app" -w /app graphknows-workspace \\`,
    "  sh -c 'ruff check --no-fix graphknows evaluation tests \\",
    '   && ruff format --check graphknows evaluation tests \\',
    '   && mypy graphknows --strict --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores \\',
    '   && lint-imports \\',
    '   && python -m pytest -q -m "not integration"\'',
    '```',
    'Both env vars are load-bearing: MSYS_NO_PATHCONV stops Git Bash mangling the mount path, GRAPHKNOWS_SKIP_PREFLIGHT',
    'stops the image entrypoint aborting the run. While iterating, run one stage at a time (drop the `&&` tail, swap in',
    'a single test path); run the whole chain before you claim green.',
    'Integration tests need a live ArcadeDB -- `-m "not integration"` deselects them, leave it that way.',
    'If docker is unavailable, STOP and report it (ran=false / green=false). Never fall back to host pytest, and never',
    'report a check you did not actually run.',
  ].join('\n')

// Dogfood runs code rather than checks, so it gets the mount incantation without the
// check chain: the same two load-bearing env vars, a shell to work in.
const shellIn = (worktree) =>
  [
    'Run it INSIDE a container mounted on your worktree -- the host virtualenv is incomplete and cannot even import the package. Open a shell there:',
    '```',
    `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${winPath(worktree)}:/app" -w /app graphknows-workspace sh`,
    '```',
    'Both env vars are load-bearing: MSYS_NO_PATHCONV stops Git Bash mangling the mount path, GRAPHKNOWS_SKIP_PREFLIGHT stops the image entrypoint aborting the run.',
    'If docker is unavailable, say so and stop -- never fall back to the host, and never report output you did not actually produce.',
  ].join('\n')

// Nothing is committed until Ship, so `git diff origin/BASE...HEAD` is empty at review
// time: HEAD is still the base commit. The change lives in the working tree.
const diffHowTo = (issue) =>
  [
    `cd ${issue.worktree}.`,
    'The change is UNCOMMITTED -- read the working tree, not a commit range:',
    '  1. `git add -A -N`   (intent-to-add, so new files show up in the diff)',
    `  2. \`git diff origin/${BASE}\`  and  \`git status --short\``,
  ].join('\n')

const BASELINE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ran', 'commands', 'failing', 'notes'],
  properties: {
    ran: { type: 'boolean', description: 'false if the checks could not be executed at all' },
    commands: { type: 'array', items: { type: 'string' } },
    failing: {
      type: 'array',
      description: 'checks that already fail on the base branch, before any of our work',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'summary'],
        properties: {
          id: { type: 'string', description: 'pytest node id (path::test_name), or the lint/typecheck rule + file' },
          summary: { type: 'string' },
        },
      },
    },
    // Read once, used by every later agent. These facts decide what "correct" means
    // for a change: what ships, what may be imported, which Pythons must keep working.
    distribution: {
      type: 'object',
      additionalProperties: false,
      properties: {
        name: { type: 'string', description: 'the [project] name -- the distribution name on PyPI' },
        import_name: { type: 'string', description: 'the top-level import package, if it differs' },
        version: { type: 'string' },
        version_is_dynamic: { type: 'boolean', description: 'true if [project] declares dynamic = ["version"]' },
        version_source: { type: 'string', description: 'file or mechanism the version comes from' },
        build_backend: { type: 'string' },
        layout: { type: 'string', enum: ['src', 'flat', 'unknown'] },
        requires_python: { type: 'string' },
        dependencies: { type: 'array', items: { type: 'string' } },
        extras: { type: 'array', items: { type: 'string' }, description: 'optional-dependency group names' },
        ships_data_files: { type: 'boolean', description: 'true if non-.py files are declared to the backend' },
      },
    },
    notes: { type: 'string' },
  },
}

const ISSUES_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['issues'],
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'title', 'summary', 'worktree', 'branch'],
        properties: {
          number: { type: 'integer' },
          title: { type: 'string' },
          summary: { type: 'string', description: 'what the issue actually asks for, 1-3 sentences' },
          worktree: {
            type: 'string',
            description:
              'absolute path of the created worktree in DRIVE-LETTER form (C:/Users/.../gk-issue-<n>), not the Git Bash /c/... form -- later stages mount this path into docker',
          },
          branch: { type: 'string' },
        },
      },
    },
    skipped: {
      type: 'array',
      description: 'open issues deliberately not picked up, with the reason',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'reason'],
        properties: { number: { type: 'integer' }, reason: { type: 'string' } },
      },
    },
  },
}

const PLAN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['root_cause', 'tasks', 'test_strategy'],
  properties: {
    root_cause: { type: 'string', description: 'the actual defect, not the reported symptom' },
    tasks: {
      type: 'array',
      description: 'ordered, each independently executable by a fresh agent; exactly one has kind="test"',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'kind', 'instruction', 'files'],
        properties: {
          title: { type: 'string' },
          kind: {
            type: 'string',
            enum: ['test', 'code'],
            description: 'test = writes the reproduction that must fail before the fix; code = the fix itself',
          },
          instruction: { type: 'string', description: 'self-contained: the executing agent has no other context' },
          files: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    test_strategy: { type: 'string', description: 'the test that fails before the fix and passes after' },
    api_surface: {
      type: 'string',
      description: 'which public, importable names this fix changes -- empty if the change is entirely internal',
    },
    new_dependencies: {
      type: 'array',
      description: 'third-party packages the fix will need that are not already declared, with the table they belong in',
      items: { type: 'string' },
    },
    out_of_scope: {
      type: 'array',
      description: 'defects you noticed while reading that this fix will NOT address -- follow-up issue candidates',
      items: { type: 'string' },
    },
  },
}

// Every stage's channel for "I found something nobody knew about". One shape, so the single
// filing agent at the end of the pipeline can treat them all alike. A stage that consumes
// context and never reports back is how a known bug stays unknown.
const DISCOVERIES = {
  type: 'array',
  description:
    'anything previously unknown you hit while working: a bug, a broken assumption, a problem, or a concrete enhancement the code is missing. NOT the thing you were asked to fix, and not a style preference. Empty is a perfectly good answer.',
  items: {
    type: 'object',
    additionalProperties: false,
    required: ['kind', 'summary', 'evidence'],
    properties: {
      kind: { type: 'string', enum: ['bug', 'enhancement'] },
      summary: { type: 'string', description: 'symptom first, specific enough for someone to search for it later' },
      where: { type: 'string', description: 'file:line, or the module' },
      evidence: {
        type: 'string',
        description:
          'bug: concrete inputs -> wrong output, or the command you ran and what it printed. enhancement: what a user cannot do today, and why it matters now.',
      },
    },
  },
}

// The reproduction test is written by a different agent than the fix, and must be
// observed failing before any production code moves. A test that never went red
// proves nothing about the code that later makes it green.
const RED_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['test_id', 'command', 'fails_now', 'failure_excerpt'],
  properties: {
    test_id: { type: 'string', description: 'pytest node id of the new/extended test' },
    command: { type: 'string', description: 'the exact command that runs just this test' },
    fails_now: { type: 'boolean', description: 'true only if you ran it and it FAILED on its assertion' },
    failure_excerpt: { type: 'string', description: 'last ~15 lines of the actual failure output' },
    discoveries: DISCOVERIES,
    note: { type: 'string' },
  },
}

// Code tasks used to return free text, so the agent with the deepest view of the code had
// nowhere to put anything it stumbled over. It has somewhere now.
const IMPL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary'],
  properties: {
    summary: { type: 'string', description: 'what you changed, in a sentence or two' },
    files: { type: 'array', items: { type: 'string' } },
    discoveries: DISCOVERIES,
  },
}

const DOGFOOD_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ran', 'verdict', 'smells', 'notes'],
  properties: {
    ran: { type: 'boolean', description: 'false when the change produces no data anyone reads, or the data could not be produced at all' },
    artifact: { type: 'string', description: 'what was cooked and the exact command that cooked it' },
    records_read: { type: 'integer', description: 'how many real records were actually read, field by field' },
    verdict: { type: 'string', enum: ['edible', 'needs-work', 'structurally-wrong', 'not-applicable'] },
    smells: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['smell', 'evidence', 'caused_by_this_change'],
        properties: {
          smell: { type: 'string' },
          evidence: { type: 'string', description: 'a real record, quoted -- not a description of one' },
          caused_by_this_change: { type: 'boolean' },
        },
      },
    },
    refined: { type: 'array', items: { type: 'string' }, description: 'what was changed in this pass to fix a smell, and what the data looked like afterwards' },
    still_wrong: { type: 'array', items: { type: 'string' }, description: 'smells this change caused that one refinement pass did not fix' },
    discoveries: DISCOVERIES,
    notes: { type: 'string' },
  },
}

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['resolves_issue', 'findings', 'packaging'],
  properties: {
    resolves_issue: { type: 'boolean', description: 'judged from the diff itself, not from the claimed root cause' },
    verdict: { type: 'string', description: 'one paragraph justifying resolves_issue' },
    findings: {
      type: 'array',
      description: 'defects introduced or left behind BY THIS DIFF',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'severity', 'summary', 'failure_scenario'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          summary: { type: 'string' },
          failure_scenario: { type: 'string', description: 'concrete inputs -> wrong output' },
        },
      },
    },
    scope_violations: {
      type: 'array',
      description: 'changes outside what the plan called for: drive-by refactors, unrelated files, weakened existing tests',
      items: { type: 'string' },
    },
    // Everything a green test suite structurally cannot see about a distribution.
    packaging: {
      type: 'object',
      additionalProperties: false,
      // Only what a build CANNOT answer. The reviewer used to also predict undeclared
      // dependencies, unshipped files and requires-python violations from reading a diff --
      // all three of which the wheel gate then MEASURES for real, with `zipfile -l` and a
      // clean-venv install. A guess feeding a gate that proves the same thing is one stage
      // too many; the gate owns those now, and fixes what it finds.
      required: ['release_impact', 'changelog_line'],
      properties: {
        version_touched: { type: 'boolean', description: 'true if the diff changes the version anywhere -- always a blocker' },
        public_api_change: { type: 'string', description: 'public names added/removed/re-signatured; empty if none' },
        release_impact: {
          type: 'string',
          enum: ['none', 'patch', 'minor', 'major'],
          description: 'patch = backwards-compatible bug fix, minor = backwards-compatible addition, major = incompatible API change',
        },
        changelog_line: { type: 'string', description: 'one user-facing line for the changelog; empty if release_impact is none' },
      },
    },
    out_of_scope_defects: {
      type: 'array',
      description: 'real defects in the surrounding code NOT caused by this diff -- candidates for their own issue',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['summary', 'failure_scenario'],
        properties: {
          summary: { type: 'string' },
          file: { type: 'string' },
          failure_scenario: { type: 'string' },
        },
      },
    },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['green', 'checks_run', 'notes'],
  properties: {
    green: { type: 'boolean', description: 'true only if lint, typecheck and the unit suite all passed' },
    red_to_green: { type: 'boolean', description: 'the reproduction test that failed before the fix now passes' },
    checks_run: { type: 'array', items: { type: 'string' } },
    // The only MEASUREMENT of what this work touched. Everything before this stage works
    // from the plan's prediction, and an implementer routinely touches a file the plan
    // never named. Docs and the wheel gate both run after this and classify from it.
    changed_files: {
      type: 'array',
      items: { type: 'string' },
      description:
        'every path this work touched, repo-relative with forward slashes, from `git add -A -N && git diff --name-only origin/' +
        BASE +
        '`',
    },
    fixed: { type: 'array', items: { type: 'string' } },
    unresolved: { type: 'array', items: { type: 'string' } },
    new_failures: {
      type: 'array',
      description: 'failures NOT present in the recorded baseline -- i.e. caused by this work',
      items: { type: 'string' },
    },
    metadata_changes: {
      type: 'array',
      description: 'edits made to pyproject.toml, e.g. a dependency declared for a new import',
      items: { type: 'string' },
    },
    needs_human: { type: 'boolean', description: 'true if you are stuck or had to leave something unsafe' },
    notes: { type: 'string' },
  },
}

// A green suite proves the SOURCE TREE works. This proves the ARTIFACT works.
const DOCS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['user_visible', 'edited', 'notes'],
  properties: {
    user_visible: { type: 'boolean', description: 'whether this change alters anything a user of the package can observe' },
    edited: {
      type: 'array',
      description: 'the doc files actually changed, and what changed in each',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'change'],
        properties: { file: { type: 'string' }, change: { type: 'string' } },
      },
    },
    stale_found: {
      type: 'array',
      description: 'documentation that was ALREADY wrong before this change -- not yours to fix here, filed as its own issue',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'summary'],
        properties: { file: { type: 'string' }, summary: { type: 'string' } },
      },
    },
    broken_links: { type: 'array', items: { type: 'string' }, description: 'links that do not resolve, found while editing' },
    notes: { type: 'string' },
  },
}

const PACKAGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['built', 'notes'],
  properties: {
    built: { type: 'boolean', description: 'sdist and wheel were produced' },
    artifacts: { type: 'array', items: { type: 'string' } },
    twine_check: { type: 'boolean' },
    wheel_complete: { type: 'boolean', description: 'every file this change added is present inside the wheel' },
    installs_clean: { type: 'boolean', description: 'the wheel installed into a fresh venv with its declared deps' },
    imports_from_site_packages: {
      type: 'boolean',
      description: 'the installed package imported, and its __file__ resolved inside site-packages -- not the source tree',
    },
    repro_passes_installed: { type: 'boolean', description: 'the reproduction test passes against the installed package' },
    fixed: {
      type: 'array',
      description: 'packaging metadata you corrected and re-proved by rebuilding, e.g. a dependency declared for a new import',
      items: { type: 'string' },
    },
    worktree_clean: { type: 'boolean', description: 'no dist/, build/ or *.egg-info left behind in the worktree' },
    problems: { type: 'array', items: { type: 'string' } },
    discoveries: DISCOVERIES,
    notes: { type: 'string' },
  },
}

const FOLLOWUP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['filed', 'commented', 'dropped'],
  properties: {
    filed: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'title'],
        properties: { number: { type: 'integer' }, title: { type: 'string' }, url: { type: 'string' } },
      },
    },
    commented: {
      type: 'array',
      description: 'existing issues that already covered a candidate; evidence was added there instead',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'why'],
        properties: { number: { type: 'integer' }, why: { type: 'string' } },
      },
    },
    dropped: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['summary', 'why'],
        properties: { summary: { type: 'string' }, why: { type: 'string' } },
      },
    },
  },
}

const SHIP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['shipped', 'notes'],
  properties: {
    shipped: { type: 'boolean' },
    pr_url: { type: 'string' },
    branch: { type: 'string' },
    action: { type: 'string', enum: ['created', 'updated', 'skipped', 'failed'] },
    notes: { type: 'string' },
  },
}

// One agent per round does the whole loop body: wait for the run to settle, and if it
// failed, fix the cause and push. Splitting watch from fix would double the agent count
// for no gain -- the agent that read the log is the one best placed to fix it.
const CI_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['outcome', 'notes'],
  properties: {
    outcome: {
      type: 'string',
      enum: ['green', 'fixed-and-pushed', 'still-red', 'no-run', 'error'],
      description:
        'green = the run finished successfully. fixed-and-pushed = it failed, you fixed the cause and pushed, so a new run is starting. still-red = it failed and you could not fix it. no-run = no CI run ever appeared for this branch.',
    },
    run_id: { type: 'string' },
    run_url: { type: 'string' },
    conclusion: { type: 'string', description: "GitHub's own conclusion string: success, failure, cancelled, timed_out" },
    failing_jobs: { type: 'array', items: { type: 'string' }, description: 'job or step names that failed, with the error line' },
    fixed: { type: 'array', items: { type: 'string' }, description: 'what you changed to address them' },
    commit: { type: 'string', description: 'the sha you pushed, if you pushed' },
    discoveries: DISCOVERIES,
    notes: { type: 'string' },
  },
}

const MERGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['merged', 'notes'],
  properties: {
    merged: { type: 'boolean' },
    method: { type: 'string', enum: ['squash', 'merge', 'rebase', 'none'] },
    merge_commit: { type: 'string' },
    issue_closed: { type: 'boolean', description: 'the linked issue closed itself via "Closes #n"' },
    post_merge_run_url: { type: 'string', description: 'the CI run the merge triggered on main' },
    post_merge_green: { type: 'boolean', description: 'true only if that run finished successfully' },
    notes: { type: 'string' },
  },
}

// -- Baseline ------------------------------------------------------
// Run the checks once on an untouched base branch, and read the distribution
// metadata once. Without the first, "that failure is pre-existing" is an
// unfalsifiable claim; without the second, every agent re-guesses what ships.
phase('Baseline')
let baseline = null
if (DO_BASELINE) {
  baseline = await call(
    [
      'You are recording the pre-change state of this repository so later agents can tell a regression from a pre-existing failure, and know what this distribution actually is.',
      '`git fetch origin`, then from the repo root:',
      `  \`git worktree add ../gk-baseline --detach origin/${BASE}\`  (reuse it if it already exists, but \`git checkout origin/${BASE}\` inside it first)`,
      'cd there and do two things. Change nothing: no fixes, no edits, no commits.',
      '',
      'A. Run the checks ONCE.',
      checksIn(null),
      'Record every failure as a stable identifier: a pytest node id (path::test_name), or rule + file for lint/typecheck. If a check cannot run at all (missing tool, collection error), set ran=false and say exactly why -- do not guess.',
      '',
      'B. Read pyproject.toml and fill in "distribution": the [project] name and version (note if version is dynamic, and where it comes from), the [build-system] build-backend, whether the code is under src/ or flat, requires-python, the declared [project.dependencies], the names of any [project.optional-dependencies] extras, and whether any non-.py data files (or py.typed) are declared to the build backend.',
      'Report what the file says, not what you think it should say.',
    ].join('\n'),
    { label: 'baseline', schema: BASELINE_SCHEMA },
  )
  const n = baseline?.failing?.length || 0
  log(baseline?.ran ? `baseline: ${n} pre-existing failure(s) on ${BASE}` : `baseline could not run: ${baseline?.notes || 'unknown'}`)
  if (baseline?.distribution?.name) {
    log(`distribution: ${baseline.distribution.name} ${baseline.distribution.version || ''} (${baseline.distribution.layout || '?'} layout, ${baseline.distribution.build_backend || '?'})`)
  }
}

const BASELINE_NOTE = baseline?.ran
  ? baseline.failing.length
    ? `These checks ALREADY fail on ${BASE} and are not yours to fix or delete:\n${baseline.failing
        .map((f) => `  - ${f.id}: ${f.summary}`)
        .join('\n')}\nAnything failing that is NOT on this list was caused by this work.`
    : `${BASE} was fully green at the start of this run, so ANY failure you see is caused by this work.`
  : `No baseline was recorded. Do not claim a failure is pre-existing unless you prove it by checking out origin/${BASE} and reproducing it there.`

const d = baseline?.distribution
// The wheel gate imports the installed package by name. It used to hardcode `graphknows`
// three lines after DIST_FACTS supplied import_name -- which exists precisely because the
// import package can differ from the distribution name.
const IMPORT_NAME = d?.import_name || d?.name?.replace(/-/g, '_') || 'graphknows'
const DIST_FACTS = d
  ? [
      'Distribution facts, read from pyproject.toml at the start of this run:',
      `  name: ${d.name || '?'}${d.import_name ? ` (imported as ${d.import_name})` : ''}`,
      `  version: ${d.version || '?'}${d.version_is_dynamic ? ` (dynamic, from ${d.version_source || 'the backend'})` : ''} -- DO NOT CHANGE IT`,
      `  build backend: ${d.build_backend || '?'} | layout: ${d.layout || '?'} | requires-python: ${d.requires_python || '?'}`,
      `  declared dependencies: ${(d.dependencies || []).join(', ') || 'none'}`,
      `  extras: ${(d.extras || []).join(', ') || 'none'}`,
      `  non-.py files declared to the backend: ${d.ships_data_files ? 'yes' : 'no -- anything non-.py you add will NOT ship unless you declare it'}`,
    ].join('\n')
  : 'Distribution facts were not recorded this run: read pyproject.toml yourself before assuming anything about what ships, what is declared, or which Pythons are supported.'

// -- Triage --------------------------------------------------------
// One agent creates every worktree: concurrent `git worktree add` in the same
// repo races on the index lock, and one sequential agent is cheaper than a lock.
phase('Triage')
// Blocking edges come from prepare-tickets' tickets: it files them in dependency order
// with a "## Blocked by" section naming real issue numbers. Working a blocked ticket first
// means implementing against code that does not exist yet.
// ROSTER_CHECKED: an orchestrator that has already read this issue's labels, state and
// blocking edges says so, and Triage does not pay for the same `gh issue view` calls again.
// A train does exactly that -- once when it builds the roster, and again at the dispatch
// check immediately before this run -- and the second read is the more recent of the two.
// It is only ever set by a caller that measured; it is not a way to skip the label gate,
// which stands either way, below.
const BLOCKED = ROSTER_CHECKED
  ? 'The caller has already verified every "## Blocked by" edge on these issues, immediately before this run. Do not re-read the blockers.'
  : [
      'A ticket may declare its dependencies in a "## Blocked by" section listing other issue numbers.',
      'For every candidate that has one, check each blocker with `gh issue view <n> --json state,title`.',
      'If ANY blocker is still open, skip the candidate and give "blocked by #<n>" as the reason -- its',
      'blockers must land first or you would be implementing against code that does not exist yet.',
    ].join(' ')

const filter = ONLY
  ? [
      `Handle exactly these issues: ${ONLY.join(', ')}.`,
      FORCE
        ? `The operator passed force, so work them whatever labels they carry. Still report anything you skip and why.`
        : [
            `First check each one's labels: \`gh issue view <n> --json number,title,labels,body,state\`.`,
            `SKIP any issue still carrying "${GATE_LABEL}" and give that as the reason. That label means a machine`,
            'proposed the work and no human has agreed to it yet -- being named on the command line is not agreement,',
            'because the thing doing the naming may itself be a machine. A human removes the label; that is the gate.',
            'Also skip any issue that is already closed.',
          ].join(' '),
      BLOCKED,
    ].join('\n')
  : [
      `List open issues with \`gh issue list --search "is:open is:issue -label:\\"${GATE_LABEL}\\"" --limit 50\`` +
        (opt.labels ? ` and also require --label "${opt.labels}"` : '') +
        `. The -label filter is REQUIRED: issues still carrying "${GATE_LABEL}" were filed by an agent and have not been reviewed by a human yet.`,
      `Pick at most ${LIMIT} that are concretely fixable in code.`,
      'Skip anything that is a question, a discussion, a duplicate, blocked on a decision, or too vague to implement.',
      BLOCKED,
      'Also skip anything that is a release request or asks for a version bump -- this workflow does not release.',
      `Also skip any issue that already has work in flight: check \`gh pr list --state open --json number,headRefName,title\` and \`git ls-remote --heads origin "fix/issue-*"\`.`,
      'Report every skip under "skipped" with the reason.',
    ].join('\n')

const triage = await call(
  [
    `You are triaging open GitHub issues for this repository. ${GH}`,
    filter,
    "Read each candidate's full body and comments (`gh issue view <n> --comments`) before judging it.",
    'Then, for EACH picked issue, sequentially (never in parallel):',
    '  1. `git fetch origin`',
    `  2. from the repo root: \`git worktree add ../gk-issue-<n> -b fix/issue-<n> origin/${BASE}\``,
    '  3. record the ABSOLUTE path of that worktree in DRIVE-LETTER form -- `C:/Users/User/Documents/dev/gk-issue-<n>`, not `/c/Users/...`. Later stages mount that path into docker, which needs the drive letter.',
    'If a worktree or branch for an issue already exists locally, reuse it instead of failing.',
    "Do not read source files or plan any fix -- that is the next phase's job.",
  ].join('\n'),
  { label: 'triage', schema: ISSUES_SCHEMA },
)

const issues = (triage?.issues || []).slice(0, LIMIT)
for (const s of triage?.skipped || []) log(`skipped #${s.number}: ${s.reason}`)
if (!issues.length) {
  return { picked: [], skipped: triage?.skipped || [], baseline, note: 'no actionable open issues' }
}
log(`picked ${issues.length} issue(s): ${issues.map((i) => '#' + i.number).join(', ')}`)

// -- Change classification ------------------------------------------
// One derived class, one policy table, no router agent: every fact a router would need is
// already a field in a schema, and re-deriving them with an LLM adds cost, non-determinism
// and a new way to be wrong. The wheel gate is the precedent -- it used to ask opus whether
// a diff "touches packaging" and now answers it in four lines of JS.
//
// This may only ever REMOVE work when it is certain, and "no evidence" is not certainty:
// an unclassifiable change is the most expensive class, not the cheapest. Same safe fallback
// Nx uses when the lock file moves, and TIA uses on a commit it cannot reason about.
const norm = (f) => String(f).replace(/\\/g, '/').replace(/^\.\//, '')
const isTestFile = (f) => /^tests?\//.test(f)
const isDocFile = (f) => /^docs\//.test(f) || /\.md$/i.test(f)
const isMetaFile = (f) => /(^|\/)(pyproject|ruff)\.toml$/i.test(f)

// plan.tasks[].files is a PREDICTION; verify.changed_files is a MEASUREMENT. Union them,
// never replace one with the other -- an implementer who touched a file the plan never
// named must not be able to shrink the change by being absent from one of the two.
// pkg is the IMPORT package name -- the directory the wheel actually installs, which is not
// always the distribution name. It defaults the same way IMPORT_NAME does, so a run with no
// baseline still classifies rather than throwing.
const classify = (plan, verify, pkg = 'graphknows') => {
  const files = [...plan.tasks.flatMap((t) => t.files || []), ...(verify?.changed_files || [])].map(norm)
  const deps = (plan.new_dependencies || []).length > 0
  const api = Boolean((plan.api_surface || '').trim())
  const meta = files.some(isMetaFile)
  const shipped = files.some((f) => !isTestFile(f) && !isDocFile(f) && !isMetaFile(f))
  // Not "is it code" but "is it code that SHIPS": scripts/, .github/ and the Dockerfile are
  // all kind:'code' and none of them produce a record anyone reads.
  const touchesPackage = files.some((f) => f === `${pkg}` || f.startsWith(`${pkg}/`))
  const kind = !files.length
    ? 'unknown'
    : shipped || deps || meta
      ? 'code'
      : files.every(isDocFile)
        ? 'docs'
        : 'test-only'
  // An unclassifiable change is the most expensive class, not the cheapest -- so with no
  // file list at all, assume it touches the package rather than skipping on ignorance.
  return { kind, deps, api, meta, files, touchesPackage: touchesPackage || !files.length, measured: Boolean(verify?.changed_files?.length) }
}

// A stage RUNS unless this says otherwise, and every entry names the class of bug it is
// willing to miss. If you cannot name one, the stage is not skippable.
const SKIPPABLE = {
  // Misses: ugly data in a change that produces none. Cheap -- Review, Verify and CI all
  // still run behind it. The third clause is the expensive one to omit: dogfood reads the
  // DATA the code produces, so a change that touches no file inside the import package
  // cannot have changed any. Its own prompt opens by asking the agent to decide exactly
  // that and return ran=false -- a container, a shell and a full turn spent reaching a
  // conclusion the file list already carried.
  dogfood: (c) => c.kind === 'docs' || c.kind === 'test-only' || !c.touchesPackage,
  // Misses: a wheel packaging defect. This workflow deliberately never releases, so NOTHING
  // downstream ever builds a wheel: it is the one skip in the pipeline with no post-submit
  // safety net. Hence the extra clauses -- a new dependency, a pyproject edit or a public
  // API change each force it back on.
  package: (c) => c.kind === 'test-only' && !c.deps && !c.meta && !c.api,
  // Misses: undocumented user-visible behaviour. The JS gate is the smaller risk here; the
  // writer's own user_visible=false is the larger one, cross-checked at Followups.
  docs: (c) => c.kind === 'test-only' && !c.api,
}

// Whether the stage should run. In shadow mode it always returns true and only reports the
// decision it would have taken, so the policy can be watched before it is trusted.
const gate = (stage, c, issue) => {
  if (GATES === 'always' || !SKIPPABLE[stage](c)) return true
  const why = `${c.kind}${c.measured ? '' : ' (predicted from the plan; no measured file list)'}`
  log(`#${issue.number}: ${GATES === 'shadow' ? 'WOULD skip' : 'skipped'} ${stage} -- ${why}`)
  return GATES === 'shadow'
}

// -- Per-issue pipeline ---------------------------------------------
// pipeline, not parallel: issue B plans while issue A is already implementing.
// Each stage carries the whole context forward so late stages (Package, Ship)
// can still see the plan and the review without a barrier.
const results = await pipeline(
  issues,

  // Plan
  async (issue) => {
    const plan = await call(
      [
        `Issue #${issue.number}: ${issue.title}`,
        `What it asks for: ${issue.summary}`,
        `Work in the worktree at ${issue.worktree} -- cd there first; it is a full checkout of this repo on branch ${issue.branch}.`,
        'If the issue references a parent spec issue, READ IT FIRST (`gh issue view <n>`). Tickets filed by prepare-tickets carry the design decision in the parent, not in themselves: planning from the ticket alone re-decides a question that was already settled, and settled differently. The parent may open with a quoted banner recording that an adversarial reviewer called the design unsound, or that a fact it rests on was refuted -- that is not decoration, it is the thing to resolve before you plan around it.',
        'The issue text is the contract, and the parent spec is its context. Where the decision was architecturally significant the spec cites an ADR under docs/adr/ -- that IS in your worktree, so read it when the spec names one. Do not go looking for any other design document; there is none.',
        // Two paths reach this stage. A ticket from prepare-tickets carries the contract in
        // its body. An issue somebody filed by hand, or that this workflow found for itself,
        // carries a reporter's words in the body and the contract in an "## Agent Brief"
        // comment a person wrote at triage -- because editing a reporter's own issue body to
        // impose a shape on it is the wrong trade. Read both; the brief wins.
        `Check the comments as well (\`gh issue view ${issue.number} --comments\`). If one of them is an "## Agent Brief", THAT is the contract and the body is background: a person wrote it during triage precisely because the body was a raw report. It carries the same sections a pipeline ticket does -- acceptance criteria, Verify, Out of scope -- and where the two disagree, the brief is the later decision.`,
        'A "[NEEDS CLARIFICATION: ...]" marker anywhere in the body or the brief means the work is NOT settled. Do not resolve it yourself and do not plan around it: stop, and report it as the reason. Somebody left that marker rather than guessing, and guessing here throws that away.',
        'Read the code before planning. Find the ROOT CAUSE, not the symptom: grep every caller of what you are about to change, so the fix lands where all callers route through instead of patching the one path the issue names.',
        skill('diagnosing-bugs/SKILL.md', 'the root-cause discipline this repo uses -- build the feedback loop before theorising. You may not modify the tree, so follow its "Diagnosis for planning" branch: spec the loop, do not run it.'),
        'Read CLAUDE.md and the surrounding modules; match their idiom.',
        // Until now this reached exactly one stage, Review -- which meant "shortest working
        // diff" was a judgement passed on work already done, by an agent with no power to
        // un-write it. The cheapest place to not build something is the plan.
        skill('ponytail/SKILL.md', 'the ladder: does this need to exist at all, is it already in this codebase, does the stdlib do it, can it be one line. Climb it for every task before you write the task down, and stop at the first rung that holds. It shortens the SOLUTION, never the reading -- trace the whole flow first.'),
        'A plan is a budget. The smallest number of tasks that closes the issue is the correct number; a task you added because it seemed thorough is a file somebody has to review.',
        GRAPHIFY,
        `Produce the smallest plan that actually fixes it -- at most ${MAX_TASKS} tasks. Each task must be self-contained: the agent executing it starts with a blank context and sees only your instruction. Give tasks disjoint files where you can.`,
        // This used to read "EXACTLY ONE task must have kind='test'", unconditionally. On a
        // deletion or config ticket there is no behaviour to reproduce, so "fails before,
        // passes after" can only be satisfied by asserting that each deleted thing is
        // absent -- which is how a cleanup whose entire point was removal acquired a
        // 528-line test file transcribing its acceptance criteria one by one.
        'AT MOST ONE task has kind="test": the reproduction that fails on the current code and passes once the fix lands. It touches only test files. Every other task is kind="code".',
        'Include it when there is BEHAVIOUR to reproduce -- something the code does wrong now and will do right after. Omit it when there is not: a deletion, a dependency or build-config change, a rename, a docs edit. For those the acceptance is the existing suite still passing, and the later stages know a plan can arrive with no test task.',
        'A test that asserts a file, a flag or a section is ABSENT is not a reproduction, it is the acceptance criteria copied into Python. Never write one. If the only test you can think of restates the ticket, that is the signal to omit the test task, not to write it anyway.',
        'The test must assert the CORRECT behaviour as the issue describes it -- never the behaviour the buggy code currently produces.',
        DIST_FACTS,
        PACKAGING,
        'Fill in api_surface with the public importable names this fix changes (empty if purely internal), and new_dependencies with any third-party package the fix will need that is not already declared -- name the pyproject table it belongs in. Prefer a fix that needs no new dependency at all: every dependency you add is one every user must install.',
        'List under out_of_scope anything real you noticed while reading that this fix will NOT address -- a defect, a broken assumption, or a concrete enhancement the code is missing. Each one is deduped and filed as a GitHub issue at the end of this run, so be strict:\n' + EVIDENCE,
        'Change nothing yourself.',
        HOUSE,
      ].join('\n'),
      { label: `plan#${issue.number}`, phase: 'Plan', schema: PLAN_SCHEMA, ...DEEP },
    )
    return plan ? { issue, plan } : null
  },

  // Implement -- one subagent per task, sequential within an issue.
  // Sequential on purpose: tasks of one plan share a worktree and parallel
  // writers to the same tree collide. Parallelism lives ACROSS issues.
  async (ctx) => {
    if (!ctx) return null
    const { issue, plan } = ctx
    // Deterministic ordering in code, not in a prompt: the test runs first, and the
    // MAX_TASKS cap can therefore only ever drop code tasks, never the reproduction.
    const ordered = plan.tasks.filter((t) => t.kind === 'test').concat(plan.tasks.filter((t) => t.kind !== 'test'))
    const tasks = ordered.slice(0, MAX_TASKS)
    if (ordered.length > tasks.length) {
      log(`#${issue.number}: dropped ${ordered.length - tasks.length} task(s) over the ${MAX_TASKS} cap`)
    }

    let red = null
    const done = []
    const discoveries = []
    for (let i = 0; i < tasks.length; i++) {
      const t = tasks[i]
      const isTest = t.kind === 'test'
      const out = await call(
        [
          `cd ${issue.worktree} and work only there.`,
          `Context -- GitHub issue #${issue.number}: ${issue.title}`,
          `Root cause: ${plan.root_cause}`,
          `Your task (${i + 1}/${tasks.length}): ${t.title}`,
          t.instruction,
          `Expected files: ${t.files.join(', ') || 'discover them'}`,
          done.length ? `Already done by earlier tasks: ${done.map((x) => '- ' + x).join(' ')}` : '',
          isTest
            ? [
                'This is the REPRODUCTION task. Touch test files only -- no production code.',
                skill('tdd/SKILL.md', 'what makes a test worth keeping -- assert external behaviour, never an implementation detail. tests.md beside it says where tests go in this repo.'),
                'Then RUN it and report what happened, using the single-test form of the command below.',
                checksIn(issue.worktree),
                'It must FAIL, and fail on its own assertion -- not on an import error, a typo, a missing fixture or a collection error. If it fails for one of those reasons, fix your test and run it again.',
                'If it PASSES, your reproduction is wrong: set fails_now=false and explain in note. Do NOT rewrite the assertion to match whatever the current code returns -- that turns the bug into the spec.',
              ].join('\n')
            : [
                'Implement it fully. Do not commit. Do not touch anything outside this task -- no drive-by refactors, no unrelated files.',
                skill('ponytail/SKILL.md', 'the ladder and the rules: reuse what is already here, stdlib before a dependency, deletion over addition, the shortest diff that actually works. Never lazy about UNDERSTANDING -- read the whole flow, then write the least code.'),
                'If you need an import that is not already a declared dependency, add it to the correct pyproject.toml table in the SAME task, with a lower bound. Never leave an import undeclared.',
                PACKAGING,
              ].join('\n'),
          isTest ? '' : DIST_FACTS,
          GRAPHIFY,
          'You are the agent closest to the code. If you hit something previously unknown -- a bug you did not cause, a broken assumption, a problem, or a concrete enhancement this code is missing -- do NOT fix it and do NOT let it evaporate: record it under "discoveries". Everything recorded there is deduped and filed as a GitHub issue at the end of this run.\n' + EVIDENCE,
          HOUSE,
        ]
          .filter(Boolean)
          .join('\n'),
        isTest
          ? { label: `red#${issue.number}`, phase: 'Implement', schema: RED_SCHEMA }
          : { label: `impl#${issue.number}:${i + 1}`, phase: 'Implement', schema: IMPL_SCHEMA },
      )
      discoveries.push(...(out?.discoveries || []).map((x) => ({ source: `implement: ${t.title}`, ...x })))
      if (isTest) {
        red = out
        if (!out?.fails_now) log(`#${issue.number}: WARNING -- reproduction never went red (${out?.note || 'no detail'})`)
        done.push(`${t.title} -> added ${out?.test_id || 'a test'} (red: ${out?.fails_now === true})`)
      } else {
        done.push(`${t.title} -> ${out?.summary || 'no output'}`)
      }
    }
    if (discoveries.length) log(`#${issue.number}: ${discoveries.length} discovery(ies) while implementing`)
    return { ...ctx, red, done, discoveries }
  },

  // Dogfood -- eat what was cooked, before anyone reviews the recipe.
  // Every stage before this one reads CODE: the plan reads it, the implementer writes it,
  // the suite asserts one field of one record. None of them ever looks at the data the
  // change actually produces, which is where this system's bugs live -- a score that is
  // identical across every row passes every assertion written for it.
  async (ctx) => {
    if (!ctx) return null
    const { issue, plan, red } = ctx
    if (!DO_DOGFOOD) return { ...ctx, dogfood: null }
    // Dogfood runs before Verify, so it has only the plan's prediction to classify from.
    if (!gate('dogfood', classify(plan, null, IMPORT_NAME), issue)) return { ...ctx, dogfood: null }

    const dogfood = await call(
      [
        `The fix for issue #${issue.number} is written. Before anyone reviews the code, you are going to read the DATA it produces.`,
        `What the issue asked for: ${issue.summary}`,
        `Root cause that was fixed: ${plan.root_cause || 'see the diff'}`,
        diffHowTo(issue),
        skill('dogfood/SKILL.md', 'the four steps -- cook, plate, taste, refine -- and the smells to taste for. Follow it exactly.'),
        '',
        'FIRST DECIDE: does this change produce data anyone reads -- nodes, edges, embeddings, chunks, retrieval hits, a report, a payload, a log line a user acts on? A pure type fix, a docstring or a build-config change produces none. If it produces none, set ran=false, say why, and change nothing.',
        'If it does produce data, cook it for real. This repo has live entry points -- the `Memory` facade, the MCP server, the evaluation harness under evaluation/, the examples under examples/. Use whichever already exercises the path you changed, on real input.',
        shellIn(issue.worktree),
        'Integration paths need a live ArcadeDB. If the data you need cannot be produced without one, set ran=false and say so.',
        red?.test_id
          ? `The reproduction test ${red.test_id} exercises the changed path; it is a good starting point for a script, but it is NOT the plate -- it asserts one field, and you are reading the whole record.`
          : '',
        '',
        'The skill owns the tasting discipline -- evidence bar, one refine pass, when to stop; follow it. Workflow specifics: everything not caused by this change goes under discoveries, and ran=false pairs with the "not-applicable" verdict.',
        HOUSE,
        'Do not commit.',
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `dogfood#${issue.number}`, phase: 'Dogfood', schema: DOGFOOD_SCHEMA },
    )
    log(
      dogfood?.ran === false
        ? `#${issue.number}: dogfood skipped -- ${dogfood.notes || 'the change produces no readable data'}`
        : `#${issue.number}: dogfood ${dogfood?.verdict || 'no verdict'} -- ${(dogfood?.smells || []).length} smell(s), ${(dogfood?.refined || []).length} refined`,
    )
    return { ...ctx, dogfood }
  },

  // Review -- adversarial, on the working diff only
  async (ctx) => {
    if (!ctx) return null
    const { issue, plan, dogfood } = ctx
    const review = await call(
      [
        diffHowTo(issue),
        `Issue #${issue.number}: ${issue.title} -- ${issue.summary}`,
        'Judge FIRST, from the diff alone, whether it resolves that issue. Only then read the claim below, and treat it as a hypothesis to falsify rather than a summary to agree with.',
        dogfood?.ran
          ? `An agent ran this change for real and read ${dogfood.records_read || 0} of the records it produced. Verdict: ${dogfood.verdict}.${
              (dogfood.still_wrong || []).length ? ` Still wrong after one refinement pass: ${dogfood.still_wrong.join('; ')}.` : ''
            }${
              (dogfood.refined || []).length ? ` Refined during that pass: ${dogfood.refined.join('; ')} -- those edits are in the diff you are reading.` : ''
            } Treat that as evidence about the OUTPUT, not as a verdict on the code.`
          : 'Nobody read the data this change produces -- either it produces none, or it could not be produced. Weigh the diff on its own.',
        `Claimed root cause: ${plan.root_cause}`,
        `The plan expected to touch: ${plan.tasks.flatMap((t) => t.files).join(', ') || 'unspecified'}. Anything else the diff touches is a scope violation -- report it.`,
        'Specifically check whether any EXISTING test was weakened, deleted, retargeted, or had an assertion loosened. That is always a blocker.',
        EVIDENCE,
        skill('ponytail-review/SKILL.md', 'the over-engineering axis -- its finding taxonomy and examples: an abstraction with one caller, a reimplemented stdlib helper, config for a constant. Its own one-line format does not apply; your output is the schema below.'),
        skill('graph-metrics/SKILL.md', 'how to read the impact of the changed files -- what they can break, and what is architecturally central or expensive to change. Use it to judge blast radius and scope violations instead of eyeballing them; the graph predates recent commits, so confirm anything it suggests against the actual file.'),
        GRAPHIFY,
        '',
        'Then fill in "packaging" -- the release judgement a build cannot make for itself. A later gate builds the wheel and installs it in a clean venv, so do NOT try to predict what ships or what is declared; it measures that. Judge only these:',
        DIST_FACTS,
        '  - any change to the version anywhere -> version_touched = true (always a blocker).',
        '  - public_api_change: names added, removed or re-signatured on the public surface.',
        '  - release_impact: patch for a backwards-compatible bug fix, minor for a backwards-compatible addition, major for an incompatible API change. Judge it from the user\'s side, not the diff size.',
        '  - changelog_line: one user-facing line, written for someone who did not read this diff.',
        '',
        'Separately, list under out_of_scope_defects any real defect you noticed in the surrounding code that this diff did NOT cause and should NOT fix. Same evidence bar. These become their own issues later, so precision matters -- do not pad the list.',
        'Change nothing.',
      ].join('\n'),
      { label: `review#${issue.number}`, phase: 'Review', schema: FINDINGS_SCHEMA, ...DEEP },
    )
    return review ? { ...ctx, review } : null
  },

  // Verify -- apply review fixes, then run the repo's checks until green (bounded)
  async (ctx) => {
    if (!ctx) return null
    const { issue, red, review } = ctx
    const pk = review.packaging || {}
    const verify = await call(
      [
        `cd ${issue.worktree}.`,
        `Code review of the fix for issue #${issue.number} returned:`,
        `Resolves the issue: ${review.resolves_issue === true ? 'yes' : 'NO -- treat this as a blocker'}. ${review.verdict || ''}`,
        review.findings.length
          ? review.findings
              .map((f) => `- [${f.severity}] ${f.file}${f.line ? ':' + f.line : ''} -- ${f.summary} (${f.failure_scenario})`)
              .join('\n')
          : '- no findings',
        (review.scope_violations || []).length
          ? `Scope violations to revert unless they are load-bearing: ${review.scope_violations.join('; ')}`
          : '',
        pk.version_touched
          ? "PACKAGING BLOCKER -- the version was changed. Revert it: releases are not this workflow's job, and a version already on PyPI can never be re-uploaded. Record the revert under metadata_changes."
          : '',
        'Fix every blocker and major finding. Judge the minors: fix if cheap, otherwise report them as unresolved. If a finding is wrong, say so in notes instead of "fixing" it.',
        red?.test_id
          ? `Then confirm the reproduction now passes: \`${red.command || 'run it'}\` for ${red.test_id}. It was ${red.fails_now ? 'observed failing' : 'NOT confirmed failing'} before the fix. Set red_to_green accordingly.`
          : 'There is no confirmed reproduction test for this fix -- set red_to_green=false and note it.',
        checksIn(issue.worktree),
        BASELINE_NOTE,
        PACKAGING,
        HOUSE,
        `You get at most ${VERIFY_ROUNDS} attempts at getting green. If it is still red after that, STOP: set green=false, needs_human=true, and list exactly what fails. Do not keep grinding and do not disable, skip or xfail a test to get there.`,
        'List under new_failures anything failing that is not in the baseline above. Do not commit.',
        `Finally, record changed_files: run \`git add -A -N && git diff --name-only origin/${BASE}\` and report every path it prints, verbatim. Later stages decide what still needs checking from that list, so an omission there silently removes a check.`,
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `verify#${issue.number}`, phase: 'Verify', schema: VERIFY_SCHEMA },
    )
    return verify ? { ...ctx, verify } : null
  },

  // Docs -- the tech writing stage. It runs AFTER the code is green (there is no point
  // documenting a fix that did not land) and BEFORE Ship, so the docs travel in the same
  // commit and the same PR as the change that made them true. Documentation written in a
  // follow-up PR is documentation that never gets written.
  async (ctx) => {
    if (!ctx) return null
    const { issue, plan, review, verify } = ctx
    if (!DO_DOCS) return { ...ctx, docs: null }
    // Nothing green, nothing to document: writing up a fix that does not work yet
    // produces a page that has to be rewritten or reverted.
    if (verify.green !== true) {
      log(`#${issue.number}: docs skipped -- the checks are not green`)
      return { ...ctx, docs: null }
    }
    // The reviewer -- a different agent, told to falsify, reading the diff -- already
    // priced the user-facing impact of this change. When it says "none", there is nothing
    // for a writer to make true, and a writer sent in anyway finds something to edit: that
    // is how a Docker cleanup reviewed as `none` came back having rewritten a versioning
    // page nobody named. The opposite direction is still cross-checked at Followups, where
    // a writer claiming user_visible=false against a priced release files an issue.
    if ((review.packaging?.release_impact || 'unknown') === 'none') {
      log(`#${issue.number}: docs skipped -- the review priced this change as no release impact`)
      return { ...ctx, docs: null }
    }
    if (!gate('docs', classify(plan, verify, IMPORT_NAME), issue)) return { ...ctx, docs: null }

    const docs = await call(
      [
        `cd ${issue.worktree}.`,
        `You are the technical writer for issue #${issue.number}: ${issue.title}. The fix is written and the checks are green. Your job is to leave the documentation true again.`,
        `What the issue asked for: ${issue.summary}`,
        `Root cause that was fixed: ${plan.root_cause || 'see the diff'}`,
        plan.api_surface ? `Public importable names this change touches: ${plan.api_surface}` : 'The change touches no public importable name.',
        `Release impact as reviewed: ${review.packaging?.release_impact || 'unknown'}${review.packaging?.changelog_line ? ` -- "${review.packaging.changelog_line}"` : ''}`,
        diffHowTo(issue),
        'READ THE DIFF FIRST. Document what the code now does, not what the issue hoped it would do -- those differ more often than anyone expects.',
        // docs/ is not one audience any more. docs/agents/ is written FOR agents and partly
        // BY them -- rules they read, and a gitignored runs/ directory full of machine output
        // from previous runs. A writer sent to "leave the documentation true" over all of
        // docs/ would happily edit last week's run report.
        'SCOPE: you write USER-facing documentation -- `README.md` and the pages under `docs/` that a person installing this package reads. Three things under `docs/` are NOT yours and must not be edited here: `docs/agents/` (rules written for agents, plus `docs/agents/runs/`, which is machine output from other runs and is gitignored), `docs/adr/` (decision records are immutable -- superseded, never rewritten), and `docs/agents/research/`. If one of them is genuinely wrong, say so in notes and leave it alone.',
        '',
        skill('tech-writing/SKILL.md', 'the documentation tree, the user-visible test, the smallest-true-edit procedure and the house voice. It is binding -- follow it exactly.'),
        '',
        'Fill the schema from it: user_visible per its FIRST DECIDE test (false means you change NOTHING and say why in notes); edited with the pages you changed; stale_found with documentation that was ALREADY wrong before this change; broken_links with any link that does not resolve.',
        'Keep the diff small and reviewable: this lands in the same PR as the fix, and a reviewer reads it alongside the code.',
        'Do not commit -- a later stage does that.',
        HOUSE,
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `docs#${issue.number}`, phase: 'Docs', schema: DOCS_SCHEMA },
    )
    log(
      docs?.user_visible === false
        ? `#${issue.number}: docs -- nothing user-visible to document`
        : `#${issue.number}: docs -- ${(docs?.edited || []).length} page(s) updated`,
    )
    return { ...ctx, docs }
  },

  // Package -- build the artifact and exercise it from a clean install.
  // Gated in plain JS, not by an agent's judgement. 'auto' used to ask opus to predict
  // whether the diff "touches packaging"; whether a plan changes anything but test files
  // is a fact about the plan, and filtering is code's job, not an agent's. Conservative on
  // purpose: this is the only check that can catch a fix that works in the source tree and
  // is missing from the wheel, so it runs unless the change is provably test-only.
  async (ctx) => {
    if (!ctx) return null
    const { issue, plan, red, review, verify } = ctx
    // Classified by PATH now, not by task kind. The old predicate was `t.kind !== 'test'`,
    // which is true for ANY non-test task, so it could not tell a docs edit from a code one.
    // This still builds a wheel for a docs-only change, deliberately: README.md IS the wheel
    // long_description, and `twine check` is the only thing in this pipeline that renders it.
    // The gain is that the decision is now readable and testable, not that it skips more.
    const wanted = PKG_GATE === 'always' || (PKG_GATE !== 'never' && gate('package', classify(plan, verify, IMPORT_NAME), issue))
    if (!wanted) {
      if (PKG_GATE === 'never') log(`#${issue.number}: package gate skipped -- packageGate:'never'`)
      return { ...ctx, pkg: null }
    }

    const pkg = await call(
      [
        `cd ${issue.worktree}.`,
        'The unit suite passing proves the SOURCE TREE works. It does not prove the WHEEL works: tests import the source tree, users import the installed package. This gate builds the artifact and exercises it.',
        DIST_FACTS,
        'Everything runs INSIDE an ephemeral container mounted on your worktree: the host has neither `build` nor `twine`, and its virtualenv is incomplete. Open a shell there:',
        '```',
        `MSYS_NO_PATHCONV=1 docker run --rm -e GRAPHKNOWS_SKIP_PREFLIGHT=1 -v "${winPath(issue.worktree)}:/app" -w /app graphknows-workspace sh`,
        '```',
        'Inside it, /tmp is a real Linux tmpdir that dies with the container, so no artifact can ever land in the worktree. Run the steps one at a time and record what each actually printed.',
        '  1. `uvx --from build pyproject-build --outdir /tmp/dist` -- the image ships `uv`, not `build`. Both an sdist (.tar.gz) and a wheel (.whl) must appear.',
        '  2. `uvx twine check /tmp/dist/*` -- metadata is valid and the README renders.',
        '  3. `python -m zipfile -l /tmp/dist/*.whl`. Confirm EVERY file this change added is inside it -- new modules, and especially non-.py data files and py.typed. The backend is hatchling: it ships exactly what pyproject tells it to and nothing else. A missing file here is precisely the bug this gate exists to catch.',
        '  4. Fresh venv: `python -m venv /tmp/venv`, then `/tmp/venv/bin/pip install /tmp/dist/*.whl` -- ONLY the built wheel. Not the source tree, not editable.',
        `  5. \`cd /tmp\` (OUT of /app), then \`/tmp/venv/bin/python -c "import ${IMPORT_NAME}; print(${IMPORT_NAME}.__file__)"\`. It must resolve inside /tmp/venv/lib/*/site-packages. If it prints a path under /app you are still importing the source tree -- cd out and rerun, the check is worthless otherwise.`,
        red?.test_id
          ? `  6. Copy the reproduction test out of the source tree (\`cp /app/<test file> /tmp/\`, plus /app/tests/conftest.py if it needs fixtures) and run ${red.test_id} from /tmp with \`/tmp/venv/bin/python -m pytest\`. Copying is the point: run it from /app and pytest puts the source tree on sys.path, so you would be testing the code you did not install. Passing from the source tree but failing here means the fix does not ship -- a blocker, record it under problems.`
          : '  6. No reproduction test exists; instead exercise the changed code path directly against the installed package from /tmp, and say exactly what you ran.',
        '  7. Exit the container (`--rm` takes /tmp with it), then on the HOST confirm `git status --short` in the worktree shows no dist/, build/ or *.egg-info. hatchling should leave none; delete any that appeared and set worktree_clean accordingly.',
        '',
        'You are the only stage that MEASURES packaging rather than reasoning about it, so you also fix what you break. If a step above fails because this change left the metadata wrong -- a third-party import missing from [project.dependencies], a runtime package sitting in a dev group, a new directory without __init__.py, a data file the backend was never told to ship -- then edit pyproject.toml (or add the __init__.py), rebuild from step 1, and record each edit under "fixed".',
        skill('python-packaging/REFERENCE.md', 'the authority on dependency specifiers and what a build backend ships -- consult it before editing pyproject.toml.'),
        skill('python-packaging/SKILL.md', 'dependency specifier grammar (PEP 508) and extras vs dependency-groups -- which table a new dependency belongs in.'),
        'Declare every dependency in the correct table with a lower bound. Two rules bound this: NEVER change the version, and never widen what ships just to make a step pass -- if you cannot fix it honestly, leave it and record it under "problems".',
        'NEVER upload anything: no `twine upload`, no TestPyPI, no PyPI, no git tag, no version edit. This gate builds and inspects only -- publishing belongs to a separate release workflow.',
        'If a step genuinely cannot run (no network for uvx, docker unavailable), set that field false and say exactly why in notes. Never report a check you did not actually run.',
        'Anything you find that is NOT caused by this change -- a file that was already missing from the wheel, metadata that was already wrong, a dependency nobody declared -- goes under "discoveries", not "problems". Those get filed as their own GitHub issues: packaging defects are invisible to the test suite, so an unfiled one stays unknown indefinitely.',
        verify.green ? '' : 'Note: the unit checks were NOT green. Run this gate anyway and report what you find.',
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `package#${issue.number}`, phase: 'Package', schema: PACKAGE_SCHEMA },
    )
    if (pkg?.problems?.length) log(`#${issue.number}: package gate problems -- ${pkg.problems.join('; ')}`)
    return { ...ctx, pkg }
  },

  // Followups -- the self-evolving edge. Everything filed here lands behind the
  // human triage gate, so a run can extend the backlog but never hand itself work.
  async (ctx, _issue, index) => {
    if (!ctx) return null
    const { issue, plan, red, review, verify, pkg, docs, dogfood } = ctx
    const candidates = [
      ...(dogfood?.discoveries || []).map((x) => ({ source: 'dogfood', ...x })),
      ...(dogfood?.smells || [])
        .filter((x) => x.caused_by_this_change === false)
        .map((x) => ({ source: 'dogfood', kind: 'bug', summary: `bad data: ${x.smell}`, where: dogfood.artifact || 'the produced data', evidence: x.evidence })),
      // Documentation that was already wrong before this change. Nobody reads a doc
      // sceptically, so a stale page nobody files stays stale until it misleads someone.
      ...(docs?.stale_found || []).map((x) => ({ source: 'docs', kind: 'bug', summary: `stale documentation: ${x.summary}`, where: x.file, evidence: 'noticed while documenting this change' })),
      ...(docs?.broken_links || []).map((l) => ({ source: 'docs', kind: 'bug', summary: `broken documentation link: ${l}`, where: 'docs/', evidence: 'the link does not resolve' })),
      // The writer decided nothing was user-visible; the reviewer -- a different agent, told
      // to falsify -- priced it as a real release. Both cannot be right, and the one with an
      // incentive to say "nothing to do here" is the writer.
      ...(docs?.user_visible === false && !['none', '', undefined].includes(review.packaging?.release_impact)
        ? [
            {
              source: 'docs',
              kind: 'bug',
              summary: `#${issue.number} was documented as not user-visible but reviewed as a ${review.packaging.release_impact} release`,
              where: 'docs/',
              evidence: docs.notes || 'no reason given for user_visible=false',
            },
          ]
        : []),
      ...(plan.out_of_scope || []).map((s) => ({ source: 'plan', kind: 'bug', summary: s })),
      ...(review.out_of_scope_defects || []).map((x) => ({ source: 'review', kind: 'bug', ...x })),
      ...(verify.unresolved || []).map((s) => ({ source: 'verify-unresolved', kind: 'bug', summary: s })),
      // The wheel gate's findings used to be logged here and then dropped on the floor --
      // the one class of defect a green suite structurally cannot see, thrown away.
      ...(pkg?.problems || []).map((s) => ({ source: 'package-gate', kind: 'bug', summary: s })),
      ...(pkg?.discoveries || []).map((x) => ({ source: 'package-gate', ...x })),
      ...(ctx.discoveries || []),
      ...(red?.fails_now === false
        ? [
            {
              source: 'reproduction',
              kind: 'bug',
              summary: `the reproduction written for #${issue.number} would not go red`,
              where: red.test_id,
              evidence: red.note || red.failure_excerpt || 'no detail given',
            },
          ]
        : []),
      // Baseline failures belong to the run, not to an issue. Filed once, from the first
      // issue's filer -- otherwise every issue in the run files the same ones again.
      ...(index === 0 && baseline?.ran
        ? (baseline.failing || []).map((f) => ({
            source: 'baseline',
            kind: 'bug',
            summary: `${f.id} already fails on ${BASE}`,
            where: f.id,
            evidence: f.summary,
          }))
        : []),
    ]
    if (!MAX_NEW_ISSUES || !candidates.length) return { ...ctx, followups: null }

    const followups = await call(
      [
        `You are filing GitHub issues for everything this run discovered while fixing issue #${issue.number} on branch ${issue.branch}. ${GH}`,
        'Every stage of this run reports what it stumbled over, and you are the only place those become durable. Anything you drop is forgotten. But this repository also runs an automated loop that picks issues up and fixes them, so a bad issue costs real machine time and a duplicate costs a human. Precision over volume: filing nothing is a perfectly good outcome.',
        `Candidates, gathered from every stage:\n${JSON.stringify(candidates, null, 2)}`,
        'For EACH candidate, in this order:',
        `  1. JUDGE it by its "kind".\n${EVIDENCE}\nDrop anything already fixed on this branch and anything that is speculation. Verify it against the actual code in ${issue.worktree} before believing it -- a candidate is a claim, not a fact.`,
        GRAPHIFY,
        '  2. DEDUPE, before creating anything: `gh issue list --state all --search "<3-5 distinctive terms>" --limit 20` and read the close matches. If an open issue already covers it, `gh issue comment <n>` with the new evidence and record it under "commented". Never open a second issue for something already known.',
        `  3. CREATE: \`gh issue create --title "<symptom first, specific, searchable>" --label "${FOUND_LABEL}" --label "${GATE_LABEL}" --label "<bug|enhancement>" --body "<body>"\``,
        '     A BUG body, in order: **Symptom** / **Where** (file:line) / **Concrete failure scenario** (inputs -> wrong output) / **How it was found** (' +
          `during ${RUN}, while fixing #${issue.number}) / **Why it was out of scope for #${issue.number}** / **First thing to check** (one line, no fix design).`,
        "     An ENHANCEMENT body, in order: **What to build** (the end-to-end behaviour this makes work, from the user's side -- not a layer-by-layer implementation list) / **Acceptance criteria** (a checklist) / **Blocked by** (issue references, or \"None -- can start immediately\") / **How it was found** (as above). Keep file paths and code snippets out of an enhancement body; they go stale fast.",
        '     If it is a packaging defect (missing from the wheel, an undeclared dependency, broken metadata), say so in the TITLE -- those are invisible to the test suite and a human should see that immediately.',
        `     A candidate whose source is "baseline" already fails on ${BASE} and belongs to nobody: say that in the body, and do not attribute it to #${issue.number}.`,
        `     Labels: "${FOUND_LABEL}" + "${GATE_LABEL}" + exactly one of "bug" / "enhancement". "${GATE_LABEL}" is the gate that stops this loop from feeding itself work it invented -- never omit it, and never remove it from an existing issue.`,
        `     NEVER apply "${READY_LABEL}". Some skills in this repo end by telling you to; they are written for a human driving a slash command, not for this loop. "${READY_LABEL}" is what a human adds AFTER removing "${GATE_LABEL}".`,
        `Hard cap: create at most ${MAX_NEW_ISSUES} new issues. If more clear the bar, file the highest-impact ones and put the rest under "dropped" with why.`,
        `Do not modify any code. Do not touch issue #${issue.number} or its branch. Do not close, label or reopen anything else.`,
      ].join('\n'),
      { label: `followups#${issue.number}`, phase: 'Followups', schema: FOLLOWUP_SCHEMA },
    )
    if (followups?.filed?.length) log(`#${issue.number}: filed ${followups.filed.map((f) => '#' + f.number).join(', ')}`)
    return { ...ctx, followups }
  },

  // Ship -- commit, push, open or update a draft PR
  async (ctx) => {
    if (!ctx) return null
    const { issue, red, review, verify, pkg, followups, docs, dogfood } = ctx
    // `ship: false` stops at the last honest moment: the work is finished, checked and
    // sitting in the worktree, but nothing has touched the remote. This is how you smoke
    // test the whole workflow on a real issue without leaving a branch or a PR behind.
    if (opt.ship === false) {
      log(`#${issue.number}: ship skipped (ship:false) -- work left uncommitted in ${issue.worktree}`)
      return {
        ...ctx,
        ship: {
          shipped: false,
          action: 'skipped',
          branch: issue.branch,
          notes: `ship:false -- nothing committed, pushed or opened. Read the diff in ${issue.worktree}.`,
        },
      }
    }
    const refs = (followups?.filed || []).map((f) => '#' + f.number)
    const pk = review.packaging || {}
    const pkgProblems = pkg?.problems || []
    const gapsKnown =
      !verify.green || verify.red_to_green === false || red?.fails_now === false || pkgProblems.length > 0 || pk.release_impact === 'major' ||
      (dogfood?.ran === true && dogfood.verdict !== 'edible')

    const ship = await call(
      [
        `cd ${issue.worktree}.`,
        verify.green
          ? 'All checks passed.'
          : `Checks are NOT green: ${(verify.unresolved || []).join('; ') || verify.notes}. Still open the PR, but keep it a draft and mark the gaps in the body.`,
        red?.fails_now === false
          ? 'NOTE: the reproduction test never failed before the fix, so this change is unproven. Say so plainly under "Known gaps".'
          : '',
        pkgProblems.length ? `NOTE: the package gate found problems -- ${pkgProblems.join('; ')}. Keep the PR a draft and list these under "Known gaps".` : '',
        pk.release_impact === 'major'
          ? 'NOTE: this is judged a BREAKING change. Keep the PR a draft and put "Breaking change" at the top of the body -- a human decides whether that is acceptable.'
          : '',
        '1. Before committing, check `git status --short` for `dist/`, `build/` or `*.egg-info/` and delete any that appear -- build artifacts are never committed.',
        '2. `git add -A` and commit. Commit message: extremely concise, lower-case imperative subject, no body unless it earns one. NEVER add Co-Authored-By or any author trailer. Let the pre-commit hooks run -- if one fails, fix the cause and recommit; never use --no-verify.',
        `3. \`git push -u origin ${issue.branch}\`. ${PREPUSH}`,
        '   Then confirm with `git ls-remote origin ' + issue.branch + '`.',
        `4. Check for an existing PR first: \`gh pr list --head ${issue.branch} --state open --json number,url\`.`,
        `   - none: \`gh pr create ${DRAFT || gapsKnown ? '--draft ' : ''}--base ${BASE} --head ${issue.branch} --title "<concise>" --body "<body>"\``,
        '   - one already open: `gh pr edit <n> --body "<body>"` instead. Never open a second PR for the same branch.',
        '',
        'PR body, in this order:',
        '   - **Root cause** and what changed.',
        `   - **How it was tested**: checks run (${(verify.checks_run || []).join(', ') || 'none'}); reproduction ${
          red?.test_id ? `${red.test_id}, red before / ${verify.red_to_green ? 'green after' : 'NOT confirmed green after'}` : 'none'
        }${
          pkg
            ? `; package gate: built=${pkg.built === true}, twine=${pkg.twine_check === true}, wheel complete=${pkg.wheel_complete === true}, clean install imports=${pkg.imports_from_site_packages === true}`
            : '; package gate: not required for this change'
        }.`,
        `   - **Release impact: ${pk.release_impact || 'unknown'}** -- ${pk.changelog_line || 'no user-facing change'}.${
          pk.public_api_change ? ` Public API: ${pk.public_api_change}.` : ''
        } The version was deliberately NOT bumped: releasing is a separate, human-run step.`,
        [...(verify.metadata_changes || []), ...(pkg?.fixed || [])].length
          ? `   - **Dependency / metadata changes**: ${[...(verify.metadata_changes || []), ...(pkg?.fixed || [])].join('; ')}.`
          : '',
        dogfood?.ran
          ? `   - **The data it produces**: read ${dogfood.records_read || 0} record(s) from ${dogfood.artifact || 'the changed path'} -- ${dogfood.verdict}.${
              (dogfood.still_wrong || []).length ? ` Still wrong: ${dogfood.still_wrong.join('; ')}.` : ''
            }`
          : '',
        docs?.user_visible === true && (docs.edited || []).length
          ? `   - **Documentation**: ${(docs.edited || []).map((e) => `${e.file} (${e.change})`).join('; ')}.`
          : '   - **Documentation**: no user-visible change, so none was written.',
        `   - \`Closes #${issue.number}\`.`,
        refs.length ? `   - **Found along the way**: ${refs.join(', ')} as \`Refs\` -- these must NOT be auto-closed by this PR.` : '',
        gapsKnown || (verify.unresolved || []).length ? '   - **Known gaps**: spell out everything unresolved.' : '',
        '',
        `Verify notes: ${verify.notes}`,
        'Do not tag, do not build, do not upload anything.',
        GH,
        'Report the PR URL. If the push or PR fails (no auth, protected branch, no remote), set shipped=false, action="failed" and say exactly what failed -- do not retry blindly.',
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `ship#${issue.number}`, phase: 'Ship', schema: SHIP_SCHEMA },
    )
    // Never propagate null from here on. A late stage that returns null drops the whole ctx,
    // and the summary then reports reproduced=false / green=false / "not run" for work that
    // demonstrably succeeded -- the run lies about itself, which is worse than an honest
    // failure. If the agent gave us nothing, say exactly that and keep everything upstream.
    return {
      ...ctx,
      ship: ship || {
        shipped: false,
        action: 'failed',
        branch: issue.branch,
        notes: `the ship agent returned no structured result -- inspect ${issue.worktree}: it may have committed, pushed, or opened a PR before dying`,
      },
    }
  },

  // CI -- the only judge that is not one of our own agents. Iterate on the PR until its
  // run is green, then merge. Deliberately in this order: ci.yml triggers on
  // `pull_request` as well as `push`, so the PR run is available BEFORE main is touched,
  // and merging a red branch to prove it is red is not a thing worth doing.
  async (ctx) => {
    if (!ctx) return null
    const { issue, ship } = ctx
    // Watching CI is read-only; merging is the only step that writes to main. They used to
    // share one gate, so a default run (merge:false) pushed a PR and never looked at its CI
    // -- the whole phase was dead unless you also authorised a merge. Separate gates now:
    // watch whatever we shipped, merge only when asked AND only when CI went green on its own.
    if (!ship?.shipped) {
      return { ...ctx, ci: null, merge: null }
    }

    // `gh pr ready|merge` take a POSITIONAL (number | url | branch). `--head` is a `gh pr list`
    // flag, so the old fallback rendered `gh pr ready --head fix/issue-12` and failed in exactly
    // the degraded case it existed for. A bare branch name is a valid positional.
    const prRef = ship.pr_url || issue.branch
    let ci = null
    for (let round = 1; round <= CI_ROUNDS; round++) {
      ci = await call(
        [
          `cd ${issue.worktree}. You are round ${round} of at most ${CI_ROUNDS} on GitHub Actions for PR ${prRef} (branch ${issue.branch}). ${GH}`,
          '',
          'STEP 1 -- find the run and wait for it to settle.',
          `  \`gh run list --branch ${issue.branch} --limit 5 --json databaseId,status,conclusion,event,headSha,url\``,
          '  Take the newest run whose headSha matches the branch tip (`git rev-parse HEAD` after `git fetch origin`).',
          '  A run can take several minutes and your individual commands time out well before that, so POLL -- do not block:',
          '    `for i in $(seq 1 18); do gh run view <id> --json status,conclusion -q .status; sleep 20; done`',
          '  and repeat that command until status is "completed". Roughly 6 minutes of waiting per call.',
          '  If NO run appears within about 3 minutes of polling, stop and set outcome="no-run". Say whether',
          '  .github/workflows/ci.yml actually exists on the base branch -- if it does not, no push can ever trigger a run,',
          '  and that is the finding, not a failure of this PR.',
          '',
          'STEP 2 -- if the conclusion is success, set outcome="green" and stop. You are done, change nothing.',
          '',
          'STEP 3 -- if it failed, read the actual log before touching anything:',
          '  `gh run view <id> --log-failed` (fall back to `gh run view <id> --log` if that is empty).',
          '  Record every failing job/step under failing_jobs with the error line, not a paraphrase.',
          '  Then judge WHY it failed, because the three causes need different responses:',
          '    a) OUR CODE is wrong -- a real failure this change caused. Fix the code.',
          '    b) The CI CONFIG is wrong, or the check is flaky/infra (a runner timeout, a network blip, a cache miss).',
          '       Do NOT weaken the check to get green. Re-run it once with `gh run rerun <id> --failed` and see.',
          '       If it fails the same way twice, it is not flaky -- report it as still-red with what you found.',
          "    c) The failure is PRE-EXISTING on the base branch and nothing to do with this change. Say so, and do not",
          '       paper over it here.',
          '  Fix only case (a). Then commit and push:',
          '    `git add -A`, commit with a concise lower-case imperative subject, no author trailer, hooks NOT bypassed,',
          `    then \`git push origin ${issue.branch}\`. ${PREPUSH}`,
          '  Set outcome="fixed-and-pushed" so the next round picks up the new run. If you cannot fix it, outcome="still-red".',
          '',
          'Anything you learn that is NOT this failure -- a check that should exist and does not, a flaky test, a config',
          'smell -- goes under "discoveries" for filing, not into this diff.',
          BASELINE_NOTE,
          HOUSE,
        ].join('\n'),
        { label: `ci#${issue.number}:${round}`, phase: 'CI', schema: CI_SCHEMA },
      )
      log(`#${issue.number}: CI round ${round}/${CI_ROUNDS} -- ${ci?.outcome || 'no result'}`)
      if (!ci || ci.outcome !== 'fixed-and-pushed') break
    }

    if (!DO_MERGE) {
      log(`#${issue.number}: CI ${ci?.outcome || 'unknown'}; not merging (merge:false)`)
      return { ...ctx, ci, merge: null }
    }
    if (ci?.outcome !== 'green') {
      log(`#${issue.number}: NOT merging -- CI is ${ci?.outcome || 'unknown'}`)
      return { ...ctx, ci, merge: null }
    }

    const merge = await call(
      [
        `CI is green on PR ${prRef} for issue #${issue.number}. Merge it. ${GH}`,
        `  1. \`gh pr ready ${prRef}\` -- take it out of draft, a draft PR cannot be merged.`,
        `  2. \`gh pr merge ${prRef} --squash --delete-branch\`. Squash: one issue, one commit on main.`,
        '     If the merge is refused (conflict, protected branch, not mergeable), do NOT force anything: set merged=false',
        '     and say exactly what GitHub refused.',
        `  3. Confirm issue #${issue.number} closed itself from the "Closes" line: \`gh issue view ${issue.number} --json state\`.`,
        '     If it is still open, leave it open and note that -- do not close it by hand, that hides a broken link.',
        WATCH_MAIN
          ? [
              '  4. The merge pushes to main, which triggers CI again. Watch that run the same way: poll',
              '     `gh run list --branch main --limit 3 --json databaseId,status,conclusion,url` until it completes.',
              '     Report post_merge_green. If it went red, say so plainly and do NOT try to fix main here -- a red main is a',
              '     human decision (revert or roll forward), not something to improvise at the end of an automated run.',
            ].join('\n')
          : '  4. Do NOT wait for main\'s CI. Leave post_merge_green unset: nothing in this run acts on it, and the PR\'s own CI already went green on this exact tree. Stop once the merge is confirmed.',
        'Do not tag, do not release, do not bump a version, do not upload anything.',
      ].join('\n'),
      { label: `merge#${issue.number}`, phase: 'CI', schema: MERGE_SCHEMA },
    )
    log(`#${issue.number}: ${merge?.merged ? 'merged' : 'NOT merged'} -- ${merge?.notes || ''}`)
    return { ...ctx, ci, merge }
  },
)

const summary = issues.map((issue, i) => {
  const r = results[i]
  const pk = r?.review?.packaging || {}
  const pkgOk = !r?.pkg ? null : (r.pkg.problems || []).length === 0 && r.pkg.built === true
  return {
    issue: issue.number,
    title: issue.title,
    branch: issue.branch,
    worktree: issue.worktree,
    reproduced: r?.red?.fails_now === true,
    green: r?.verify?.green === true,
    red_to_green: r?.verify?.red_to_green === true,
    dogfood: !r?.dogfood ? 'not run' : r.dogfood.ran === false ? 'no data to read' : r.dogfood.verdict,
    package_gate: r?.pkg ? (pkgOk ? 'pass' : 'problems') : 'not run',
    docs: !r?.docs ? 'not run' : r.docs.user_visible === false ? 'nothing to document' : `${(r.docs.edited || []).length} page(s)`,
    release_impact: pk.release_impact || 'unknown',
    changelog_line: pk.changelog_line || '',
    shipped: r?.ship?.shipped === true,
    pr_url: r?.ship?.pr_url,
    ci: r?.ci?.outcome || 'not run',
    merged: r?.merge?.merged === true,
    post_merge_green: r?.merge?.post_merge_green,
    filed: (r?.followups?.filed || []).map((f) => f.number),
    discovered:
      (r?.discoveries || []).length + (r?.pkg?.discoveries || []).length + (r?.pkg?.problems || []).length,
    needs_human:
      r?.verify?.needs_human === true ||
      (r?.dogfood?.ran === true && r.dogfood.verdict !== 'edible') ||
      r?.verify?.green !== true ||
      r?.red?.fails_now !== true ||
      r?.ship?.shipped !== true ||
      pkgOk === false ||
      pk.release_impact === 'major' ||
      (DO_MERGE && r?.ship?.shipped === true && r?.merge?.merged !== true) ||
      r?.merge?.post_merge_green === false,
    notes: r?.ship?.notes || r?.verify?.notes || 'a pipeline stage returned nothing -- see the run journal',
  }
})

const filed = summary.flatMap((s) => s.filed)
log(`shipped ${summary.filter((s) => s.shipped).length}/${summary.length}; filed ${filed.length} follow-up issue(s)`)

// -- Run report ----------------------------------------------------
// One agent, no judgement: the markdown is composed here and it only writes the bytes.
// Named after the issues worked, so re-running an issue supersedes its own report
// instead of littering the runs directory with near-duplicates.
const REPORT_PATH = opt.reportPath || `${RUNS_DIR}/process-issues-${issues.map((i) => i.number).join('-')}.md`
const yn = (v) => (v === true ? 'yes' : v === false ? 'NO' : String(v))

if (DO_REPORT) {
  const body = [
    `# process-issues -- ${issues.map((i) => '#' + i.number).join(', ')}`,
    '',
    `Base branch \`${BASE}\`, ${RUN}. Merging was ${DO_MERGE ? 'enabled' : 'disabled'}.`,
    '',
    '| Issue | Repro red | Green | Dogfood | Docs | Wheel | Shipped | CI | Merged | Needs human |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ...summary.map(
      (s) =>
        `| #${s.issue} ${s.title} | ${yn(s.reproduced)} | ${yn(s.green)} | ${s.dogfood} | ${s.docs} | ${s.package_gate} | ${yn(s.shipped)} | ${s.ci} | ${yn(s.merged)} | ${yn(s.needs_human)} |`,
    ),
    '',
    '## Baseline',
    '',
    !baseline
      ? 'Skipped, so nothing in this report may claim a failure is pre-existing.'
      : !baseline.ran
        ? `Could not run: ${baseline.notes || 'no reason given'}. Nothing here may claim a failure is pre-existing.`
        : baseline.failing?.length
          ? [`${baseline.failing.length} check(s) already failing on \`${BASE}\` before this run:`, ...baseline.failing.map((f) => `- \`${f.id}\` -- ${f.summary}`)].join('\n')
          : `\`${BASE}\` was fully green at the start of this run, so every failure seen was caused by it.`,
    '',
    ...summary.flatMap((s, i) => {
      const r = results[i]
      const dg = r?.dogfood
      return [
        `## #${s.issue} -- ${s.title}`,
        '',
        `Branch \`${s.branch}\` in \`${s.worktree}\`.${s.pr_url ? ` PR: ${s.pr_url}` : ' No PR was opened.'}`,
        '',
        `**Root cause.** ${r?.plan?.root_cause || 'not recorded'}`,
        '',
        `**Reproduction.** ${
          r?.red?.test_id
            ? `\`${r.red.test_id}\` -- ${s.reproduced ? 'red before the fix' : 'NEVER went red before the fix, so this change is unproven'}, ${s.red_to_green ? 'green after' : 'not confirmed green after'}.`
            : 'none was written.'
        }`,
        '',
        `**The data it produces.** ${
          !dg
            ? 'Not read.'
            : dg.ran === false
              ? `Not read: ${dg.notes || 'the change produces no data anyone reads'}.`
              : `${dg.records_read || 0} record(s) from ${dg.artifact || 'the changed path'} -- **${dg.verdict}**.`
        }`,
        ...(dg?.smells || []).flatMap((x) => [
          `- ${x.caused_by_this_change ? 'caused here' : 'pre-existing'}: ${x.smell}`,
          '  ~~~',
          `  ${String(x.evidence).replace(/\s*\n\s*/g, ' ')}`,
          '  ~~~',
        ]),
        ...(dg?.refined || []).map((x) => `- refined: ${x}`),
        ...(dg?.still_wrong || []).map((x) => `- **still wrong**: ${x}`),
        '',
        `**Documentation.** ${
          !r?.docs
            ? 'Not run.'
            : r.docs.user_visible === false
              ? `Nothing user-visible: ${r.docs.notes}`
              : (r.docs.edited || []).map((e) => `\`${e.file}\` -- ${e.change}`).join('; ') || 'no page changed'
        }`,
        '',
        `**Wheel gate.** ${
          !r?.pkg
            ? 'Not required for this change.'
            : `built=${yn(r.pkg.built)}, twine=${yn(r.pkg.twine_check)}, wheel complete=${yn(r.pkg.wheel_complete)}, clean install imports=${yn(r.pkg.imports_from_site_packages)}${
                (r.pkg.problems || []).length ? `. Problems: ${r.pkg.problems.join('; ')}` : ''
              }`
        }`,
        '',
        `**Release impact.** ${s.release_impact}${s.changelog_line ? ` -- ${s.changelog_line}` : ''}. The version was deliberately not bumped.`,
        '',
        (r?.verify?.unresolved || []).length ? `**Unresolved after verify.** ${r.verify.unresolved.join('; ')}` : null,
        s.filed.length
          ? `**Filed along the way.** ${s.filed.map((n) => '#' + n).join(', ')} -- all gated behind \`${GATE_LABEL}\`.`
          : '**Filed along the way.** nothing.',
        '',
      ].filter((x) => x !== null)
    }),
    (triage?.skipped || []).length
      ? ['## Not picked up', '', ...triage.skipped.map((x) => `- #${x.number}: ${x.reason}`), ''].join('\n')
      : null,
    '## Next',
    '',
    filed.length
      ? `- Triage ${filed.map((n) => '#' + n).join(', ')}: remove \`${GATE_LABEL}\` from the ones worth doing, or this workflow will never pick them up.`
      : '- Nothing new was filed.',
    ...summary.filter((s) => s.needs_human).map((s) => `- **#${s.issue} needs a human**: ${s.notes}`),
    `- Worktrees left in place: ${[...issues.map((i) => i.worktree), DO_BASELINE ? '../gk-baseline' : '']
      .filter(Boolean)
      .join(', ')} (\`git worktree remove <path>\`).`,
  ]
    .filter((x) => x !== null)
    .join('\n')

  await call(
    [
      'Write a file. That is the whole task: no analysis, no summarising, no edits to the text.',
      `Path: ${MAIN_REPO}/${REPORT_PATH} -- create the directory if it is missing, overwrite the file if it exists.`,
      'Content: everything between the two marker lines below, byte for byte. The markers themselves are NOT part of the file.',
      'Do not reformat it, do not fix its grammar, do not add a heading or a footer, and do not append a note about yourself.',
      '',
      '<<<REPORT-BEGINS>>>',
      body,
      '<<<REPORT-ENDS>>>',
      '',
      'Then confirm the file exists and report its path and its size in bytes.',
    ].join('\n'),
    { label: 'report', phase: 'Report' },
  )
  log(`run report: ${REPORT_PATH}`)
}

return {
  report: DO_REPORT ? REPORT_PATH : 'not written',
  summary,
  baseline: baseline ? { ran: baseline.ran, failing: baseline.failing, distribution: baseline.distribution, notes: baseline.notes } : null,
  skipped: triage?.skipped || [],
  filed_issues: filed,
  needs_human: summary.filter((s) => s.needs_human).map((s) => s.issue),
  // The workflow deliberately never bumps a version, tags, or uploads. This is the
  // handover to whoever does: the changelog lines and the highest impact seen.
  release_notes: summary
    .filter((s) => s.shipped && s.changelog_line)
    .map((s) => `${s.release_impact}: ${s.changelog_line} (#${s.issue})`),
  next_step: filed.length
    ? `Review ${filed.map((n) => '#' + n).join(', ')} and remove the "${GATE_LABEL}" label from the ones worth doing -- until then this workflow will not pick them up.`
    : 'nothing new filed',
  cleanup: `worktrees left in place on purpose: ${[...issues.map((i) => i.worktree), DO_BASELINE ? '../gk-baseline' : null]
    .filter(Boolean)
    .join(', ')} (remove with \`git worktree remove <path>\`)`,
}
