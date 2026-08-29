// Self-checks for the decisions the two workflows make in plain JS rather than in a prompt.
//
// Four of them, all load-bearing and all silent when wrong:
//   - process-issues' change classifier, which decides whether the wheel gate runs -- the
//     one skip in that pipeline with no post-submit safety net.
//   - the label vocabulary the two workflows must share, with no imports to share it.
//   - prepare-tickets' handover check, which is the only thing standing between a malformed issue body
//     and the process-issues half that reads nothing else.
//   - run-merge-train's admission, ordering and dependent-parking helpers, which decide what an
//     N-merge autonomous run is allowed to touch, and its express classifier, which decides
//     which cars get reviewed as a batch instead of one at a time.
//   - that .gitignore still covers where the workflows write, and still does NOT cover the
//     rules they read.
//
// Every one is sliced out of the workflow source rather than copied: a copy drifts, and
// then the test tests the copy.
//
//   node .claude/workflows/gates.test.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const src = readFileSync(new URL('./process-issues.js', import.meta.url), 'utf8')
const from = src.indexOf('const norm = ')
const to = src.indexOf('// Whether the stage should run')
assert.ok(from > 0 && to > from, 'classifier block not found in process-issues.js -- did the markers move?')
const { classify, SKIPPABLE } = await import(
  'data:text/javascript,' + encodeURIComponent(src.slice(from, to) + '\nexport { classify, SKIPPABLE }')
)

const plan = (files, extra = {}) => ({ tasks: [{ files }], ...extra })

// -- kinds --
assert.equal(classify(plan(['tests/test_x.py']), null).kind, 'test-only')
assert.equal(classify(plan(['docs/modes.md', 'README.md']), null).kind, 'docs')
assert.equal(classify(plan(['graphknows/symbolic/index.py']), null).kind, 'code')
assert.equal(classify(plan([]), null).kind, 'unknown', 'no evidence is the expensive class, not the cheap one')
// Mixed docs+tests is neither pure class; it must not read as docs-only.
assert.equal(classify(plan(['docs/modes.md', 'tests/test_x.py']), null).kind, 'test-only')
// Windows separators and ./ prefixes normalise.
assert.equal(classify(plan(['tests\\test_x.py', './tests/test_y.py']), null).kind, 'test-only')

// -- what forces a test-only change back up to code --
assert.equal(classify(plan(['tests/test_x.py', 'pyproject.toml']), null).kind, 'code')
assert.equal(classify(plan(['tests/test_x.py'], { new_dependencies: ['rdflib>=7'] }), null).kind, 'code')

// -- measurement wins over an under-reporting plan --
const measured = classify(plan(['tests/test_x.py']), { changed_files: ['graphknows/core.py'] })
assert.equal(measured.kind, 'code', 'a file the plan never named must still count')
assert.equal(measured.measured, true)
assert.equal(classify(plan(['tests/test_x.py']), null).measured, false)

// -- the policy: the wheel gate is the one that must stay hard to skip --
const pkg = SKIPPABLE.package
assert.equal(pkg(classify(plan(['tests/test_x.py']), null)), true)
assert.equal(pkg(classify(plan(['graphknows/core.py']), null)), false)
assert.equal(pkg(classify(plan([]), null)), false, 'unknown must never skip the wheel gate')
assert.equal(pkg(classify(plan(['tests/test_x.py'], { api_surface: 'Memory.recall' }), null)), false)
assert.equal(pkg(classify(plan(['tests/test_x.py'], { new_dependencies: ['x'] }), null)), false)
// A docs-only change is NOT wheel-skippable, on purpose: README.md is the wheel
// long_description, and `twine check` is the only thing here that renders it.
assert.equal(pkg(classify(plan(['docs/modes.md']), null)), false)

// -- dogfood and docs are the cheap-if-wrong pair --
assert.equal(SKIPPABLE.dogfood(classify(plan(['docs/modes.md']), null)), true)
assert.equal(SKIPPABLE.dogfood(classify(plan(['graphknows/core.py']), null)), false)

// dogfood reads the DATA the code produces, so what matters is whether the diff touches the
// import package -- not whether it is "code". scripts/, .github/ and the Dockerfile are all
// kind:'code' and none of them produce a record anyone reads.
assert.equal(classify(plan(['graphknows/core.py']), null).kind, 'code')
assert.equal(SKIPPABLE.dogfood(classify(plan(['scripts/gate.sh', 'Dockerfile']), null)), true, 'a build-only change produces no data to taste')
assert.equal(classify(plan(['scripts/gate.sh']), null).kind, 'code', 'and it is still code -- only dogfood skips')
assert.equal(SKIPPABLE.dogfood(classify(plan([]), null)), false, 'no file list must not skip on ignorance')
// the import package can differ from the distribution name; the third arg is that name
assert.equal(SKIPPABLE.dogfood(classify(plan(['graph_knows/core.py']), null, 'graph_knows')), false)
assert.equal(SKIPPABLE.dogfood(classify(plan(['graph_knows/core.py']), null)), true, 'the wrong package name reads as untouched')
// a prefix that merely starts the same is a different directory
assert.equal(SKIPPABLE.dogfood(classify(plan(['graphknows_extra/x.py']), null)), true)
assert.equal(SKIPPABLE.docs(classify(plan(['tests/test_x.py']), null)), true)
assert.equal(SKIPPABLE.docs(classify(plan(['docs/modes.md']), null)), false, 'a docs change is exactly when docs runs')

// -- the two workflows must agree on the vocabulary of the gate --
//
// prepare-tickets.js files issues behind GATE_LABEL and process-issues.js refuses to pick them up while
// they carry it. Workflow scripts cannot import, so each declares the constants itself. If
// they ever drift, prepare-tickets files behind a gate process-issues no longer checks and nothing
// errors -- the issues just sit there, or worse, get worked without a human agreeing.
const rnd = readFileSync(new URL('./prepare-tickets.js', import.meta.url), 'utf8')
const train = readFileSync(new URL('./run-merge-train.js', import.meta.url), 'utf8')
const konst = (text, name) => text.match(new RegExp(`^const ${name} = '([^']*)'`, 'm'))?.[1]

// RUNS_DIR joins the drift check for a reason of its own: it is the one path in these
// scripts that .gitignore has an opinion about. If a workflow drifts to a directory the
// ignore rule does not cover, that run's report gets committed; if .gitignore drifts away
// from the directory, every report gets committed. Neither errors, and both are noticed
// months later by someone reading a diff full of machine output.
for (const name of ['GATE_LABEL', 'READY_LABEL', 'MAIN_REPO', 'RUNS_DIR']) {
  const a = konst(src, name)
  for (const [fname, text] of [['prepare-tickets.js', rnd], ['run-merge-train.js', train]]) {
    const b = konst(text, name)
    assert.ok(a, `${name} not found in process-issues.js`)
    assert.ok(b, `${name} not found in ${fname}`)
    assert.equal(b, a, `${name} has drifted between process-issues.js and ${fname}`)
  }
}

// ...and the ignore rule must actually cover it. This is the half the constant-drift check
// above cannot see: the three scripts can agree perfectly on a directory git commits.
const ignoreRules = readFileSync(new URL('../../.gitignore', import.meta.url), 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l && !l.startsWith('#'))
const runsDir = konst(src, 'RUNS_DIR')
assert.ok(
  ignoreRules.includes(`${runsDir}/`) || ignoreRules.includes(runsDir),
  `.gitignore does not cover ${runsDir} -- every run report would be committed`,
)
// The rules an agent READS sit beside what it writes and must NOT be swept up by that rule.
for (const kept of ['docs/agents/definition-of-done.md', 'docs/agents/triage-labels.md']) {
  assert.ok(
    !ignoreRules.some((r) => r === kept || r === 'docs/agents/' || r === 'docs/agents'),
    `${kept} is ignored -- an agent in a worktree cannot read it, so the rules end up restated in every ticket`,
  )
}

// -- the dry-run harness must not shadow its own fixtures --
//
// CANNED is one object literal keyed by agent-label prefix, shared by every workflow. A
// duplicate key is not an error in JS: the last one silently wins, the stage keyed by the
// first gets another workflow's data, and the dry-run reports "no problems found" while
// quietly skipping a stage. That happened -- an express-lane `ship` fixture shadowed
// process-issues', and the only symptom was an agent count one lower than the run before.
const dry = readFileSync(new URL('./dryrun.mjs', import.meta.url), 'utf8')
const cannedBody = dry.slice(dry.indexOf('const CANNED = {'), dry.indexOf('// Canned CHILD WORKFLOW results'))
const cannedKeys = [...cannedBody.matchAll(/^ {2}'?([A-Za-z][\w:-]*)'?:/gm)].map((m) => m[1])
const dupes = cannedKeys.filter((k, i) => cannedKeys.indexOf(k) !== i)
assert.deepEqual([...new Set(dupes)], [], 'duplicate keys in dryrun.mjs CANNED -- the later one silently wins and a stage gets the wrong fixture')

// -- prepare-tickets' handover check: the bodies are the whole contract ------
//
// The issue body is the only thing process-issues reads, so a missing section is a broken
// handover. Sliced out of prepare-tickets.js rather than copied, for the same reason as above.
const from2 = rnd.indexOf('const checkHandover = ')
const to2 = rnd.indexOf('const drafted = draft.tickets')
assert.ok(from2 > 0 && to2 > from2, 'checkHandover not found in prepare-tickets.js -- did the markers move?')
// labelCheck names the two labels, which are declared far above the slice. Inject them --
// and assert first that what we inject still matches what the file declares, or the test
// would happily prove a rule about strings the workflow no longer uses.
const LABELS = "const GATE_LABEL = 'needs-triage'\nconst READY_LABEL = 'ready-for-agent'\n"
assert.equal(konst(rnd, 'GATE_LABEL'), 'needs-triage')
assert.equal(konst(rnd, 'READY_LABEL'), 'ready-for-agent')
const { checkHandover, gateFor, HARNESS, labelCheck } = await import(
  'data:text/javascript,' + encodeURIComponent(LABELS + rnd.slice(from2, to2) + '\nexport { checkHandover, gateFor, HARNESS, labelCheck }')
)

const SPEC = '## Problem Statement\nx\n## Testing Decisions\nTested at the chunk writer.\n'
const body =
  '## Parent\nthe parent spec issue\n## What to build\nx\n' +
  '## Acceptance criteria\n- [ ] WHEN a chunk is stored THE SYSTEM SHALL return it by instant\n' +
  '## Verify\n`pytest -q` passes\n## Out of scope\nBack-filling.\n'
const tk = (blocked_by_index, over = {}) => ({ title: 't', body, blocked_by_index, ...over })

// clean set passes and comes back in dependency order
const ok = checkHandover(SPEC, ['the chunk writer'], [tk([1]), tk([])])
assert.deepEqual(ok.problems, [])
assert.deepEqual(ok.order, [1, 0], 'blockers must be filed before what they block')

// a seam the body never names is the failure this check exists for
assert.match(
  checkHandover(SPEC, ['the retrieval fuser'], [tk([])]).problems.join(),
  /never names the seam/,
)
assert.match(checkHandover('## Problem Statement\nx\n', [], [tk([])]).problems.join(), /Testing Decisions/)

// a ticket missing any required section
for (const missing of ['## Parent', '## What to build', '## Acceptance criteria', '## Verify', '## Out of scope']) {
  const stripped = body.replace(missing + '\n', '')
  assert.ok(
    checkHandover(SPEC, [], [tk([], { body: stripped })]).problems.length > 0,
    `a body missing ${missing} must be refused`,
  )
}

// -- the acceptance-criterion form --
//
// A criterion that describes the TREE instead of the behaviour is a test specification
// wearing a checkbox, and the implementer writes that test: 25 of them once came back as a
// 528-line test file in a change whose entire point was deletion. Caught by shape, because
// by the time a reviewer could catch it by taste the file already exists.
const withCriterion = (text) =>
  body.replace('- [ ] WHEN a chunk is stored THE SYSTEM SHALL return it by instant', `- [ ] ${text}`)
const criterionProblems = (text) => checkHandover(SPEC, [], [tk([], { body: withCriterion(text) })]).problems.join()

assert.match(criterionProblems('scripts/ contains exactly four files'), /state of the tree/)
assert.match(criterionProblems('docs/versioning.md is deleted'), /state of the tree/)
assert.match(criterionProblems('the HEALTHCHECK block has been removed'), /state of the tree/)
assert.match(criterionProblems('the lock-hash write no longer exists'), /state of the tree/)
assert.equal(criterionProblems('WHEN the image is built THE SYSTEM SHALL start with a working interpreter'), '')
// The word "removed" describing BEHAVIOUR is not a tree assertion -- the check keys on the
// grammar of "<thing> is removed", not on the vocabulary.
assert.equal(criterionProblems('WHEN a stale entry expires THE SYSTEM SHALL drop it from the index'), '')

// A marker is a question for the requester, not a ticket. It must never be filed as work
// an agent may start on.
assert.match(
  checkHandover(SPEC, [], [tk([], { body: body + '\n[NEEDS CLARIFICATION: which backend?]\n' })]).problems.join(),
  /NEEDS CLARIFICATION/,
)

// The spec was the one body nothing bounded, and one reached 56 KB of GitHub's cap.
const bigSpec = SPEC + 'filler\n'.repeat(401)
assert.match(checkHandover(bigSpec, [], [tk([])]).problems.join(), /spec body is \d+ lines, over 400/)

// size. Structure was checkable here from the start; size was not bounded anywhere, and
// that is the asymmetry that produced a twenty-six-criterion "tracer bullet".
const TAIL = '## Verify\n`pytest -q` passes\n## Out of scope\nBack-filling.\n'
const acs = (n) => `## Parent\nx\n## What to build\nx\n## Acceptance criteria\n${'- [ ] x\n'.repeat(n)}${TAIL}`
assert.deepEqual(checkHandover(SPEC, [], [tk([], { body: acs(8) })]).problems, [], '8 criteria is the ceiling, not over it')
assert.match(checkHandover(SPEC, [], [tk([], { body: acs(9) })]).problems.join(), /acceptance criteria, over the 8/)
const long = `## Parent\nx\n## What to build\n${'prose\n'.repeat(70)}## Acceptance criteria\n- [ ] x\n${TAIL}`
assert.match(checkHandover(SPEC, [], [tk([], { body: long })]).problems.join(), /line body, over 70/)

// the graph. A cycle makes Kahn stall rather than throw, so it gets its own case.
assert.match(checkHandover(SPEC, [], [tk([1]), tk([0])]).problems.join(), /cycle/)
assert.match(checkHandover(SPEC, [], [tk([0])]).problems.join(), /blocks itself/)
assert.match(checkHandover(SPEC, [], [tk([7])]).problems.join(), /not a ticket in this set/)
// every ticket blocked by another is a graph with no root -- nothing can ever start
assert.match(checkHandover(SPEC, [], [tk([1]), tk([0])]).problems.join(), /nothing can start|cycle/)

// -- prepare-tickets' gate: which tickets a person still owes a decision on --
//
// This used to be all of them, which made triage a chore rather than a signal. The rule is
// narrow on purpose, and narrow rules are the ones that widen quietly.
const tick = (over = {}) => ({ title: 't', needs_human: false, files_named: [], ...over })
const CLEAN = { specUnsound: false, refutedCount: 0 }

assert.equal(gateFor(tick(), CLEAN), '', 'the default is ready -- that is the whole point')
assert.match(gateFor(tick({ needs_human: true, needs_human_why: 'pick a floor' }), CLEAN), /pick a floor/)
assert.match(gateFor(tick({ needs_human: true }), CLEAN), /did not say why/, 'a flag with no reason is still a gate, and says so')
assert.match(gateFor(tick(), { ...CLEAN, specUnsound: true }), /not-ready/, 'an unsound design gates every slice of it')
assert.match(gateFor(tick(), { ...CLEAN, refutedCount: 2 }), /refuted 2/)
assert.match(gateFor(tick({ files_named: ['.github/workflows/ci.yml'] }), CLEAN), /rewrites the harness/)
assert.match(gateFor(tick({ files_named: ['scripts\\gate.sh'] }), CLEAN), /rewrites the harness/)
assert.equal(gateFor(tick({ files_named: ['docs/ci.md', 'graphknows/core.py'] }), CLEAN), '', 'a lookalike path must not gate')
assert.ok(HARNESS.includes('scripts/gate.sh'))

// -- the labels the filer actually applied --
//
// This is the check that exists because a prompt saying "carry exactly ONE of the two" is
// advisory. Four tickets of one spec were filed carrying BOTH; admit() checks the gate label
// first, so the pair read as "gated" and the whole remaining train silently refused to board.
// Nothing errored, and nobody found out until someone asked why the train had not moved.
const issue = (over) => ({ number: 1, labels: [], expected_gate: false, ...over })

assert.deepEqual(labelCheck([issue({ labels: ['ready-for-agent'] })]), [])
assert.deepEqual(labelCheck([issue({ labels: ['needs-triage'], expected_gate: true })]), [])
assert.match(labelCheck([issue({ labels: ['ready-for-agent', 'needs-triage'] })]).join(), /BOTH/)
assert.match(labelCheck([issue({ labels: ['enhancement'] })]).join(), /NEITHER/)
// filed gated when the script said ready, and the other way round -- both are drift between
// what was decided and what exists, and only re-reading GitHub can see it.
assert.match(labelCheck([issue({ labels: ['needs-triage'], expected_gate: false })]).join(), /filed gated but the script decided ready/)
assert.match(labelCheck([issue({ labels: ['ready-for-agent'], expected_gate: true })]).join(), /filed ready but the script decided gated/)

// -- run-merge-train's helpers: what an N-merge run may touch ---------
//
// admit() is the per-car human gate in code; topoOrder() is the landing order; and
// dependentsOf() is what keeps a parked car's descendants from dispatching against
// code that never landed. All sliced from run-merge-train.js, same as above.
const from3 = train.indexOf('const admit = ')
const to3 = train.indexOf('// -- Roster --')
assert.ok(from3 > 0 && to3 > from3, 'helpers not found in run-merge-train.js -- did the markers move?')
const consts = "const GATE_LABEL = 'needs-triage'\nconst READY_LABEL = 'ready-for-agent'\n"
const { admit, topoOrder, dependentsOf, isExpress } = await import(
  'data:text/javascript,' +
    encodeURIComponent(consts + train.slice(from3, to3) + '\nexport { admit, topoOrder, dependentsOf, isExpress }')
)

const car = (number, over = {}) => ({ number, title: 't', state: 'OPEN', labels: ['enhancement', 'ready-for-agent'], blocked_by: [], ...over })

// admission: presence of ready-for-agent, not mere absence of needs-triage
assert.equal(admit(car(1), [1]), '')
assert.match(admit(car(1, { labels: ['enhancement'] }), [1]), /ready-for-agent/, 'silence is not consent')
assert.match(admit(car(1, { labels: ['enhancement', 'ready-for-agent', 'needs-triage'] }), [1]), /needs-triage/, 'the gate label wins even alongside approval')
assert.match(admit(car(1, { state: 'CLOSED' }), [1]), /not open/)
assert.match(admit(car(1, { open_pr: 'https://x/pr/9' }), [1]), /parked from an earlier run/)
// a blocker outside the train parks the car only while it is open
assert.match(admit(car(2, { blocked_by: [99], blocker_open: { 99: true } }), [2]), /#99/)
assert.equal(admit(car(2, { blocked_by: [99], blocker_open: { 99: false } }), [2]), '', 'a closed outside blocker is satisfied')
assert.equal(admit(car(2, { blocked_by: [3] }), [2, 3]), '', 'an in-roster blocker is an ordering edge, not an exclusion')

// the harness guard: a car cannot verify a change to the thing that verifies it
assert.match(admit(car(1, { files_named: ['.github/workflows/ci.yml'] }), [1]), /rewrites the harness/)
assert.match(admit(car(1, { files_named: ['scripts\\gate.sh'] }), [1]), /rewrites the harness/, 'windows separators must not slip past')
assert.equal(admit(car(1, { files_named: ['graphknows/core.py', 'docs/ci.md'] }), [1]), '', 'a normal file that merely resembles one must still board')

// ordering: blockers first, cycles surfaced instead of hiding as an empty order
assert.deepEqual(topoOrder([car(3, { blocked_by: [2] }), car(2)]).order, [2, 3])
const cyc = topoOrder([car(1, { blocked_by: [2] }), car(2, { blocked_by: [1] })])
assert.deepEqual(cyc.cyclic.sort(), [1, 2], 'a cycle must be named, not silently dropped')

// -- the express lane: which cars may share one CI run --
//
// The reduced scrutiny this lane buys is only defensible while the changes really are tiny,
// so the rule has to be strict AND it has to fail closed. Every branch below is a way the
// batch could quietly acquire a change that deserved its own review.
const tiny = (over = {}) => car(9, { sizing: 'stated', expected_lines: 8, files_named: ['pyproject.toml'], ...over })

assert.equal(isExpress(tiny(), [9]), '', 'a stated two-file, eight-line ticket is exactly what this lane is for')
// Absence of a size is not smallness. An unsized ticket fails closed and takes the long
// road -- the failure that matters is a big change riding a lane that reviews once.
assert.match(isExpress(tiny({ sizing: undefined }), [9]), /could not be sized/)
assert.match(isExpress(car(9, {}), [9]), /could not be sized/, 'a car with no field at all is not express')
assert.match(isExpress(tiny({ expected_lines: 0 }), [9]), /no usable size/)
// A measured size has to be defended: files read and evidence cited, or it is a guess.
assert.match(isExpress(tiny({ sizing: 'measured', sizing_evidence: '' }), [9]), /cites no evidence/)
assert.equal(
  isExpress(tiny({ sizing: 'measured', sizing_evidence: 'read pyproject.toml: one table row moves' }), [9]),
  '',
  'a measured size backed by evidence rides',
)
assert.match(isExpress(tiny({ expected_lines: 31 }), [9]), /over the 30/)
assert.equal(isExpress(tiny({ expected_lines: 30 }), [9]), '', '30 is the ceiling, not over it')
assert.match(isExpress(tiny({ files_named: ['a.py', 'b.py', 'c.py'] }), [9]), /over 2/)
// A car waiting on another car cannot be in the batch: the batch is cut before its blocker
// has landed, so it would be implemented against code that does not exist yet.
assert.match(isExpress(tiny({ blocked_by: [7] }), [7, 9]), /blocked by another car/)
// ...but a blocker that is NOT on this train has already been cleared by admit(), which
// refuses a car whose outside blocker is still open. So it does not block the express lane.
assert.equal(isExpress(tiny({ blocked_by: [99] }), [9]), '', 'a settled outside blocker is not an express exclusion')

// dependent parking is transitive
const dag = [car(1), car(2, { blocked_by: [1] }), car(3, { blocked_by: [2] }), car(4)]
assert.deepEqual(dependentsOf(dag, 1).sort(), [2, 3], 'parking #1 parks its whole chain and spares the independent car')
assert.deepEqual(dependentsOf(dag, 4), [])

// prepare-tickets and run-merge-train both refuse to let an agent verify a change to the thing that
// verifies it. Two copies, no imports: if they drift, prepare-tickets files a ticket ready that the
// train would have refused, and nothing errors.
const harnessList = (text) => text.match(/^const HARNESS = \[([^\]]*)\]/m)?.[1]
assert.ok(harnessList(rnd) && harnessList(train), 'HARNESS not found in both files')
assert.equal(harnessList(train), harnessList(rnd), 'the HARNESS list has drifted between prepare-tickets.js and run-merge-train.js')

console.log('ok')
