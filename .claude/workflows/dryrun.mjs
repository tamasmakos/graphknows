// Dry-run harness for the workflows in this directory.
//
//   node .claude/workflows/dryrun.mjs process-issues [outdir]
//
// Stubs the workflow runtime (agent/pipeline/parallel/log/phase) and executes the
// script for real, so every prompt is rendered with real interpolation and the data
// actually flows stage to stage -- without spending a token or touching the repo.
// Each rendered prompt is written to <outdir>/NN-<label>.txt for reading.
//
// It checks what a workflow can only get wrong once it is already running:
//   - the script parses and runs to completion
//   - meta is a pure literal (no concatenation), and every phase() has a meta entry
//   - no prompt contains "undefined", "[object Object]" or an unrendered ${...}
//   - every agent() carries a label, and a schema wherever the script reads fields back
import fs from 'node:fs'
import path from 'node:path'

const name = process.argv[2] || 'process-issues'
const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))
const src = fs.readFileSync(path.join(here, `${name}.js`), 'utf8')
const outDir = process.argv[3] || path.join(here, `.dryrun-${name}`)

// -- canned agent returns ---------------------------------------------------
// Keyed by label prefix. Deliberately includes a discovery from every stage that
// can report one, so the follow-up filer's candidate list can be inspected.
const CANNED = {
  baseline: {
    ran: true,
    commands: ['docker run ... pytest'],
    failing: [{ id: 'tests/topics/test_topic_stage.py::test_cluster', summary: 'assertion on cluster count' }],
    distribution: {
      name: 'graphknows',
      version: '0.4.1',
      version_is_dynamic: true,
      version_source: 'graphknows/_version.py',
      build_backend: 'hatchling.build',
      layout: 'flat',
      requires_python: '>=3.11',
      dependencies: ['pydantic>=2.7.0', 'numpy>=1.26.0'],
      extras: ['assisted', 'srl'],
      ships_data_files: true,
    },
    notes: 'one pre-existing failure',
  },
  triage: {
    issues: [
      {
        number: 157,
        title: 'INSTANCE_OF.score is per-class, not per-entity',
        summary: 'the score is computed once per class so it cannot rank entities',
        worktree: 'C:/Users/User/Documents/dev/gk-issue-157',
        branch: 'fix/issue-157',
      },
    ],
    skipped: [{ number: 160, reason: 'too vague to implement' }],
  },
  plan: {
    root_cause: 'the score is keyed by class id, so every entity in a class shares one value',
    tasks: [
      { title: 'reproduce the per-class score', kind: 'test', instruction: 'assert two entities differ', files: ['tests/test_typing.py'] },
      { title: 'key the score per entity', kind: 'code', instruction: 'move the computation inside the loop', files: ['graphknows/symbolic/typing.py'] },
    ],
    test_strategy: 'two entities in one class must get different scores',
    api_surface: '',
    new_dependencies: [],
    out_of_scope: ['the class embedding cache is never invalidated'],
  },
  red: {
    test_id: 'tests/test_typing.py::test_instance_of_score_is_per_entity',
    command: 'pytest tests/test_typing.py::test_instance_of_score_is_per_entity',
    fails_now: true,
    failure_excerpt: 'assert 0.51 != 0.51',
    discoveries: [{ kind: 'enhancement', summary: 'no fixture builds a two-entity class', where: 'tests/conftest.py', evidence: 'every typing test hand-rolls it' }],
  },
  impl: { summary: 'moved the score computation inside the entity loop', files: ['graphknows/symbolic/typing.py'], discoveries: [{ kind: 'bug', summary: 'the sibling caller in retrieval.py has the same per-class bug', where: 'graphknows/retrieval.py:88', evidence: 'same expression, unfixed' }] },
  dogfood: {
    ran: true,
    artifact: 'evaluation/run_locomo.py --turns 20, 41 INSTANCE_OF edges',
    records_read: 41,
    verdict: 'edible',
    smells: [
      { smell: 'every INSTANCE_OF score in a class was 0.51', evidence: '{"entity":"Caroline","class":"Person","score":0.51} x14', caused_by_this_change: true },
      { smell: 'chunk text keeps the markdown fence from extraction', evidence: '{"text":"```\nCaroline met..."}', caused_by_this_change: false },
    ],
    refined: ['moved the computation inside the entity loop; the 14 rows now span 0.31-0.72'],
    still_wrong: [],
    discoveries: [{ kind: 'bug', summary: 'extracted chunk text keeps markdown fences', where: 'graphknows/ingest/extract.py', evidence: 'every fenced turn stores the fence' }],
    notes: 'tasted twice',
  },
  review: {
    resolves_issue: true,
    verdict: 'the diff moves the computation per entity',
    findings: [],
    scope_violations: [],
    packaging: { touches_packaging: true, version_touched: false, release_impact: 'patch', changelog_line: 'INSTANCE_OF.score is now computed per entity' },
    out_of_scope_defects: [{ summary: 'ONTOLOGY_CLASS nodes are embedded but unused', file: 'graphknows/symbolic/index.py', failure_scenario: '1437 embeddings serve 9 lookups' }],
  },
  verify: { green: true, red_to_green: true, checks_run: ['ruff', 'mypy', 'pytest'], fixed: [], unresolved: [], new_failures: [], metadata_changes: [], needs_human: false, notes: 'green' },
  package: { built: true, artifacts: ['graphknows-0.4.1.whl'], twine_check: true, wheel_complete: true, installs_clean: true, imports_from_site_packages: true, repro_passes_installed: true, worktree_clean: true, problems: ['py.typed is absent from the wheel'], discoveries: [], notes: '' },
  docs: {
    user_visible: true,
    edited: [{ file: 'docs/architecture.md', change: 'INSTANCE_OF.score is described as per entity' }],
    stale_found: [{ file: 'docs/topics.md', summary: 'names a grader flag that was removed two releases ago' }],
    broken_links: [],
    notes: 'one page corrected',
  },
  // One well-formed body and one missing "## Verify", so the dry run exercises both sides of
  // checkFollowupBody rather than only the happy path.
  followups: {
    filed: [
      {
        number: 162,
        title: 'py.typed is absent from the wheel',
        kind: 'bug',
        body: '## Symptom\npy.typed is missing from the built wheel\n## Where\npyproject.toml\n## Concrete failure scenario\nmypy on an installed copy finds no marker\n## Verify\n`python -m zipfile -l dist/*.whl | grep py.typed` lists it\n## Out of scope\nthe rest of the wheel contents\n',
      },
      {
        number: 163,
        title: 'chunk text keeps markdown fences',
        kind: 'bug',
        body: '## Symptom\nfences survive extraction\n## Where\ngraphknows/ingest/extract.py\n## Concrete failure scenario\na fenced turn stores the fence\n## Out of scope\nthe parser rewrite\n',
      },
    ],
    commented: [],
    dropped: [],
  },
  ship: { shipped: true, pr_url: 'https://github.com/tamasmakos/graphknows/pull/163', branch: 'fix/issue-157', action: 'created', notes: 'draft opened' },
  // DRYRUN_CI=fixed-and-pushed exercises the retry loop; =still-red / =no-run exercise the
  // paths that must refuse to merge. Default green, so the happy path runs end to end.
  ci: {
    outcome: process.env.DRYRUN_CI || 'green',
    run_id: '1699',
    run_url: 'https://github.com/tamasmakos/graphknows/actions/runs/1699',
    conclusion: process.env.DRYRUN_CI ? 'failure' : 'success',
    failing_jobs: process.env.DRYRUN_CI ? ['checks / mypy: embedder.py:95 arg-type'] : [],
    fixed: [],
    commit: 'abc1234',
    discoveries: [],
    notes: 'stub',
  },
  // -- prepare-tickets ------------------------------------------------------
  intake: {
    topic: 'should window chunks carry their own episodic index?',
    kind: 'feature',
    frame: 'The chunk writer is the only path that persists a chunk; do not change the public store() signature.',
    out_of_scope: ['back-filling chunks written before this lands'],
    done_when: ['a temporal query returns chunks in instant order without reading the ts string'],
    severity: '',
    approach_settled: false,
    clarifications: [
      { question: 'Should two chunks written in the same instant tie-break by insertion order, or may the order vary?', default_if_unanswered: 'insertion order' },
    ],
    slug: 'window-chunk-episodic-index',
    notes: 'form complete apart from the tie-break',
  },
  screen: {
    verdict: 'go',
    reason: 'no episodic index is attached at write time anywhere; the ts string is the only temporal signal',
    evidence: 'graphknows/storage/chunks.py:88',
    searched: ['graphknows/symbolic/temporal.py', 'graphknows/storage/', '.out-of-scope/', 'docs/adr/'],
    duplicate_of: 0,
  },
  research: {
    findings: [{ claim: 'window chunks carry no episodic index', source: 'graphknows/storage/chunks.py:88', so_what: 'temporal recall falls back to the string timestamp' }],
    open_questions: ['does the answerer ever read the ts field'],
    notes: 'one lane, one finding',
  },
  design: {
    approach: 'attach the episodic index to the chunk at write time',
    seam: 'the chunk writer, because every path already funnels through it',
    interface: 'store(chunk) unchanged; the index is derived, not passed',
    why: 'callers learn nothing new and get temporal recall',
    costs: ['one more write per chunk'],
    blast_radius: ['graphknows/storage/chunks.py', 'graphknows/symbolic/temporal.py'],
    new_dependencies: [],
    prototype_question: 'does serial conditioning survive when two chunks share an instant',
  },
  synthesis: {
    chosen: 'attach the episodic index at write time, in the chunk writer',
    rationale: 'the smallest design that the research supports',
    grafted: ['the deep-module variant of the interface'],
    rejected: [{ approach: 'seam-moving', reason: 'blast radius across every reader for no recall gain' }],
    prototype_question: 'does serial conditioning survive when two chunks share an instant',
    depends_on: ['the chunk writer is the only path that persists a chunk'],
    risks: ['back-filling existing chunks is not covered'],
    adr_written: true,
    adr_path: 'docs/adr/0002-episodic-index-on-window-chunks.md',
    adr_skipped_why: '',
  },
  prototype: {
    ran: true,
    verdict: 'serial conditioning holds; ties break by insertion order',
    what_it_showed: 'two chunks at one instant still decode in order',
    design_changes: [],
    branch: 'proto/episodic-chunk',
    run_command: 'python scratch/proto_episodic.py',
    snippet: 'Instance(ts, seq) sorts by (ts, seq)',
  },
  spec: {
    title: 'Window chunks carry their own episodic index',
    body: '## User Scenarios & Testing\n### User Story 1 - Recall by instant (Priority: P1)\nx\n## Requirements\n- **FR-001**: the chunk writer MUST derive an episodic index.\n## Success Criteria\n- **SC-001**: recall returns the right chunk for a known instant in 100% of cases.\n## Implementation Decisions\nRecorded in docs/adr/0004-episodic-index.md.\n## Testing Decisions\nTested at the chunk writer.\n',
    seams: ['the chunk writer'],
    success_criteria: ['recall returns the right chunk for a known instant in 100% of cases'],
    out_of_scope: ['back-filling existing chunks'],
  },
  critic: {
    verdict: 'fixable',
    must_fix: [{ problem: 'the spec never says what happens to chunks already stored', fix: 'state that back-fill is out of scope and why' }],
    missing: ['nobody measured recall before the change'],
    contradictions: [],
  },
  contest: { refuted: false, why: 'read graphknows/chunking/writer.py -- it is the only caller that persists', correction: '' },
  'tickets:file': {
    filed: true,
    spec_issue: 201,
    spec_url: 'https://github.com/tamasmakos/graphknows/issues/201',
    tickets: [
      { index: 0, number: 202, url: 'https://github.com/tamasmakos/graphknows/issues/202' },
      { index: 1, number: 203, url: 'https://github.com/tamasmakos/graphknows/issues/203' },
    ],
    notes: 'two slices',
  },
  'tickets:draft': {
    spec_body: '## User Scenarios & Testing\n### User Story 1 - Recall by instant (Priority: P1)\nx\n## Requirements\n- **FR-001**: the chunk writer MUST derive an episodic index.\n## Success Criteria\n- **SC-001**: recall returns the right chunk for a known instant in 100% of cases.\n## Implementation Decisions\nRecorded in docs/adr/0004-episodic-index.md.\n## Testing Decisions\nTested at the chunk writer.\n',
    tickets: [
      {
        title: 'derive an episodic index in the chunk writer',
        delivers: 'a stored chunk is retrievable by instant',
        files_named: ['graphknows/storage/chunks.py'],
        body:
          '## Parent\nthe parent spec issue\n## What to build\nDerive the index in graphknows/storage/chunks.py at write time.\n## Acceptance criteria\n- [ ] WHEN a chunk is stored THE SYSTEM SHALL make it retrievable by its instant\n## Verify\n`pytest tests/storage/test_chunks.py -q` passes\n## Out of scope\nBack-filling chunks written before this lands.\n',
        verify: 'pytest tests/storage/test_chunks.py -q',
        expected_lines: 60,
        blocked_by_index: [],
        acceptance: ['a chunk written today is found by its instant'],
      },
      {
        title: 'rank temporal candidates by episodic index',
        delivers: 'a temporal query prefers the right instant',
        body:
          '## Parent\nthe parent spec issue\n## What to build\nRank temporal candidates in graphknows/symbolic/temporal.py.\n## Acceptance criteria\n- [ ] WHEN a temporal query runs THE SYSTEM SHALL rank the nearer instant first\n## Verify\n`python evaluation/harness.py --temporal` reports recall at or above the baseline\n## Out of scope\nTuning the fusion weights.\n',
        verify: 'python evaluation/harness.py --temporal',
        expected_lines: 40,
        blocked_by_index: [0],
        acceptance: ['the harness recall improves'],
      },
    ],
    wide_refactor: false,
    notes: 'two slices',
  },
  'wrap-up': {
    issues: [
      { number: 202, labels: ['enhancement', 'ready-for-agent'] },
      { number: 203, labels: ['enhancement', 'ready-for-agent'] },
    ],
    repaired: [],
    feature_dir: 'specs/001-window-chunk-episodic-index',
    notes: 'stub',
  },
  merge: { merged: true, method: 'squash', merge_commit: 'def5678', tree_matches_ci: true, issue_closed: true, post_merge_run_url: 'https://github.com/tamasmakos/graphknows/actions/runs/1700', post_merge_green: true, notes: 'squashed' },
  roster: {
    spec_title: 'Window chunks carry their own episodic index',
    spec_open: true,
    children: [
      { number: 202, title: 'derive an episodic index in the chunk writer', state: 'OPEN', labels: ['enhancement', 'ready-for-agent'], blocked_by: [], expected_lines: 60, files_named: ['graphknows/storage/chunks.py'] },
      { number: 203, title: 'rank temporal candidates by episodic index', state: 'OPEN', labels: ['enhancement', 'ready-for-agent'], blocked_by: [202], expected_lines: 40, files_named: ['graphknows/symbolic/temporal.py'] },
      { number: 204, title: 'gated straggler', state: 'OPEN', labels: ['enhancement', 'needs-triage'], blocked_by: [] },
      { number: 205, title: 'add 3.13 to the classifiers', state: 'OPEN', labels: ['enhancement', 'ready-for-agent'], blocked_by: [], expected_lines: 2, sizing: 'measured', sizing_evidence: 'pyproject.toml classifiers list, one line added', files_named: ['pyproject.toml'] },
      { number: 206, title: 'fix a stale path in the deprecation table', state: 'OPEN', labels: ['enhancement', 'ready-for-agent'], blocked_by: [], expected_lines: 4, sizing: 'measured', sizing_evidence: 'one path string in a markdown table', files_named: ['docs/versioning.md'] },
    ],
    notes: '',
  },
  recheck: { state: 'OPEN', labels: ['enhancement', 'ready-for-agent'] },

  // -- express-lane ---------------------------------------------------------
  'express:cut': {
    ready: true,
    reused: false,
    base_sha: 'aaa1111',
    tickets: [
      {
        number: 205,
        title: 'add 3.13 to the classifiers',
        state: 'OPEN',
        labels: ['enhancement', 'ready-for-agent'],
        body: '## Parent\n#200\n## What to build\nAdd the 3.13 classifier to pyproject.toml.\n## Acceptance criteria\n- [ ] WHEN the wheel metadata is read THE SYSTEM SHALL list Python 3.13\n## Verify\n`uv build` then `twine check dist/*` passes\n## Out of scope\nThe CI matrix.\n',
      },
      {
        number: 206,
        title: 'fix a stale path in the deprecation table',
        state: 'OPEN',
        labels: ['enhancement', 'ready-for-agent'],
        body: '## Parent\n#200\n## What to build\nCorrect the module path in the deprecation table.\n## Acceptance criteria\n- [ ] WHEN a reader follows the named path THE SYSTEM SHALL resolve to a module that exists\n## Verify\n`pytest tests/test_docs_links.py -q` passes\n## Out of scope\nThe rest of the page.\n',
      },
      { number: 207, title: 'gated one', state: 'OPEN', labels: ['enhancement', 'needs-triage'], body: '## Parent\n#200\n' },
    ],
    notes: 'worktree cut',
  },
  'express:commit': { ejected: false, committed: true, sha: 'bbb2222', subject: 'add 3.13 classifier', changed_lines: 2, files: ['pyproject.toml'], checks_green: true, notes: '' },
  'express:ship': { pushed: true, pr_number: 950, pr_url: 'https://github.com/tamasmakos/graphknows/pull/950', gate_passed: true, notes: '' },
  'express:land': { ci: 'green', merged: true, merge_commit: 'ccc3333', closed: [205, 206], still_open: [], post_merge_green: true, failing_check: '', notes: 'rebase-merged' },
}

// Canned CHILD WORKFLOW results for the `workflow()` global, keyed by workflow name.
// The process-issues shape mirrors its real return: the train reads summary[0].
const WORKFLOW_CANNED = {
  'process-issues': (wargs) => ({
    summary: [
      {
        issue: wargs?.issues?.[0] ?? 0,
        title: 'stub',
        branch: `fix/issue-${wargs?.issues?.[0] ?? 0}`,
        worktree: `C:/Users/User/Documents/dev/gk-issue-${wargs?.issues?.[0] ?? 0}`,
        shipped: true,
        pr_url: `https://github.com/tamasmakos/graphknows/pull/${900 + (wargs?.issues?.[0] ?? 0)}`,
        ci: 'green',
        merged: true,
        merge_commit: 'abc1234',
        post_merge_green: true,
        release_impact: 'patch',
        changelog_line: 'stub change',
        filed: [],
        needs_human: false,
        notes: 'stub',
      },
    ],
    skipped: [],
  }),
  // The express lane returns landed/ejected rather than a summary: the train reads which
  // tickets rode it and re-queues whatever turned out bigger than its estimate claimed.
  'express-lane': (wargs) => {
    const issues = wargs?.issues || []
    const ejected = issues.length > 1 ? [{ number: issues[issues.length - 1], reason: 'needed 90 lines, over the 50 this lane takes' }] : []
    const landed = issues.filter((n) => !ejected.some((e) => e.number === n))
    return {
      lane: 'express',
      branch: `express/${issues.join('-')}`,
      pr: 950,
      pr_url: 'https://github.com/tamasmakos/graphknows/pull/950',
      ci: 'green',
      merged: true,
      merge_commit: 'e0e0e0e',
      post_merge_green: true,
      landed,
      // A batch that committed and then failed to land; empty on this green stub, but the
      // train reads it and re-queues each number as a full car.
      red_batch: [],
      ejected,
      refused: [],
      report: 'docs/agents/runs/express-lane-stub.md',
      next_step: 'stub',
    }
  },
}

const calls = []
const logs = []
const phases = []
let current = null

const agent = async (prompt, opts = {}) => {
  const label = opts.label || '(no label)'
  calls.push({ label, phase: opts.phase || current, prompt, schema: !!opts.schema, model: opts.model })
  const key = Object.keys(CANNED).find((k) => label.startsWith(k))
  return key ? structuredClone(CANNED[key]) : `stub output for ${label}`
}
const pipeline = async (items, ...stages) => {
  const out = []
  for (let i = 0; i < items.length; i++) {
    let v = items[i]
    for (const stage of stages) v = await stage(v, items[i], i)
    out.push(v)
  }
  return out
}
const parallel = async (thunks) => Promise.all(thunks.map((t) => Promise.resolve().then(t).catch(() => null)))
const workflowCalls = []
const workflow = async (ref, wargs) => {
  const wname = typeof ref === 'string' ? ref : ref?.scriptPath || '(scriptPath)'
  workflowCalls.push({ name: wname, args: wargs })
  calls.push({ label: `workflow:${wname}`, phase: current, prompt: `child workflow ${wname} with args ${JSON.stringify(wargs)}`, schema: false })
  const canned = WORKFLOW_CANNED[wname]
  return canned ? structuredClone(canned(wargs)) : null
}
const log = (m) => logs.push(m)
const phase = (t) => {
  current = t
  phases.push(t)
}

// -- run --------------------------------------------------------------------
const body = src.replace(/^export const meta/m, 'const meta')
const factory = new Function(
  'agent', 'pipeline', 'parallel', 'log', 'phase', 'args', 'budget', 'workflow',
  `return (async () => { ${body} \n})()`,
)

const args = JSON.parse(process.env.DRYRUN_ARGS || 'null') || undefined
const budget = { total: null, spent: () => 0, remaining: () => Infinity }

let result, failure
try {
  result = await factory(agent, pipeline, parallel, log, phase, args, budget, workflow)
} catch (e) {
  failure = e
}

// -- write the prompts out --------------------------------------------------
fs.rmSync(outDir, { recursive: true, force: true })
fs.mkdirSync(outDir, { recursive: true })
calls.forEach((c, i) => {
  const safe = c.label.replace(/[^\w.#-]/g, '_')
  fs.writeFileSync(path.join(outDir, `${String(i + 1).padStart(2, '0')}-${safe}.txt`), c.prompt)
})
fs.writeFileSync(path.join(outDir, 'result.json'), JSON.stringify({ result, logs }, null, 2))

// -- checks -----------------------------------------------------------------
const problems = []
if (failure) problems.push(`the script threw: ${failure.message}\n${failure.stack?.split('\n').slice(1, 4).join('\n')}`)

// CRLF-tolerant: a plain indexOf('\n}\n') misses '\r\n}\r\n' on a Windows-checkout script,
// so metaBlock became the WHOLE FILE and the purity check below tripped on the first
// legitimate string concatenation anywhere in it.
const metaStart = src.indexOf('export const meta')
const metaEnd = src.slice(metaStart).search(/\r?\n\}\r?\n/)
const metaBlock = src.slice(metaStart, metaEnd === -1 ? undefined : metaStart + metaEnd)
if (/'\s*\+\s*$|'\s*\+\s*'/m.test(metaBlock)) problems.push('meta is not a pure literal: it contains string concatenation')

const declared = [...metaBlock.matchAll(/title: '([^']+)'/g)].map((m) => m[1])
const used = [...new Set([...phases, ...calls.map((c) => c.phase)])].filter(Boolean)
for (const u of used) if (!declared.includes(u)) problems.push(`phase "${u}" has no entry in meta.phases`)

for (const c of calls) {
  if (c.label === '(no label)') problems.push(`an agent() call has no label (phase ${c.phase})`)
  for (const bad of ['undefined', '[object Object]', 'null']) {
    if (c.prompt.includes(bad)) problems.push(`${c.label}: prompt contains "${bad}" -- an interpolation went missing`)
  }
  if (/\$\{/.test(c.prompt)) problems.push(`${c.label}: prompt contains an unrendered \${...}`)
}

// -- report -----------------------------------------------------------------
console.log(`\n${name}: ${calls.length} agent call(s) across ${new Set(calls.map((c) => c.phase)).size} phase(s)\n`)
for (const c of calls) {
  console.log(`  ${String(c.prompt.length).padStart(6)} chars  ${c.schema ? 'schema' : '  text'}  ${c.phase || '-'} / ${c.label}`)
}
console.log('\nlog output:')
for (const l of logs) console.log(`  ${l}`)
console.log(`\nprompts written to ${outDir}`)

if (problems.length) {
  console.log(`\nPROBLEMS (${problems.length}):`)
  for (const p of problems) console.log(`  - ${p}`)
  process.exit(1)
}
console.log('\nno problems found.')
