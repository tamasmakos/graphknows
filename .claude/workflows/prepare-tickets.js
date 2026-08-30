export const meta = {
  name: 'prepare-tickets',
  description: 'A filed request form -> screened -> researched -> designed -> specced -> tracer-bullet tickets an agent can start on',
  whenToUse:
    'Everything between a request and an implementable ticket, done as a technical project owner and a principal engineer would do it. Takes the number of a request issue filed from .github/ISSUE_TEMPLATE/request.yml and produces the thing process-issues consumes: a spec issue plus tracer-bullet tickets with real blocking edges, the ones an agent can start on labelled ready-for-agent and only the few genuinely waiting on a person gated. Research and design are skipped when the request already settles the approach. It is the /research -> /prototype -> /codebase-design -> /to-spec -> /to-tickets sequence with the interviews replaced by an adversarial critic and by explicit [NEEDS CLARIFICATION] markers, because a workflow cannot stop and ask.',
  phases: [
    { title: 'Intake', detail: 'read the request form, derive the frame, mark what it does not settle', model: 'sonnet' },
    { title: 'Screen', detail: 'go/no-go: already built, previously rejected, or outside the frame', model: 'sonnet' },
    { title: 'Research', detail: 'four lanes: primary sources, the code today, prior art in-repo, constraints', model: 'sonnet' },
    { title: 'Design', detail: 'three independent approaches, deliberately different stances', model: 'opus' },
    { title: 'Synthesis', detail: 'pick one, graft the best of the others, write the ADR', model: 'opus' },
    { title: 'Contest', detail: 'one skeptic per finding the winning design leans on, told to refute it', model: 'sonnet' },
    { title: 'Prototype', detail: 'throwaway code on a throwaway branch, only when a design question needs running code', model: 'sonnet' },
    { title: 'Spec', detail: 'draft the spec from the chosen design (no filing yet)', model: 'opus' },
    { title: 'Critic', detail: 'adversarial read of the draft against the research brief', model: 'opus' },
    { title: 'Tickets', detail: 'draft the slices, check the bodies in JS, then file the spec and the tickets', model: 'opus' },
    { title: 'Report', detail: 'one wrap-up agent: repair labels, post clarifications, mirror the spec to specs/, write the run report', model: 'sonnet' },
  ],
}

// args: { request: 300,  // the number of a filed request issue -- the normal entry point
//         topic?: 'the question, in one sentence',  // escape hatch: run with no filed form
//         notes?: 'anything already decided', angles?: 4, designs?: 3, maxTickets?: 6,
//         prototype?: 'auto', file?: true, gateAll?: false, depth?: 'auto',
//         report?: true, reportPath?: 'docs/agents/runs/x.md',
//         slug?: 'window-chunk-recall', date?: '2026-08-26', kind?: 'enhancement', runLabel?: 'nightly' }
//
// `args` is an OBJECT, not prose. THE ENTRY POINT IS A REQUEST ISSUE: someone (or, later, a
// create-request run) files .github/ISSUE_TEMPLATE/request.yml, and this workflow is handed
// its number. That issue is the root of the chain every later artifact hangs off:
//
//   request -> spec -> tickets -> PR (Closes #ticket) -> commit -> CHANGELOG
//
// `topic` remains for a run with no filed form -- testing, or a question too small to file.
// It skips the Screen stage's prior-rejection check, because there is no request to reject.
//
// The handover to process-issues is the return value: { spec_issue, tickets: [<numbers>] }.
//
// WHAT COMES BACK TO A PERSON, and nothing else:
//   1. [NEEDS CLARIFICATION] markers -- one comment on the request issue. The pipeline marks
//      what the form does not settle instead of guessing it, which is spec-kit's mechanism
//      and the reason a wrong assumption does not reach an implementer.
//   2. A gated ticket -- one of the four reasons in docs/agents/triage-labels.md.
//   3. A Screen no-go -- already built, or previously rejected.
//
// The gate is SELECTIVE, and provenance decides its default. The property "needs-triage"
// protects is that an agent must not feed itself work it invented -- a claim about ORIGIN,
// not about who types the label. Work tracing to a human-filed request defaults to
// ready-for-agent; work an agent found for itself (process-issues' Followups, agent-found)
// is gated forever. The spec issue itself is always gated: it is a document, not work, and
// the implementing loop must never pick it up as a slice. { gateAll: true } restores the old
// blanket behaviour. Do not widen the gating criteria to buy comfort: a run that gates
// everything has decided nothing, and the label stops meaning anything.
//
// Prove it cheaply first:
//   1. `node .claude/workflows/dryrun.mjs prepare-tickets` -- renders every prompt, spends nothing.
//   2. { request: <n>, file: false } -- a real run that files nothing; you read the draft.
//   3. { request: <n> } -- the full run, spec issue + tickets.
const opt = args || {}
const REQUEST = Number.isInteger(opt.request) ? opt.request : 0
let TOPIC = typeof opt.topic === 'string' ? opt.topic.trim() : ''
const NOTES = opt.notes || 'nothing beyond the request itself'
const ANGLES = Math.min(opt.angles || 4, 4)
const DESIGNS = Math.min(opt.designs || 3, 3)
const MAX_TICKETS = opt.maxTickets || 6
const PROTO = opt.prototype || 'auto' // 'auto' = only when the synthesis names a question code must settle
const FILE = opt.file !== false
const KIND = opt.kind === 'bug' ? 'bug' : 'enhancement'
// The escape hatch for a topic the launcher does not trust yet: file everything gated, as
// this workflow used to unconditionally. The default is the other way round -- a ticket is
// ready unless something specific is waiting on a person.
const GATE_ALL = opt.gateAll === true
// 'auto' reads the depth off the request: a form that already names the approach or the
// files has had its design done by a person, and re-deriving it is what made one run spend
// five opus stages rebuilding a frame a grilling session had already settled. 'full' forces
// research and design; 'shallow' forces the skip.
const DEPTH = ['auto', 'full', 'shallow'].includes(opt.depth) ? opt.depth : 'auto'
const RUN = opt.runLabel || 'an automated prepare-tickets run'
// The run report. The return value lands in one session's context and dies with it; the
// report is the copy still there tomorrow, and it is where a reader finds out what lost.
const DO_REPORT = opt.report !== false
// The design record is an ADR under docs/adr/, COMMITTED. It used to be a note in .scratch/,
// which is gitignored -- so nobody reading the spec issue could open it, so the spec had to
// restate every fact in prose, so one spec body reached 56 KB of GitHub's 65,536-char cap.
// Committing it breaks that chain: the spec links to the "why" instead of carrying it.
// See docs/adr/0001-record-architecture-decisions.md.
let SLUG =
  opt.slug ||
  (TOPIC || `request-${REQUEST}`).toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .split('-')
    .slice(0, 6)
    .join('-')
const ADR_DIR = 'docs/adr'

// The gate. Identical vocabulary to process-issues on purpose: one label means one thing
// across both halves of the loop. docs/agents/triage-labels.md owns the rules; these are
// just the strings, and gates.test.mjs asserts they have not drifted between the workflows.
const GATE_LABEL = 'needs-triage'
const READY_LABEL = 'ready-for-agent'

// .claude/ is gitignored, so skills are read from the main checkout by absolute path.
const MAIN_REPO = opt.repo || 'C:/Users/User/Documents/dev/graphknows'
// Where an agent's own output goes. One directory, one ignore rule, and it sits beside the
// rules agents READ (definition-of-done.md, triage-labels.md) rather than off in a scratch
// tree of its own -- so "what does an agent read here, and what did it write" is one folder.
// docs/agents/runs/ is gitignored; the files beside it are not, deliberately: a rule an
// agent cannot open from a fresh clone is a rule that gets restated into every ticket
// instead, which is the bloat this whole pipeline was rebuilt to stop.
const RUNS_DIR = 'docs/agents/runs'
const skill = (path, why) => `Read ${MAIN_REPO}/.claude/skills/${path} and follow it: ${why}`

const GH = `Use \`gh\`. If it is not on PATH, fall back to "${opt.ghPath || '/c/Program Files/GitHub CLI/gh.exe'}" (quote it).`

const RUN_ON = { model: 'sonnet', effort: 'medium' }
// Design and spec work is where a wrong answer is expensive and invisible: every ticket
// downstream implements it perfectly. Research and filing are mechanical.
const DEEP = { model: 'opus' }
const call = (prompt, opts = {}) => agent(prompt, { ...RUN_ON, ...opts })

const GRAPHIFY = [
  `Orientation: this repo has a knowledge graph. Before grepping blind, run these from ${MAIN_REPO} (graphify-out is gitignored, so it exists only there):`,
  '  `graphify query "<your question>"`   -- broad context, BFS over the graph',
  '  `graphify path "A" "B"`              -- how two concepts connect',
  '  `graphify explain "<symbol>"`        -- plain-language explanation of a node',
  'Use it to find where a thing lives and what touches it, then confirm against the actual file before acting on it. If the CLI is missing, skip it and read the code.',
].join('\n')

const HOUSE = [
  'Repo rules (CLAUDE.md) are binding and they constrain the DESIGN, not just the code:',
  'pythonic OOP, type hints, one thing per function, composition over inheritance, small diffs,',
  'no speculative abstraction, no interface with one implementation.',
  'This repo is modelled on the Tensor Brain, summarised in CLAUDE.md: a subsymbolic representation layer and a',
  'symbolic index layer, and a channel that only writes is dead weight. When a module maps onto a',
  'Tensor Brain concept, name it after that concept.',
].join(' ')

const PACKAGING = [
  'THIS REPO IS A DISTRIBUTED PYTHON PACKAGE, so a design carries distribution consequences:',
  '  - a new third-party import is a new entry in [project.dependencies] that every user must install. Prefer a design that needs no new dependency; if one is unavoidable, say which pyproject table it belongs in and why the stdlib will not do.',
  '  - non-.py files (data, ontologies) do not ship unless declared to the build backend.',
  '  - src-layout-style rules apply: code that only works from the repo working directory works in the suite and breaks for installed users.',
  '  - never propose a version bump. Versions are bumped once, at release.',
].join('\n')

const EVIDENCE = [
  'Evidence bar, strict:',
  '  A CLAIM about the code is only usable if you read the file and can cite path:line.',
  '  A CLAIM about an external system is only usable if it comes from a primary source -- official docs, the spec, the library source -- not a secondary write-up.',
  '  Anything you could not verify goes under open_questions, never into findings dressed as fact.',
].join('\n')

// -- schemas -------------------------------------------------------

const INTAKE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['topic', 'kind', 'frame', 'approach_settled', 'clarifications', 'slug'],
  properties: {
    topic: { type: 'string', description: 'the request restated as one sentence' },
    kind: { type: 'string', enum: ['feature', 'business', 'bug', 'chore'] },
    frame: { type: 'string', description: 'what the requester has already decided -- constraints and settled choices, verbatim in substance' },
    out_of_scope: { type: 'array', items: { type: 'string' }, maxItems: 10 },
    done_when: { type: 'array', items: { type: 'string' }, maxItems: 8, description: "the requester's own success test, one line each" },
    severity: { type: 'string', enum: ['', 'S1', 'S2', 'S3', 'S4'], description: 'bugs only; empty otherwise. Assigned from the consequence per docs/agents/triage-labels.md' },
    // The whole depth decision, in one boolean, read off the request rather than chosen by
    // a person. A form that names the mechanism has had its design done already.
    approach_settled: { type: 'boolean', description: 'true if the request already names the approach, the mechanism or the files -- research and design would re-derive a decision a person made' },
    clarifications: {
      type: 'array',
      maxItems: 6,
      description: 'what the form does not settle. EMPTY IS THE GOOD OUTCOME -- only what would change the work.',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['question', 'default_if_unanswered'],
        properties: {
          question: { type: 'string', description: 'one sentence, answerable without reading code' },
          default_if_unanswered: { type: 'string', description: 'what the pipeline will assume meanwhile' },
        },
      },
    },
    slug: { type: 'string', description: 'at most six lowercase hyphenated words' },
    notes: { type: 'string' },
  },
}

const SCREEN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'reason', 'searched'],
  properties: {
    verdict: { type: 'string', enum: ['go', 'no-go'] },
    reason: { type: 'string', description: 'for a no-go, which of the three checks fired and what it found; for a go, one line on what you confirmed is absent' },
    evidence: { type: 'string', description: 'path:line for "already built", the .out-of-scope filename for a prior rejection, the document and line for a frame conflict' },
    searched: { type: 'array', items: { type: 'string' }, maxItems: 12, description: 'where you actually looked -- a no-go that cannot say this is a guess' },
    duplicate_of: { type: 'integer', description: 'an open issue already covering this; 0 if none. Not a no-go on its own.' },
  },
}

const RESEARCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings', 'open_questions', 'notes'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['claim', 'source', 'so_what'],
        properties: {
          claim: { type: 'string', description: 'one verified fact, stated flatly' },
          source: { type: 'string', description: 'path:line, a URL, or a command and its output -- something checkable' },
          so_what: { type: 'string', description: 'what this constrains or enables for the topic' },
        },
      },
    },
    open_questions: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const DESIGN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['approach', 'seam', 'interface', 'why', 'costs', 'blast_radius'],
  properties: {
    // Every free-text field is capped. Uncapped, a design agent writes until it trips the
    // structured-output limit and dies -- and a dead stance is silent: the synthesis just
    // runs on fewer candidates and reads exactly like a run where all of them survived.
    approach: { type: 'string', description: 'the design in 3-6 sentences. Hard ceiling: 6.' },
    seam: { type: 'string', description: 'where the interface lives, and why there. At most 3 sentences.' },
    interface: { type: 'string', description: 'the public surface a caller must learn: signatures, invariants, error modes. At most 15 lines; signatures, not prose around them.' },
    why: { type: 'string', description: 'what makes this deep -- behaviour hidden per unit of interface learned. At most 3 sentences.' },
    costs: { type: 'array', items: { type: 'string' }, maxItems: 6, description: 'what this design makes worse or harder, one line each' },
    blast_radius: { type: 'array', items: { type: 'string' }, maxItems: 10, description: 'modules and call sites this touches, one line each' },
    new_dependencies: { type: 'array', items: { type: 'string' }, maxItems: 6 },
    prototype_question: { type: 'string', description: 'the one question only running code can settle, or an empty string' },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['chosen', 'rationale', 'grafted', 'rejected', 'adr_written'],
  properties: {
    chosen: { type: 'string', description: 'the design being taken forward, restated in full -- later stages see only this' },
    rationale: { type: 'string' },
    grafted: { type: 'array', items: { type: 'string' }, description: 'ideas taken from the runners-up' },
    rejected: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['approach', 'reason'],
        properties: { approach: { type: 'string' }, reason: { type: 'string' } },
      },
    },
    prototype_question: { type: 'string', description: 'the one question only running code can settle; empty string if the design is decidable on paper' },
    // The findings the choice actually rests on. A skeptic is pointed at each of these,
    // because a wrong fact here reaches every ticket the run files.
    depends_on: {
      type: 'array',
      description: 'the research findings this choice DEPENDS ON -- the ones that, if false, would make you pick differently. Quote each claim as it appears in the brief. Load-bearing facts only, not everything you read.',
      items: { type: 'string' },
    },
    risks: { type: 'array', items: { type: 'string' } },
    // An ADR only when the decision clears Nygard's bar: structure, non-functional
    // characteristics, dependencies, interfaces, construction techniques. Most requests do
    // not, and a directory of records for routine choices is how the filter stops working.
    adr_written: { type: 'boolean', description: 'true if an ADR file was written; false when the decision is not architecturally significant' },
    adr_path: { type: 'string', description: 'the path written, or empty' },
    adr_skipped_why: { type: 'string', description: 'when adr_written is false, one line on why the decision is not architecturally significant' },
  },
}

// One skeptic per load-bearing finding. Cheap, and pointed at the small set that can
// still do damage -- refuting every finding across four lanes would cost more than the
// designs did and most of what it checked never influenced the outcome.
const CONTEST_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['refuted', 'why'],
  properties: {
    refuted: { type: 'boolean', description: 'true only if you checked and the claim is actually WRONG -- not merely unproven' },
    why: { type: 'string', description: 'what you did to check it, and what you found' },
    correction: { type: 'string', description: 'what is true instead, if you can establish it' },
  },
}

const PROTO_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ran', 'verdict', 'what_it_showed'],
  properties: {
    ran: { type: 'boolean' },
    verdict: { type: 'string', description: 'the answer to the question the prototype was built to settle' },
    what_it_showed: { type: 'string' },
    design_changes: { type: 'array', items: { type: 'string' }, description: 'what the chosen design must change because of this' },
    branch: { type: 'string' },
    run_command: { type: 'string' },
    snippet: { type: 'string', description: 'the decision-rich part only -- a state machine, a schema, a type shape -- or an empty string' },
  },
}

const SPEC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'body', 'seams', 'success_criteria'],
  properties: {
    title: { type: 'string' },
    body: { type: 'string', description: 'the full spec in Markdown, ready to be an issue body' },
    seams: { type: 'array', items: { type: 'string' }, description: 'the seams the feature will be tested at, highest first' },
    // spec-kit's SC-### line. NOT the same thing as a user story's acceptance scenario: a
    // scenario says the feature works, a success criterion is the measurable outcome that
    // says it was worth building. speckit-analyze keys its coverage checks on these, so an
    // empty list makes that half of the analysis silently vacuous.
    success_criteria: {
      type: 'array',
      items: { type: 'string' },
      description: 'measurable outcomes, one per SC-### line, technology-agnostic -- a number, a rate or a bound someone could check after the change lands. Exclude business KPIs nobody here can measure.',
    },
    out_of_scope: { type: 'array', items: { type: 'string' } },
  },
}

const CRITIC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'must_fix', 'cut', 'missing'],
  properties: {
    verdict: { type: 'string', enum: ['ready', 'fixable', 'not-ready'] },
    // The only subtractive field in the pipeline. Research appends findings, the design
    // appends decisions, the contest appends caveats, the critic appends must-fixes --
    // five places that add and, without this, none that removes. That asymmetry is what
    // turns "collapse the Dockerfile" into twenty-six acceptance criteria.
    cut: {
      type: 'array',
      items: { type: 'string' },
      maxItems: 12,
      description: 'sections, requirements or acceptance criteria to DELETE from the spec, quoted well enough to find them',
    },
    must_fix: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['problem', 'fix'],
        properties: { problem: { type: 'string' }, fix: { type: 'string' } },
      },
    },
    missing: { type: 'array', items: { type: 'string' }, description: 'a lane not run, a claim unverified, a source unread' },
    contradictions: { type: 'array', items: { type: 'string' }, description: 'places the spec contradicts a research finding' },
  },
}

// Drafting and filing are two agents, so there are two schemas. The draft carries the
// BODIES -- the only contract the process-issues half ever reads -- and a deterministic
// check runs over them before anything is created. A check can only run on text that has
// not been filed yet; checking afterwards means patching a known-bad issue with `gh issue
// edit` and leaving that trail on every run.
const DRAFT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['spec_body', 'tickets'],
  properties: {
    spec_body: {
      type: 'string',
      description: 'the final spec body in Markdown, with the critic must-fix items already folded in -- this exact text is what gets filed',
    },
    tickets: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'delivers', 'body', 'blocked_by_index'],
        properties: {
          title: { type: 'string' },
          delivers: { type: 'string', description: 'the end-to-end behaviour this slice makes work' },
          body: {
            type: 'string',
            description:
              'the ticket body in Markdown, carrying "## Parent", "## What to build" and "## Acceptance criteria" as a checklist. Do NOT write a "## Blocked by" section -- the filing step appends it with real issue numbers.',
          },
          // Indices, not issue numbers: the drafting agent files nothing, so no number
          // exists yet. The script resolves these to real numbers after filing, which
          // also makes the dependency graph checkable before a single issue is created.
          blocked_by_index: {
            type: 'array',
            items: { type: 'integer' },
            description: '0-based indices into THIS tickets array that must land first. Empty means it can start immediately.',
          },
          acceptance: { type: 'array', items: { type: 'string' } },
          // The default is ready. A ticket that needs a person has to say which decision
          // it is waiting on -- "it feels big" is not one, and a gate applied out of
          // caution is a gate nobody reads.
          needs_human: { type: 'boolean', description: 'true ONLY if a person must decide something before an agent can start. Default false.' },
          needs_human_why: { type: 'string', description: 'the specific decision a person owes this ticket, or an empty string' },
          files_named: { type: 'array', items: { type: 'string' }, maxItems: 30, description: 'every repository file path this body names' },
          verify: { type: 'string', description: 'the one runnable command from "## Verify", verbatim' },
          expected_lines: { type: 'integer', description: 'estimated changed lines. Hours-to-two-days of work; over ~400 is a signal to split.' },
        },
      },
    },
    wide_refactor: { type: 'boolean', description: 'true if the set is an expand-migrate-contract sequence rather than tracer bullets' },
    notes: { type: 'string' },
  },
}

const FILED_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['filed', 'tickets'],
  properties: {
    filed: { type: 'boolean' },
    spec_issue: { type: 'integer', description: 'the parent spec issue number; 0 if nothing was filed' },
    spec_url: { type: 'string' },
    tickets: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['index', 'number'],
        properties: {
          index: { type: 'integer', description: 'the 0-based index of this ticket in the list you were given' },
          number: { type: 'integer', description: 'the issue number created, or the existing one if you found a duplicate' },
          url: { type: 'string' },
          duplicate_of: { type: 'integer', description: 'set when an open issue already covered this slice and you commented on it instead of filing' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

// -- Intake --------------------------------------------------------
// The request form is the contract with the requester, and this stage is where it stops
// being prose and becomes data. Its one hard rule: MARK what the form does not settle,
// never fill it in. That is spec-kit's [NEEDS CLARIFICATION] mechanism, and it exists
// because a model handed an underspecified form will produce a confident answer to a
// question nobody asked -- and nothing downstream can tell that it did.
if (!REQUEST && !TOPIC) {
  return {
    error: 'prepare-tickets needs a request. File one from .github/ISSUE_TEMPLATE/request.yml and pass its number: { request: 300 }. For a run with no filed form, pass { topic: "..." } instead.',
    example: { request: 300, maxTickets: 4 },
  }
}

phase('Intake')
const intake = REQUEST
  ? await call(
      [
        `You are reading request issue #${REQUEST} in the GraphKnows repository at ${MAIN_REPO} and turning it into the frame a technical project owner would hand an engineer. ${GH}`,
        `Read it with \`gh issue view ${REQUEST} --json number,title,body,labels,state\` and its comments with \`gh issue view ${REQUEST} --comments\`. The comments matter: a previous run of this workflow may have asked questions there, and the answers are the requester's, not yours to re-decide.`,
        `The body was filled in from .github/ISSUE_TEMPLATE/request.yml. Its fields are: kind, problem, desired outcome, who it is for, constraints and decisions already made, explicitly out of scope, done when, severity, evidence. Read ${MAIN_REPO}/.github/ISSUE_TEMPLATE/request.yml if a heading is ambiguous.`,
        'Report the fields as they are. Do not improve them, do not resolve a contradiction between two of them silently, and do not answer a question the form left open.',
        '',
        'THE ONE RULE: where the form does not settle something an implementer would have to decide, add a clarification instead of choosing. Each one names the decision in a single sentence a person can answer WITHOUT reading any code, and says what you would do by default if nobody answers.',
        'Clarify only what would change the work. "Which storage backend" is a clarification; "what should the variable be called" is not. If the form is complete, return none -- an empty list is the good outcome, and inventing questions to look thorough costs the requester a round trip.',
        '',
        'Then judge DEPTH, and be honest rather than generous: does the request already name the approach, the mechanism, or the files to change -- has a person already done the design? "Collapse the Dockerfile to one stage, keep the non-root user" is settled. "Make the dev environment faster" is not. A settled request does not need research and design; re-deriving a decision someone already made produces a different answer, and the requester has to notice.',
        'Finally: a slug of at most six lowercase hyphenated words, and a one-sentence restatement of the request as a topic.',
      ].join('\n'),
      { label: 'intake', phase: 'Intake', schema: INTAKE_SCHEMA },
    )
  : {
      topic: TOPIC,
      kind: KIND === 'bug' ? 'bug' : 'feature',
      frame: NOTES,
      out_of_scope: [],
      done_when: [],
      approach_settled: DEPTH === 'shallow',
      clarifications: [],
      slug: SLUG,
    }

if (!intake?.topic) {
  return { request: REQUEST, error: 'intake could not read the request -- see the run journal' }
}
TOPIC = intake.topic
SLUG = opt.slug || intake.slug || SLUG
const clarifications = intake.clarifications || []
log(`request #${REQUEST || '(none)'}: ${TOPIC}`)

// The frame, as data. Everything downstream reads THIS rather than the issue body, so a
// field the form left blank is visibly blank rather than quietly filled in by whichever
// agent got there first.
const FRAME = [
  `THE REQUEST${REQUEST ? ` (issue #${REQUEST})` : ''}: ${TOPIC}`,
  intake.frame ? `What the requester says is already decided -- treat it as settled and do not re-litigate it:\n${intake.frame}` : '',
  (intake.out_of_scope || []).length ? `EXPLICITLY OUT OF SCOPE, per the request:\n${intake.out_of_scope.map((x) => `  - ${x}`).join('\n')}` : '',
  (intake.done_when || []).length ? `The requester's own success test:\n${intake.done_when.map((x) => `  - ${x}`).join('\n')}` : '',
  NOTES && NOTES !== 'nothing beyond the request itself' ? `Also decided by whoever launched this run: ${NOTES}` : '',
  clarifications.length
    ? `OPEN QUESTIONS the request does not settle. Each has a default; use the default and say you did -- do NOT invent a different answer:\n${clarifications
        .map((c) => `  - ${c.question}\n      default if unanswered: ${c.default_if_unanswered || 'none stated'}`)
        .join('\n')}`
    : '',
]
  .filter(Boolean)
  .join('\n')

// -- Screen --------------------------------------------------------
// Go / no-go, and the cheapest stage here by a wide margin: it is the only one that can
// stop a request before it costs anything. Nothing owned this before -- the workflow would
// research, design, spec and file seven issues for a thing that already existed, and the
// only backstop was a person skimming the result afterwards.
let screen = null
if (REQUEST && opt.screen !== false) {
  phase('Screen')
  screen = await call(
    [
      `Decide whether request #${REQUEST} should become work at all, for the GraphKnows repository at ${MAIN_REPO}. ${GH}`,
      FRAME,
      'Three checks, in this order. Stop at the first that fires.',
      '',
      '1. ALREADY BUILT. Search for an existing implementation of the requested behaviour BY DOMAIN CONCEPT, not by the words the request happens to use -- the same capability is usually already there under a different name. Answer with path:line or with "no". Read the file before claiming it; a plausible-looking name is not an implementation.',
      GRAPHIFY,
      `2. PREVIOUSLY REJECTED. Read every file in ${MAIN_REPO}/.out-of-scope/ (its README explains the format). If one of them already declined this, say which file and quote the line that says what would have to change for the answer to be different -- that line is the actual answer to whether this request is different.`,
      '3. OUTSIDE THE FRAME. Does the request contradict a stated product boundary in CLAUDE.md, an accepted ADR under docs/adr/, or its own "out of scope" field? Name the document and the line.',
      '',
      'Also check for an OPEN issue that already covers this (`gh issue list --state open --search "<domain words>"`). A duplicate is not a no-go on its own -- report the number and let the caller decide.',
      '',
      'Verdict "go" unless one of the three fired. Be specific about what you searched: a no-go that cannot name where it looked is a guess, and a guess here silently kills real work. When you are unsure, say "go" -- the cost of a wrong go is a spec nobody files; the cost of a wrong no-go is a request that vanishes.',
      'Change nothing and file nothing.',
    ].join('\n'),
    { label: 'screen', phase: 'Screen', schema: SCREEN_SCHEMA },
  )

  if (screen?.verdict === 'no-go') {
    log(`SCREEN: no-go -- ${screen.reason}`)
    if (FILE) {
      await call(
        [
          `Comment on GitHub issue #${REQUEST} and then close it. ${GH}`,
          'The text is final -- do not rephrase it, extend it, or add a signature.',
          `\`gh issue comment ${REQUEST} --body "<the text below, verbatim>"\` then \`gh issue close ${REQUEST} --reason "not planned"\`.`,
          '<<<COMMENT>>>',
          `**Screened by ${RUN}: not planned.**`,
          '',
          screen.reason,
          screen.evidence ? `\nEvidence: ${screen.evidence}` : '',
          `\nIf that is wrong, say so here and reopen -- a screen that reads the situation wrongly is a bug in the screen, not a decision.`,
          '<<<END COMMENT>>>',
        ].join('\n'),
        { label: 'screen:close', phase: 'Screen' },
      )
    }
    return {
      request: REQUEST,
      topic: TOPIC,
      screened_out: true,
      reason: screen.reason,
      evidence: screen.evidence || '',
      searched: screen.searched || [],
      next_step: `Nothing was filed. #${REQUEST} was closed as not planned. If the screen is wrong, reopen it and re-run with { request: ${REQUEST}, screen: false }.`,
    }
  }
  if (screen?.duplicate_of) log(`screen: possible duplicate of #${screen.duplicate_of} -- filing anyway, close whichever loses`)
  log(`screen: go${screen?.searched?.length ? ` (looked at ${screen.searched.length} place(s))` : ''}`)
}

// The depth decision, made once and logged. Research and design exist to settle an approach;
// when the request already carries one, running them re-opens a decision a person closed.
const SETTLED = DEPTH === 'shallow' || (DEPTH === 'auto' && intake.approach_settled === true)
if (SETTLED) {
  log(`depth: shallow -- the request already names the approach, so research and design are skipped${DEPTH === 'shallow' ? ' (forced)' : ''}`)
}
if (clarifications.length) log(`intake: ${clarifications.length} open question(s) for the requester -- they will be posted on #${REQUEST || 0} and defaulted meanwhile`)

// -- Research ------------------------------------------------------
// Four fixed lanes, each blind to the others. The lanes are fixed rather than
// generated so two runs on the same topic cover the same ground.
// The lanes and stances are fixed arrays, so asking for more than there are is silently
// capped. Say so: a run that quietly ignores an argument reads as a run that honoured it.
if (opt.angles > ANGLES) log(`angles capped at ${ANGLES} -- there are only ${ANGLES} research lanes`)
if (opt.designs > DESIGNS) log(`designs capped at ${DESIGNS} -- there are only ${DESIGNS} design stances`)

const LANES = [
  {
    key: 'sources',
    ask: [
      'Investigate the topic against PRIMARY sources: official documentation, specifications, the source code of the libraries involved, and the papers this repo is built on.',
      `Start with ${MAIN_REPO}/CLAUDE.md -- it summarises the Tensor Brain, the model this system is built after, and a design that contradicts it must say so out loud. docs/README.md indexes the user-facing documentation; a design that changes documented behaviour has to change that page too.`,
      'Follow every claim back to the source that owns it. A blog post summarising a spec is not the spec.',
    ],
  },
  {
    key: 'codebase',
    ask: [
      'Establish what the code does TODAY in the area this topic touches. Read the modules, the tests around them, and the CLI or evaluation entry points that drive them.',
      'Report the current control flow, the current interfaces, and where the seams already are. Cite path:line for everything.',
      GRAPHIFY,
    ],
  },
  {
    key: 'prior-art',
    ask: [
      'Find what already exists in this repository that solves part of the topic, so the design reuses instead of reinventing. Search for near-duplicate helpers, abandoned modules, half-built channels, and anything in evaluation/ or scripts/ that already measures the thing.',
      'Also report anything that was clearly TRIED and abandoned: `git log --oneline -40`, deleted modules, dead configuration.',
      'The most valuable finding here is "this already exists at path:line" -- look hard before concluding it does not.',
      GRAPHIFY,
    ],
  },
  {
    key: 'constraints',
    ask: [
      'Establish the constraints any design must satisfy: what the test suite already asserts in this area, what the evaluation harness measures, what pyproject.toml declares, and what the CI gate enforces.',
      'Read pyproject.toml and the importlinter contracts, and report the rules a design cannot break. Ruff and pytest are configured in pyproject.toml -- there is no ruff.toml or pytest.ini.',
      PACKAGING,
    ],
  },
].slice(0, ANGLES)

phase('Research')
// A settled request still gets the codebase lane. It does NOT get the other three: sources,
// prior art and constraints exist to help CHOOSE an approach, and the choosing is done. But
// an engineer handed a settled approach still reads the code before writing the spec, and
// skipping that is how a spec describes a seam that does not exist.
const RUN_LANES = SETTLED ? LANES.filter((l) => l.key === 'codebase') : LANES
if (SETTLED) log(`research: 1 lane (codebase) instead of ${LANES.length} -- the approach is settled, only the ground still needs reading`)
const research = await parallel(
  RUN_LANES.map((lane) => () =>
    call(
      [
        `You are researching one lane of a design question for the GraphKnows repository at ${MAIN_REPO}.`,
        `THE TOPIC: ${TOPIC}`,
        `What the user already decided: ${NOTES}`,
        FRAME,
        RUN_LANES.length > 1
          ? `YOUR LANE -- ${lane.key}. Stay in it; ${RUN_LANES.length - 1} other agents cover the other lanes and duplicated work is wasted.`
          : `YOUR LANE -- ${lane.key}. You are the only lane running: the approach is already settled, so your job is the ground it lands on, not whether it is the right approach.`,
        ...lane.ask,
        EVIDENCE,
        'Change no code. Write no files. Report findings only.',
      ].join('\n'),
      { label: `research:${lane.key}`, phase: 'Research', schema: RESEARCH_SCHEMA },
    ),
  ),
)

const lanes = research.map((r, i) => ({ key: RUN_LANES[i].key, r })).filter((x) => x.r)
const findings = lanes.flatMap((x) => (x.r.findings || []).map((f) => ({ ...f, lane: x.key })))
const openQuestions = [...new Set(lanes.flatMap((x) => x.r.open_questions || []))]
log(`research: ${findings.length} finding(s) across ${lanes.length} lane(s), ${openQuestions.length} open question(s)`)

// Reduced in plain JS, not by an agent: this is a join and a format, not a judgement.
const BRIEF = [
  'RESEARCH BRIEF -- every line below was verified by a research agent and carries its source.',
  ...findings.map((f) => `  [${f.lane}] ${f.claim}\n      source: ${f.source}\n      so what: ${f.so_what}`),
  openQuestions.length
    ? 'STILL OPEN (nobody could verify these):\n' + openQuestions.map((q) => `  - ${q}`).join('\n')
    : 'Nothing was left unverified by the research lanes.',
].join('\n')

if (!findings.length) {
  return { topic: TOPIC, error: 'every research lane came back empty -- see the run journal', open_questions: openQuestions }
}

// -- Design --------------------------------------------------------
// Design it twice, three ways, with deliberately opposed stances. Identical
// prompts would give three copies of the same answer; the stance is the diversity.
const STANCES = [
  {
    key: 'smallest',
    stance:
      'YOUR STANCE: the laziest thing that could possibly work. Question whether this needs to exist at all, then reach for what the repo already has, then the standard library, then one line. Your design should be the one a reviewer calls "is that really all?". If the honest answer is that no code is needed, say so -- that is a legitimate design.',
  },
  {
    key: 'deepest',
    stance:
      'YOUR STANCE: the deepest module. Maximise behaviour hidden behind the smallest interface a caller must learn. Push the seam as high as it will go, and be explicit about what complexity you are burying and what the caller no longer needs to know.',
  },
  {
    key: 'seam-moving',
    stance:
      'YOUR STANCE: the current seam is in the wrong place. Assume the reason this is awkward is that the interface lives somewhere it should not, and design the version that moves it -- even if that means a wider change. Be honest about the blast radius; do not pretend it is small.',
  },
].slice(0, DESIGNS)

// Skipped wholesale when the request already settles the approach. Three opus agents
// designing what a person has already decided do not converge on that decision -- they
// produce a fourth one, and the requester finds out by reading a spec that answers a
// question they had closed.
let alive = []
let CANDIDATES = 'No independent designs were run: the request already settled the approach.'
if (!SETTLED) {
phase('Design')
const designOnce = (s) =>
  call(
    [
      `You are designing a change to the GraphKnows repository at ${MAIN_REPO}. Two other agents are designing the same thing from different stances; you will never see theirs, and a synthesis agent picks between you.`,
      `THE TOPIC: ${TOPIC}`,
      `What the user already decided: ${NOTES}`,
      BRIEF,
      s.stance,
      skill('codebase-design/SKILL.md', 'the deep-module vocabulary. Use its terms exactly: module, interface, implementation, depth, seam, adapter.'),
      skill('codebase-design/DESIGN-IT-TWICE.md', 'its designer section: why you are one of several, and what makes an alternative real rather than a rename.'),
      HOUSE,
      PACKAGING,
      'Read the files the brief cites before designing against them. Design only -- change no code and write no files.',
      'Fill prototype_question ONLY if there is a specific behaviour your design depends on that reading code cannot settle -- a state machine that might deadlock, an ordering you are guessing at, a retrieval quality claim. Otherwise leave it an empty string. Wanting reassurance is not a prototype question.',
    ].join('\n'),
    { label: `design:${s.key}`, phase: 'Design', schema: DESIGN_SCHEMA },
  )

const designs = await parallel(STANCES.map((s) => () => designOnce(s)))
alive = designs.map((d, i) => ({ key: STANCES[i].key, d })).filter((x) => x.d)

// The floor. A dead stance costs nothing visible: synthesis still runs, still picks, still
// sounds decisive -- on one candidate, which is not a design-it-twice, it is one agent with
// four stages of ceremony in front of it. So: say what died, retry it once, and refuse to
// synthesise a comparison of fewer than two things.
if (alive.length < STANCES.length) {
  const dead = STANCES.filter((s) => !alive.some((x) => x.key === s.key))
  log(`designs: ${dead.length}/${STANCES.length} stance(s) returned nothing (${dead.map((s) => s.key).join(', ')}) -- retrying once`)
  const again = await parallel(dead.map((s) => () => designOnce(s)))
  alive = alive.concat(again.map((d, i) => ({ key: dead[i].key, d })).filter((x) => x.d))
}
const MIN_DESIGNS = Math.min(2, STANCES.length)
if (alive.length < MIN_DESIGNS) {
  return {
    topic: TOPIC,
    error: `only ${alive.length} of ${STANCES.length} design stances survived, after a retry. Synthesis over one candidate is not a choice -- relaunch, or narrow the topic so the design agents produce less.`,
    brief: BRIEF,
  }
}
log(`designs: ${alive.map((x) => x.key).join(', ')} (${alive.length}/${STANCES.length})`)

CANDIDATES = alive
  .map((x) =>
    [
      `--- CANDIDATE "${x.key}" ---`,
      `approach: ${x.d.approach}`,
      `seam: ${x.d.seam}`,
      `interface: ${x.d.interface}`,
      `why it is deep: ${x.d.why}`,
      `costs: ${(x.d.costs || []).join(' | ') || 'none stated'}`,
      `blast radius: ${(x.d.blast_radius || []).join(' | ') || 'none stated'}`,
      `new dependencies: ${(x.d.new_dependencies || []).join(', ') || 'none'}`,
      `prototype question: ${x.d.prototype_question || 'none'}`,
    ].join('\n'),
  )
  .join('\n\n')

}

// -- Synthesis -----------------------------------------------------
// A barrier was required above: this stage compares the candidates against each
// other, which is exactly the case a barrier is for.
phase('Synthesis')
const synth = await call(
  [
    SETTLED
      ? `You are writing down the design for a change to the GraphKnows repository at ${MAIN_REPO}. The approach was already decided by the requester -- your job is to state it precisely enough to spec against, not to reconsider it.`
      : `You are choosing between ${alive.length} independent designs for the GraphKnows repository at ${MAIN_REPO}, and writing the record of that choice.`,
    `THE TOPIC: ${TOPIC}`,
    FRAME,
    BRIEF,
    CANDIDATES,
    SETTLED
      ? 'The approach is the requester\'s and it is settled. Restate it IN FULL under "chosen", made concrete against the code you were just shown: the seam it lands at, the interface it changes, what it touches. Leave "rejected" empty -- nothing was rejected here, a person chose.'
      : 'Pick ONE as the base. You may graft specific ideas from the runners-up onto it, but the result must stay coherent -- a design assembled from three halves is worse than any of the three.',
    SETTLED
      ? 'If the settled approach cannot work as stated -- the code contradicts it, the seam is not there -- do NOT quietly design around it. Say so plainly in rationale and name the specific fact that blocks it. That is a finding for the requester, not a licence to substitute your own design.'
      : 'Restate the chosen design IN FULL under "chosen": every stage after this one sees your restatement and none of the candidates. If it is not in there, it does not exist.',
    SETTLED ? '' : 'Say why each rejected candidate lost. "Less good" is not a reason; name the cost that decided it.',
    'Bias: prefer the design that needs the least new code and no new dependency, unless a research finding shows it cannot work. The repo rules forbid speculative abstraction, and an interface with one implementation is not a design, it is a wrapper.',
    'Under depends_on, quote the findings from the brief that your choice RESTS ON -- the ones that, if they turned out to be false, would make you choose differently. Each is handed to a skeptic who will try to refute it against the actual files, so list the load-bearing facts, not everything you read. Five at most.',
    HOUSE,
    `THEN decide whether this decision earns an ADR. Read ${MAIN_REPO}/${ADR_DIR}/0001-record-architecture-decisions.md first -- it states the bar and the form.`,
    "The bar is Nygard's and it is strict: architecturally significant means it affects structure, non-functional characteristics, dependencies, interfaces, or construction techniques. Choosing a helper's location, a variable name or a test style is a commit message, not a record. If it does not clear the bar, set adr_written false, say why in one line, and write nothing -- a directory full of records for routine choices is how the filter stops working.",
    `If it does clear the bar, write ONE file to ${MAIN_REPO}/${ADR_DIR}/. Number it as the next unused four-digit prefix (list the directory first) and slug it from the decision: NNNN-<slug>.md. Sections: a "# N. <title>" heading, then Status, Context, Decision, Consequences.`,
    'ONE PAGE. Fowler: "Keep the ADR short and to the point - typically a single page." Context is value-neutral fact, Decision is active voice ("We will..."), Consequences names the good, the bad and the neutral. Where one sentence carries it, a Y-statement is a complete ADR: "In the context of X, facing Y, we decided for Z to achieve W, accepting V."',
    'The ADR holds the WHY -- options considered, what was rejected, what it costs. It is the reason the spec does not have to carry any of that. Do not restate the ADR in the spec later; link it.',
    'Set adr_written and adr_path once the file exists on disk. Do not commit it and do not change any other file.',
  ].join('\n'),
  { label: 'synthesis', phase: 'Synthesis', schema: SYNTH_SCHEMA, ...DEEP },
)

if (!synth?.chosen) return { topic: TOPIC, error: 'synthesis returned no design -- see the run journal', brief: BRIEF, candidates: CANDIDATES }
log(`chosen design recorded${synth.adr_written ? `, ADR at ${synth.adr_path || ADR_DIR}` : `, no ADR -- ${synth.adr_skipped_why || 'not architecturally significant'}`}`)

const CHOSEN = [
  'THE CHOSEN DESIGN (this is the design; the alternatives are settled and closed):',
  synth.chosen,
  synth.grafted?.length ? `Grafted from the runners-up: ${synth.grafted.join(' | ')}` : '',
  synth.risks?.length ? `Known risks: ${synth.risks.join(' | ')}` : '',
]
  .filter(Boolean)
  .join('\n')

// -- Contest -------------------------------------------------------
// A finding enters the brief with a source and is treated as fact by every stage after it.
// Nothing ever asked whether it was TRUE. This does, for the handful the winning design
// leans on -- one skeptic each, told to refute rather than to confirm.
const MAX_CONTEST = 5
const leaned = (synth.depends_on || []).map((x) => String(x).trim()).filter(Boolean)
const contested = leaned.slice(0, MAX_CONTEST)
if (leaned.length > contested.length) log(`contesting ${contested.length} of ${leaned.length} load-bearing finding(s) -- the rest were dropped over the cap`)

let refuted = []
if (contested.length) {
  phase('Contest')
  // Two claims per skeptic: the fixed per-agent overhead (session context) dwarfs the cost
  // of checking a second claim, and the claims are independent facts, not votes -- pairing
  // them loses no adversarial independence between DIFFERENT claims' verdicts.
  const pairs = []
  for (let i = 0; i < contested.length; i += 2) pairs.push(contested.slice(i, i + 2))
  const verdicts = await parallel(
    pairs.map((batch, i) => () =>
      call(
        [
          `You are trying to REFUTE ${batch.length === 1 ? 'one claim' : 'each of these claims'} that a design decision rests on, for the GraphKnows repository at ${MAIN_REPO}. Judge each claim on its own -- one being wrong says nothing about the other.`,
          ...batch.map((c, j) => `CLAIM ${j + 1}: ${c}`),
          'Each came from a research agent that cited a source for it. That agent may have misread the file, cited a line that says something else, or generalised from one case.',
          'Go and check each against the actual files, the actual command output, or the actual primary source. Read the thing itself -- do not reason about whether it sounds right.',
          CHOSEN,
          'Per claim, set refuted=true ONLY if you established it is wrong. A claim you could not verify either way is NOT refuted -- say so under why and leave refuted false. Being unable to confirm something is not the same as disproving it.',
          'If one is wrong, say what is true instead under its correction.',
          'Each verdict MUST repeat its claim verbatim under "claim" -- that is how the script attaches your verdict to the right claim, and a refutation attached to the wrong claim poisons the spec.',
          'Change nothing. Write no files.',
        ].join('\n'),
        {
          label: `contest:${i + 1}`,
          phase: 'Contest',
          schema: {
            type: 'object',
            additionalProperties: false,
            required: ['verdicts'],
            properties: {
              verdicts: {
                type: 'array',
                items: {
                  ...CONTEST_SCHEMA,
                  required: [...CONTEST_SCHEMA.required, 'claim'],
                  properties: { claim: { type: 'string', description: 'the contested claim, verbatim' }, ...CONTEST_SCHEMA.properties },
                },
              },
            },
          },
        },
      ),
    ),
  )
  // Attach each verdict to its claim by TEXT, not by position: a refutation stamped onto
  // the wrong claim is worse than a lost verdict. Position is only the tiebreak, and a
  // verdict matching nothing is dropped with a log line rather than guessed at.
  const squash = (s) => String(s).toLowerCase().replace(/\s+/g, ' ').trim()
  refuted = verdicts
    .flatMap((v, i) =>
      (v?.verdicts || []).map((x, j) => {
        const echoed = squash(x?.claim || '')
        const match = pairs[i].find((c) => squash(c) === echoed || (echoed && squash(c).includes(echoed))) ?? pairs[i][j]
        if (!match) log(`contest:${i + 1} returned a verdict matching no claim it was given -- dropped ("${String(x?.claim || '').slice(0, 80)}")`)
        return match ? { ...(x || {}), claim: match } : null
      }),
    )
    .filter((x) => x?.refuted === true)
  log(
    refuted.length
      ? `contest: ${refuted.length} of ${contested.length} load-bearing finding(s) REFUTED`
      : `contest: all ${contested.length} load-bearing finding(s) survived`,
  )
} else {
  log('contest skipped: the synthesis named no load-bearing findings')
}

// A refuted fact does not stop the run -- it travels with it, in the body the human reads.
const CONTEST_NOTE = refuted.length
  ? [
      'A SKEPTIC REFUTED FINDINGS THE CHOSEN DESIGN RESTS ON. Treat these as open questions, not as settled facts:',
      ...refuted.map((r) => `  - claim: ${r.claim}\n      refuted because: ${r.why}${r.correction ? `\n      what is true instead: ${r.correction}` : ''}`),
    ].join('\n')
  : ''

// -- Prototype -----------------------------------------------------
// Gated twice: the operator can force it off, and on 'auto' it runs only when the
// synthesis named a question that reading code cannot settle. A prototype built to
// feel reassured is throwaway code nobody reads.
const protoQ = (synth.prototype_question || '').trim()
const wantProto = PROTO === 'always' || (PROTO !== 'never' && protoQ.length > 0)
let proto = null
if (wantProto) {
  phase('Prototype')
  proto = await call(
    [
      `You are building a THROWAWAY prototype to settle one question about a design for the GraphKnows repository at ${MAIN_REPO}.`,
      `THE QUESTION: ${protoQ || 'does the chosen design behave the way it claims under the cases that are hard to reason about on paper?'}`,
      CHOSEN,
      skill('prototype/SKILL.md', 'how to pick the prototype branch and the rules for the branch you pick. You have no human to drive a TUI -- use the scripted scenario mode.'),
      'Work on a throwaway branch off the current checkout -- `git checkout -b proto/<short-slug>` -- and never on the base branch. The skill owns the throwaway rules; follow them.',
      'Then answer the question. A prototype that answers "no" is a success: report it plainly.',
      'Under design_changes, list what the chosen design must change because of what you saw -- empty if the design survives intact.',
      'Under snippet, include ONLY the decision-rich part -- the state machine, the schema, the type shape -- not a working demo.',
      'Commit the prototype on its throwaway branch and report the branch name and the run command. Do not merge it anywhere.',
    ].join('\n'),
    { label: 'prototype', phase: 'Prototype', schema: PROTO_SCHEMA },
  )
  log(proto?.ran ? `prototype verdict: ${proto.verdict}` : 'prototype did not run')
} else {
  log(PROTO === 'never' ? 'prototype skipped (prototype: never)' : 'prototype skipped: the synthesis named no question that needs running code')
}

const PROTO_NOTE = proto?.ran
  ? [
      'A PROTOTYPE WAS BUILT to settle a design question. Its verdict is evidence, not opinion:',
      `  question: ${protoQ}`,
      `  verdict: ${proto.verdict}`,
      `  what it showed: ${proto.what_it_showed}`,
      proto.design_changes?.length ? `  the design must change: ${proto.design_changes.join(' | ')}` : '  the design survived intact',
      proto.branch ? `  it lives on branch ${proto.branch} (run: ${proto.run_command || 'see the branch'})` : '',
      proto.snippet ? `  the decision it encodes:\n${proto.snippet}` : '',
    ]
      .filter(Boolean)
      .join('\n')
  : 'No prototype was built for this design.'

// -- Spec ----------------------------------------------------------
// Drafts, does not file. Filing happens once, in Tickets, after the critic has had
// its say -- an issue filed and then corrected by a comment reads as two decisions.
phase('Spec')
const spec = await call(
  [
    `You are writing the spec for a change to the GraphKnows repository at ${MAIN_REPO}. It will be filed as the parent issue that every ticket below it references.`,
    `THE TOPIC: ${TOPIC}`,
    FRAME,
    CHOSEN,
    PROTO_NOTE,
    CONTEST_NOTE,
    BRIEF,
    // spec-kit's template rather than to-spec's, because it is the more mature of the two at
    // exactly this level: it has Edge Cases, Key Entities, Assumptions and measurable SC-###
    // outcomes that to-spec has no equivalent for. It is committed at this path, so it stays
    // the single source of the shape rather than a copy in this prompt that drifts from it.
    // Deliberately NOT adopted below the spec: spec-kit turns tasks into title-only issues,
    // and the ticket contract here is the only thing the implementing workflow ever reads.
    `Read ${MAIN_REPO}/.specify/templates/spec-template.md and follow its section order and heading text exactly. It is a template: replace every bracketed placeholder, delete the sample rows and the HTML comment blocks, and drop the \`*(mandatory)*\` annotations from the headings -- they mark the template's own rules, they are not part of a written spec.`,
    'TWO SECTIONS THIS REPO ADDS on top of that template, both after "## Success Criteria" and both required:',
    '  - "## Implementation Decisions" -- where the ADR is cited. spec-kit has nowhere for it, and the implementing workflow is told the spec cites one.',
    '  - "## Testing Decisions" -- the seams. This is the ONLY channel the implementing workflow ever reads them through; spec-kit has no equivalent section at all.',
    'Do not add any other section, and do not keep a template section you have nothing to put in -- an empty heading is a heading somebody reads looking for content.',
    `Vocabulary: use the domain terms the repo already uses -- the Tensor Brain lexicon in ${MAIN_REPO}/CLAUDE.md and the terms defined in docs/ontology.md, docs/architecture.md and docs/services.md. Do not coin new ones for concepts that have names.`,
    'Sketch the seams you will test the feature at, per the template, and ALSO list them highest-first under "seams" in your output.',
    'The user-stories section is in "As an <actor>, I want <feature>, so that <benefit>" form, PRIORITIZED: mark each story P1/P2/P3 and order them highest first. Each story must be independently testable, and the P1 story alone must be a viable MVP. One story per distinct actor with a distinct want -- if two stories differ only in wording, they are one story.',
    'Number the functional requirements FR-001, FR-002, ... in order under "## Requirements" -- tickets may cite them by number.',
    'Number the success criteria SC-001, SC-002, ... under "## Success Criteria", and put the same list under `success_criteria` in your output. Each is a MEASURABLE outcome someone could check after the change lands -- a number, a rate, a bound -- and technology-agnostic: "recall returns the right chunk for a known instant in 100% of cases", not "the new index is used". An acceptance scenario says the feature works; a success criterion says it was worth building. Do not invent business metrics nobody here can measure; if the honest answer is one criterion, write one.',
    'Length is a cost, not a virtue. Every sentence an implementer would not act on is a sentence that makes them build more. Do not re-derive what the user already decided above -- state it once, in one line, and move on.',
    'No file paths and no code snippets in the body -- they go stale. The one exception is a snippet from the prototype that encodes a decision more precisely than prose can; trim it to the decision.',
    // The spec carries the contract. The rationale lives in the ADR, which is COMMITTED --
    // so unlike the gitignored note this replaced, a reader can actually open it, and the
    // spec no longer has to restate everything it knows in prose to be self-contained.
    synth.adr_written
      ? `The WHY is already written down, in \`${synth.adr_path}\`, committed to this repository. Cite it by path in "## Implementation Decisions" in one line and DO NOT restate its contents: not the options considered, not why the losers lost, not the costs accepted. This spec states what must become true and where it is tested. A spec that re-argues its own decision is a document a reader has to wade through to find the requirement.`
      : 'No ADR was written -- the decision is not architecturally significant. State the decision itself in one or two lines under "## Implementation Decisions" and do not narrate how it was reached.',
    'Everything ELSE a reader needs must be in this body. Do not cite any gitignored or local-only file as provenance -- a reader cannot open it.',
    `Keep the body under 400 lines. GitHub caps an issue at 65,536 characters and a spec here has already reached 56 KB of that; past a few hundred lines nothing downstream reads it usefully. If it will not fit, it is describing more than one change${REQUEST ? ` -- say so in your notes and scope it to what #${REQUEST} actually asked for` : ''}.`,
    'Name the seams inside "## Testing Decisions" in the same words as your "seams" output -- a deterministic check refuses to file a spec whose body omits them; that section is the only place the implementing workflow will ever see them.',
    HOUSE,
    PACKAGING,
    'Write the body as Markdown ready to be pasted into a GitHub issue. Change no code and file nothing.',
  ].join('\n'),
  { label: 'spec', phase: 'Spec', schema: SPEC_SCHEMA, ...DEEP },
)

if (!spec?.body) return { topic: TOPIC, error: 'the spec agent returned no body -- see the run journal', chosen: synth.chosen, adr_path: synth.adr_path || '' }

// -- Critic --------------------------------------------------------
// The interview the skills expect, replaced by an adversary. to-spec and to-tickets
// both quiz the user; a workflow cannot, so the draft has to survive a skeptic instead.
phase('Critic')
const critic = await call(
  [
    'You are the adversarial reader of a draft spec. Your job is to find what would make an implementer build the wrong thing. Assume it is flawed and go looking.',
    `THE TOPIC IT CLAIMS TO ADDRESS: ${TOPIC}`,
    BRIEF,
    CHOSEN,
    PROTO_NOTE,
    CONTEST_NOTE,
    `THE DRAFT SPEC:\n${spec.body}`,
    'Check, in this order:',
    '  1. Does it actually solve the topic, or a nearby easier problem?',
    '  2. Does any statement contradict a research finding above? Name both sides.',
    '  3. Is anything asserted that no lane verified -- a fact about the code, a library behaviour, a performance claim?',
    '  4. Are the seams real? Read the test files it names and confirm the seam exists where it says.',
    '  5. What is missing: a lane nobody ran, a source nobody read, a failure mode nobody named.',
    '  6. Is it speculative? A section describing something nobody asked for is scope the implementer will build.',
    '  7. What can be DELETED? Fill `cut` with everything the implementer does not need: a requirement restating what an earlier one already implies, an acceptance criterion asserting a fact rather than a behaviour, a paragraph re-deriving something the user already decided, scope that belongs in a separate ticket. You are the only stage in this pipeline that can remove anything; every other one appends. A review that leaves the spec longer than it found it has failed, so name at least one cut unless the spec is genuinely minimal, and say so explicitly if it is.',
    'Every must_fix needs a concrete fix, not a complaint. Style preferences are not findings.',
    'Verdict: "ready" if an implementer could not go wrong, "fixable" if the must_fix list closes the gaps, "not-ready" if the design itself is unsound.',
    'Read files to check claims. Change nothing.',
  ].join('\n'),
  { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA, ...DEEP },
)

log(`critic: ${critic?.verdict || 'no verdict'}, ${(critic?.must_fix || []).length} must-fix`)

const CRITIC_NOTE = critic?.must_fix?.length || critic?.cut?.length
  ? [
      `An adversarial reviewer read the draft and returned "${critic.verdict}". Fold these into the spec before you file it:`,
      ...(critic.must_fix || []).map((m) => `  - ${m.problem}\n      fix: ${m.fix}`),
      critic.cut?.length
        ? ['  DELETE these from the spec before slicing it. They are cuts, not suggestions -- do not carry them into a ticket instead:', ...critic.cut.map((c) => `    - ${c}`)].join('\n')
        : '',
      critic.contradictions?.length ? `  Contradictions with the research: ${critic.contradictions.join(' | ')}` : '',
      critic.missing?.length ? `  Known gaps, record them under Further Notes so nobody rediscovers them: ${critic.missing.join(' | ')}` : '',
    ]
      .filter(Boolean)
      .join('\n')
  : 'An adversarial reviewer read the draft and found nothing that must change. File it as written.'

// -- Tickets -------------------------------------------------------
// Two agents: one drafts, the script checks, one files. The issue body is the ONLY thing
// the process-issues half reads -- there is no structured side channel -- so a body that
// is missing a section is a broken handover, and that is a fact about a string, which is
// code's job to establish rather than an agent's promise.
phase('Tickets')
const draft = await call(
  [
    `You are breaking an approved spec into tickets for the GraphKnows repository at ${MAIN_REPO}.`,
    `THE TOPIC: ${TOPIC}`,
    `THE SPEC (title: ${spec.title}):\n${spec.body}`,
    CRITIC_NOTE,
    PROTO_NOTE,
    CONTEST_NOTE,
    skill('to-tickets/SKILL.md', 'the Ticketing rules and Agent mode sections: vertical slices, blocking edges as data, the wide-refactor expand-migrate-contract exception. You are the agent-mode branch -- draft only.'),
    `Produce at most ${MAX_TICKETS} tickets. Each is a tracer bullet: a narrow but COMPLETE path through every layer, demoable on its own, sized to fit one fresh context window. A ticket that only adds a schema, or only adds a test, is a horizontal slice and is wrong. The first unblocked ticket should deliver the spec's P1 story end to end.`,
    'FILE NOTHING. Run no `gh issue create`, comment on nothing. A later step does the filing; your job is the text.',
    'Return the FINAL spec body under spec_body, with the critic must-fix items folded in. That exact text is filed verbatim -- nobody rewrites it after you.',
    // The spec keeps the no-paths rule -- it outlives the tree it was written against. A
    // ticket does not: it is implemented within hours by an agent standing in that exact
    // tree. Banning paths there does not prevent staleness, it just forces every one-word
    // reference to become a paragraph describing the file instead of naming it.
    'Ticket bodies MAY name file paths -- `docker-compose.prod.yaml`, not "the compose file whose header opens with a customer note". Name it and stop; do not also describe its contents back to a reader who can open it. (The SPEC body still carries no paths: it outlives this tree.) Do not cite any local, gitignored file as provenance -- the reader cannot open it.',
    // The single change with the most leverage in this whole prompt. Free-prose criteria
    // asserting the state of the tree read as a test specification, and an implementer
    // writes that test: 25 such criteria once came back as a 528-line test file in a
    // change whose entire point was deletion.
    [
      'Acceptance criteria are OBSERVABLE BEHAVIOUR. Each one takes one of two forms, and no other:',
      '  - EARS: "WHEN <trigger> THE SYSTEM SHALL <observable behaviour>"',
      '  - a rule with one concrete example of it holding',
      'NEVER assert the state of the repository. "WHEN a container starts under compose THE SYSTEM SHALL have a working interpreter on PATH" -- not "scripts/ contains exactly these four files", not "the Dockerfile has one stage", not "docs/versioning.md is deleted". A criterion describing what the tree LOOKS LIKE is the ticket copied into Python, and a deterministic check will reject the whole set for it.',
      'At most 8 per ticket. If you need more, it is two tickets, or the extras are not load-bearing.',
    ].join('\n'),
    'Estimate the diff each ticket implies, in changed lines. A batch that takes hours to two days is right; over ~400 lines is a signal to split it, and over ~1000 is certainly two tickets.',
    'A ticket whose work is DELETION does not earn new tests. The existing suite passing after the deletion is the acceptance. Do not ask for a test that asserts a file is absent.',
    // The point of the whole pipeline is to hand a person work they can approve at a
    // glance, not a pile of homework. A run that gates everything has decided nothing.
    [
      'Set needs_human on a ticket ONLY when a PERSON MUST DECIDE SOMETHING an agent cannot. There are exactly four kinds:',
      '  - it needs a number that does not exist yet (a threshold, a budget, a floor) and would otherwise be guessed;',
      '  - it makes an irreversible or outward-facing change: a release, a publish, deleting user data, a public API break;',
      '  - it rests on a finding a skeptic refuted above, so the premise itself is still open;',
      '  - it needs a product judgement the spec does not settle -- two defensible behaviours and no stated preference.',
      'Size is NOT a reason. Risk of being wrong is NOT a reason -- CI is. Wanting a second opinion is NOT a reason.',
      'Everything else defaults to needs_human: false, and that must be MOST of them. If you find yourself gating more than one or two, the spec is underspecified: fix the spec instead, or fold the undecided part into a single explicitly-gated ticket and leave the rest ready.',
      'When you do set it, needs_human_why names the decision in one sentence, phrased so a person can answer it without reading the code.',
    ].join('\n'),
    'Fill files_named with every repository file path the body names. The script reads it: a ticket that rewrites the CI or the pre-push gate is gated automatically, because the work would be verified by the thing it just changed.',
    'Each ticket body carries, in this order: "## Parent" (a line referencing the parent spec issue -- write it as "the parent spec issue", its number does not exist yet), "## What to build" as end-to-end behaviour naming the files and interfaces involved, "## Acceptance criteria" as a checklist, "## Verify", and "## Out of scope".',
    // The section that makes an unsupervised run finishable. Anthropic's own guidance: a
    // check the agent can run is "the difference between a session you watch and one you
    // walk away from" -- without it "looks done" is the only signal and a person becomes
    // the verification loop, which is the thing this pipeline exists to avoid.
    '"## Verify" is ONE runnable command and what passing looks like. Not a description of testing -- the literal command, copy-pasteable, and the output or exit condition that means it worked. If the only honest command is the existing suite, say that: for a deletion ticket the suite still passing IS the acceptance, which is exactly why deletion earns no new tests.',
    '"## Out of scope" names the neighbouring work it will be tempting to do and must not be done here. Two or three lines. This is the section that stops a ticket sprawling into its neighbours.',
    'Do NOT restate the house rules, the Definition of Done, or the design rationale in any ticket body. The Definition of Done at docs/agents/definition-of-done.md is inherited by every ticket and the implementer reads it; the rationale is in the spec and its ADR. Repeating them makes the ticket longer and the implementer worse -- restated context is a distractor in the exact document it must act on.',
    'Where the spec genuinely does not settle something a ticket needs, write "[NEEDS CLARIFICATION: <the question>]" in the body rather than choosing. A marker means the ticket waits for a person; a guess means it is wrong and nobody finds out until review.',
    'Do NOT write a "## Blocked by" section. Give the edges under blocked_by_index instead, as 0-based indices into your own tickets array; the filing step turns those into real issue numbers and appends the section. At least one ticket must have no blockers, or nothing can ever start.',
    'The exception is a wide refactor: if the change is one mechanical edit whose blast radius fans across the codebase, sequence it expand -> migrate in batches -> contract instead, and set wide_refactor.',
    HOUSE,
  ].join('\n'),
  { label: 'tickets:draft', phase: 'Tickets', schema: DRAFT_SCHEMA, ...DEEP },
)

if (!draft?.spec_body || !(draft.tickets || []).length) {
  return { topic: TOPIC, error: 'the ticket agent drafted nothing -- see the run journal', chosen: synth.chosen, spec_title: spec.title }
}

// The completeness check. Deterministic, and it refuses rather than repairs: an agent
// asked to fix its own body is the same agent that left the section out. Sliced out and
// exercised by gates.test.mjs -- the cycle detection especially, which is the part that
// silently does the wrong thing rather than throwing.
const checkHandover = (specBody, seams, drafted) => {
  // The trailing group tolerates a template annotation the writer forgot to strip
  // (`## Requirements *(mandatory)*`): the section IS there, and refusing the whole run over
  // an asterisk would be a check failing on punctuation rather than on the handover.
  const has = (body, heading) => new RegExp(`^##\\s+${heading}\\s*(\\*.*\\*)?\\s*$`, 'im').test(String(body || ''))
  const problems = []
  // spec-kit's three mandatory sections, plus the two this repo adds on top of its template.
  // "Testing Decisions" carries the seams and "Implementation Decisions" cites the ADR --
  // spec-kit has no equivalent for either, and the implementing workflow is told both exist.
  for (const heading of ['User Scenarios & Testing', 'Requirements', 'Success Criteria', 'Implementation Decisions', 'Testing Decisions']) {
    if (!has(specBody, heading)) problems.push(`the spec body has no "## ${heading}" section`)
  }
  // A section with no numbered lines is a heading, not requirements: speckit-analyze builds
  // its coverage inventory by keying on these ids, so it reports nothing rather than a gap.
  for (const [id, what] of [['FR-', 'functional requirement'], ['SC-', 'success criterion']]) {
    if (!new RegExp(`\\b${id}\\d{3}\\b`).test(String(specBody || ''))) {
      problems.push(`the spec body has no ${id}### ${what} -- an unnumbered one cannot be cited by a ticket or counted by speckit-analyze`)
    }
  }
  // Nothing bounded the spec before, and one reached 56 KB of GitHub's 65,536-char cap.
  const MAX_SPEC_LINES = 400
  const specLines = String(specBody || '').split('\n').length
  if (specLines > MAX_SPEC_LINES) {
    problems.push(`the spec body is ${specLines} lines, over ${MAX_SPEC_LINES} -- it is describing more than one change, or it is re-arguing decisions that belong in its ADR`)
  }
  // The seams are the one thing process-issues cannot re-derive cheaply, and the spec body
  // is now the only place it could ever read them from.
  for (const seam of seams || []) {
    if (!String(specBody).toLowerCase().includes(String(seam).toLowerCase().slice(0, 40))) {
      problems.push(`the spec body never names the seam "${seam}"`)
    }
  }
  // The size gate. Structure was already checkable here; size was not bounded anywhere,
  // so pressure that could not become a seventh ticket became a twenty-six-criterion
  // third one. Refuse rather than repair, same as every other check in this function: a
  // ticket too big to fit one context window is not a tracer bullet, and the agent that
  // wrote it is not the one to decide which half to drop.
  const MAX_ACCEPTANCE = 8
  // Raised from 60 when "## Verify" and "## Out of scope" were added: both earn their
  // lines, and a bound that forces a ticket to drop its own verification is the wrong bound.
  const MAX_BODY_LINES = 70
  // An acceptance criterion that describes the TREE rather than the behaviour is a test
  // specification wearing a checkbox, and an implementer writes the test. Caught by shape
  // here rather than by taste in review, because by review the test already exists.
  const TREE_ASSERTION = [
    /\bcontains? exactly\b/i,
    /\b(is|are|has been|have been)\s+(deleted|removed|gone|absent)\b/i,
    /\bno longer (exists?|present)\b/i,
    /\bthere (is|are) no\b.*\bfile/i,
    /\bthe (file|directory|section|block)\b.*\b(is|are) (gone|absent)\b/i,
  ]
  const criteriaOf = (body) =>
    String(body || '')
      .split('\n')
      .filter((l) => /^\s*[-*]\s*\[[ xX]\]/.test(l))
      .map((l) => l.replace(/^\s*[-*]\s*\[[ xX]\]\s*/, '').trim())

  drafted.forEach((t, i) => {
    for (const heading of ['Parent', 'What to build', 'Acceptance criteria', 'Verify', 'Out of scope']) {
      if (!has(t.body, heading)) problems.push(`ticket ${i + 1} ("${t.title}") has no "## ${heading}" section`)
    }
    for (const c of criteriaOf(t.body)) {
      const hit = TREE_ASSERTION.find((re) => re.test(c))
      if (hit) {
        problems.push(
          `ticket ${i + 1} ("${t.title}") has an acceptance criterion asserting the state of the tree rather than observable behaviour: "${c.slice(0, 80)}". Say what the system DOES once the change lands.`,
        )
      }
    }
    // A marker means a person still owes this an answer -- it must not be filed as a
    // ticket an agent may start on. Refusing the whole set is deliberate: the question
    // belongs on the request, where the requester will see it.
    if (/\[NEEDS CLARIFICATION/i.test(String(t.body || ''))) {
      problems.push(`ticket ${i + 1} ("${t.title}") still carries a [NEEDS CLARIFICATION] marker -- it is a question for the requester, not a ticket`)
    }
    const boxes = (String(t.body || '').match(/^\s*[-*]\s*\[[ xX]\]/gm) || []).length
    if (boxes > MAX_ACCEPTANCE) {
      problems.push(`ticket ${i + 1} ("${t.title}") has ${boxes} acceptance criteria, over the ${MAX_ACCEPTANCE} a tracer bullet gets -- split it or drop the ones that are not load-bearing`)
    }
    const lines = String(t.body || '').split('\n').length
    if (lines > MAX_BODY_LINES) {
      problems.push(`ticket ${i + 1} ("${t.title}") has a ${lines}-line body, over ${MAX_BODY_LINES} -- name files instead of describing them, and cut anything the implementer would not act on`)
    }
    for (const b of t.blocked_by_index || []) {
      if (b === i) problems.push(`ticket ${i + 1} ("${t.title}") blocks itself`)
      else if (!Number.isInteger(b) || b < 0 || b >= drafted.length) {
        problems.push(`ticket ${i + 1} ("${t.title}") is blocked by index ${b}, which is not a ticket in this set`)
      }
    }
  })

  // Kahn's algorithm: it produces the filing order AND proves the graph is acyclic, which
  // used to be a promise in a prompt ("get it wrong and work starts in the wrong order").
  const order = []
  let pending = drafted.map((t, i) => ({
    i,
    deps: (t.blocked_by_index || []).filter((b) => Number.isInteger(b) && b >= 0 && b < drafted.length && b !== i),
  }))
  while (pending.length) {
    const ready = pending.filter((r) => r.deps.every((d) => order.includes(d)))
    if (!ready.length) {
      problems.push(`the blocking edges contain a cycle: ticket(s) ${pending.map((r) => r.i + 1).join(', ')} can never start`)
      break
    }
    for (const r of ready) order.push(r.i)
    pending = pending.filter((r) => !ready.includes(r))
  }
  if (drafted.length && !drafted.some((t) => !(t.blocked_by_index || []).length)) {
    problems.push('every ticket is blocked by another -- nothing can start')
  }
  return { problems, order }
}

// The verification a ticket's own CI runs on. Same list as run-merge-train's HARNESS, and
// gates.test.mjs asserts the two agree: a ticket that rewrites the gate it must pass has no
// trustworthy green, so it is a human's commit no matter how confident the drafter was.
const HARNESS = ['.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/gate.sh', '.pre-commit-config.yaml']

// Which tickets a person still owes a decision on. This used to be all of them: every issue
// was filed gated and a human unpicked them one at a time, which made the gate a chore
// rather than a signal -- and a chore gets cleared without being read. Gate the few that
// genuinely wait on a person, ship the rest ready, and the label means something again.
const gateFor = (ticket, { specUnsound, refutedCount }) => {
  if (specUnsound) return 'the spec it slices was returned "not-ready" by the critic -- the design is what needs a person, not this slice'
  if (refutedCount) return `a skeptic refuted ${refutedCount} finding(s) this design rests on -- the premise is open`
  const harness = (ticket.files_named || []).filter((f) => HARNESS.some((h) => String(f).replace(/\\/g, '/').toLowerCase().endsWith(h)))
  if (harness.length) return `it rewrites the harness (${harness.join(', ')}) -- its own CI cannot vouch for it`
  if (ticket.needs_human) return ticket.needs_human_why || 'the drafter flagged it for a human and did not say why'
  return ''
}

// The labels the filer actually applied, re-read rather than believed. A prompt saying
// "carry exactly ONE of the two" is advisory; this is not. It has already gone wrong in
// practice -- four tickets of one spec were filed carrying BOTH labels, and because
// run-merge-train's admit() checks the gate label first, the pair reads as "gated" and the
// whole remaining train silently refused to board. Nothing errored. Nobody found out until
// somebody asked why the train had not moved.
const labelCheck = (rows) => {
  const problems = []
  for (const r of rows) {
    const has = (l) => (r.labels || []).includes(l)
    if (has(GATE_LABEL) && has(READY_LABEL)) {
      problems.push(`#${r.number} carries BOTH "${GATE_LABEL}" and "${READY_LABEL}" -- the gate label wins everywhere that reads it, so this ticket would never board`)
    } else if (!has(GATE_LABEL) && !has(READY_LABEL)) {
      problems.push(`#${r.number} carries NEITHER "${GATE_LABEL}" nor "${READY_LABEL}" -- run-merge-train requires the ready label to be present, so this ticket would never board`)
    } else if (has(GATE_LABEL) !== Boolean(r.expected_gate)) {
      problems.push(`#${r.number} was filed ${has(GATE_LABEL) ? 'gated' : 'ready'} but the script decided ${r.expected_gate ? 'gated' : 'ready'}`)
    }
  }
  return problems
}

const drafted = draft.tickets
const { problems, order } = checkHandover(draft.spec_body, spec.seams, drafted)

for (const p of problems) log(`body check: ${p}`)

const gateReason = drafted.map((t) =>
  GATE_ALL
    ? 'gateAll was set on this run -- every ticket filed gated'
    : gateFor(t, { specUnsound: critic?.verdict === 'not-ready', refutedCount: refuted.length }),
)
const gatedCount = gateReason.filter(Boolean).length
drafted.forEach((t, i) => gateReason[i] && log(`#${i + 1} "${t.title}" filed gated: ${gateReason[i]}`))
log(`${drafted.length - gatedCount}/${drafted.length} ticket(s) ready for an agent, ${gatedCount} waiting on a person`)
// A run that gates most of what it drafted did not answer the question it was given; it
// wrote the question down again. Say so where the person launching it will see it.
if (gatedCount > drafted.length / 2) {
  log(`most tickets are gated -- read the reasons above as one complaint about the spec, not ${gatedCount} separate ones`)
}

// Everything the human triaging this needs to see, prepended in JS rather than re-narrated
// by an agent. The verdict travels in the issue, which is where the ungating decision is
// actually made -- the run report is a local file that dies on a fresh clone.
const BANNER = [
  critic?.verdict && critic.verdict !== 'ready'
    ? [
        `> **An adversarial reviewer returned "${critic.verdict}" on this spec.** The body below may already address these; check before removing the \`${GATE_LABEL}\` label.`,
        ...(critic.must_fix || []).map((m) => `> - ${m.problem} — *fix:* ${m.fix}`),
      ].join('\n')
    : '',
  refuted.length
    ? [
        '> **A skeptic refuted findings this design rests on.** Treat them as open questions, not settled facts:',
        ...refuted.map((r) => `> - ~~${r.claim}~~ — ${r.why}${r.correction ? ` *Instead:* ${r.correction}*` : ''}`),
      ].join('\n')
    : '',
]
  .filter(Boolean)
  .join('\n>\n')

const SPEC_BODY = [BANNER, draft.spec_body].filter(Boolean).join('\n\n')

let filedRes = null
const notFiling = !FILE ? 'file: false' : problems.length ? `${problems.length} problem(s) in the drafted bodies` : ''
if (notFiling) {
  if (problems.length) log('NOT FILING: the handover contract is not met -- see the body check above')
} else {
  filedRes = await call(
    [
      `You are filing an approved spec and its tickets as GitHub issues for the GraphKnows repository. ${GH}`,
      'The text is FINAL. Do not rewrite it, do not improve it, do not add a heading or a footer. You are creating issues from bodies somebody else wrote.',
      '',
      'STEP 1 -- check for duplicates first: `gh issue list --state all --search "<key words from the spec title>" --limit 30`. If an OPEN issue already asks for one of the slices below, comment the parent reference on it instead of filing a second one, and report it with duplicate_of set.',
      '',
      `STEP 2 -- file the spec as the parent issue:\n\`gh issue create --title "${spec.title}" --label "${GATE_LABEL}" --label "${KIND}" --body "<the SPEC BODY below, verbatim>"\`\nRecord its number; every ticket references it.`,
      REQUEST
        ? `  The spec answers request #${REQUEST}. Append one line to the END of the spec body: "Requested in #${REQUEST}." -- that is the whole edit, and it is what makes the chain request -> spec -> ticket -> PR followable in either direction without a matrix maintained by hand.`
        : '',
      '',
      `STEP 3 -- file each ticket, in the order given below (blockers first, so every blocker already has a number). For each one:`,
      `  \`gh issue create --title "<title>" --label "<the gate label named in that ticket's block below>" --label "${KIND}" --parent <the spec issue number> --body "<its body, verbatim>"\``,
      '  The --parent flag makes the ticket a GitHub sub-issue of the spec. This is load-bearing: the run-merge-train workflow rosters a train by reading the spec\'s sub-issues, so a ticket filed without it is invisible to the train. If --parent fails (old gh), file without it and then run `gh issue edit <spec number> --add-sub-issue <ticket number>`; if that also fails, say so in notes.',
      '  Three edits to each body, and only these three: replace the words "the parent spec issue" in the "## Parent" line with the real spec issue number; APPEND a "## Blocked by" section listing the real issue numbers of the tickets named as its blockers below (or the words "None - can start immediately" when it has none); and APPEND the "Estimated diff:" line each ticket block gives you, verbatim, on its own line at the very end.',
      // The size estimate has to reach the BODY or it dies with this run. run-merge-train
      // reads it back to decide whether a ticket can ride the express lane -- several tiny
      // tickets sharing one branch, one CI run and one merge instead of paying ~31 minutes
      // of gate-and-CI each. A ticket with no such line takes the long road, which is the
      // right default for anything filed by hand.
      '  The "Estimated diff:" line is machine-read by run-merge-train. Do not reword it, do not round it, and do not add one to a ticket whose block below has none.',
      `  The "## Blocked by" section is load-bearing: process-issues reads it and refuses to pick up a ticket whose blockers are still open.`,
      `  Also mention in each body that it was drafted by ${RUN}.`,
      '',
      `The SPEC issue always gets "${GATE_LABEL}" and never "${READY_LABEL}": it is a document, not a slice of work, and the implementing loop must never pick it up as one.`,
      `Each TICKET carries exactly ONE of the two labels, and its block below names which. "${READY_LABEL}" means an agent may start on it; "${GATE_LABEL}" means a person owes it a decision first, and that block says which decision. Apply the label you are given and no other -- do not add the second one, do not swap them because a ticket looks large, and do not re-decide this. The choice was made in the script from data you cannot see.`,
      'Report every ticket by the INDEX it carries below, with the number you created. Set filed once every issue exists.',
      '',
      '<<<SPEC BODY>>>',
      SPEC_BODY,
      '<<<END SPEC BODY>>>',
      '',
      ...order.map((i) => {
        const t = drafted[i]
        const blockers = (t.blocked_by_index || []).filter((b) => b !== i && drafted[b])
        // A gated ticket carries its reason IN THE BODY. The label says a person owes it
        // something; only the body can say what, and the label is where the answer gets
        // given -- a reason left in the run report is a reason nobody reads.
        const body = gateReason[i]
          ? `> **Gated: a person owes this ticket a decision before an agent starts.**\n> ${gateReason[i]}\n>\n> Answer it in a comment, then swap \`${GATE_LABEL}\` for \`${READY_LABEL}\`.\n\n${t.body}`
          : t.body
        return [
          `<<<TICKET index=${i}>>>`,
          `title: ${t.title}`,
          `label: ${gateReason[i] ? GATE_LABEL : READY_LABEL}`,
          t.expected_lines > 0
            ? `Estimated diff: ~${t.expected_lines} lines across ${(t.files_named || []).length || 1} file(s).`
            : 'no Estimated diff line -- the drafter gave no estimate, so do not invent one',
          `blocked by: ${blockers.length ? blockers.map((b) => `ticket index ${b} ("${drafted[b].title}")`).join(', ') : 'nothing -- can start immediately'}`,
          'body:',
          body,
          '<<<END TICKET>>>',
        ].join('\n')
      }),
    ]
      .filter(Boolean)
      .join('\n'),
    { label: 'tickets:file', phase: 'Tickets', schema: FILED_SCHEMA },
  )
}

// Label verify+repair rides in the wrap-up agent below: the expected label per issue is
// computed here from data the agent cannot see, so it has nothing to decide -- and a
// dedicated agent paid a full session context to run one gh loop.
const filedNumbers = filedRes?.filed ? (filedRes.tickets || []).filter((t) => t.number > 0 && !t.duplicate_of) : []
let labelProblems = []

// Index -> real issue number, resolved in the script. The blocking edges in the return
// value are therefore computed, not claimed.
const numberOf = new Map((filedRes?.tickets || []).filter((t) => t.number > 0).map((t) => [t.index, t.number]))
const ticketDetail = drafted.map((t, i) => ({
  number: numberOf.get(i) || 0,
  title: t.title,
  delivers: t.delivers,
  blocked_by: (t.blocked_by_index || []).map((b) => numberOf.get(b) || 0).filter((n) => n > 0),
  // So the caller can hand the ready ones straight on without re-reading GitHub.
  ready: !gateReason[i],
  gate_reason: gateReason[i] || '',
}))
const tickets = {
  filed: filedRes?.filed === true,
  spec_issue: filedRes?.spec_issue || 0,
  spec_url: filedRes?.spec_url || '',
  tickets: ticketDetail,
  wide_refactor: draft.wide_refactor === true,
  notes: filedRes?.notes || draft.notes || '',
}

const filed = ticketDetail.map((t) => t.number).filter((n) => n > 0)
log(
  tickets.filed
    ? `filed spec #${tickets.spec_issue} and ${filed.length} ticket(s): ${filed.map((n) => '#' + n).join(', ')}`
    : `drafted ${drafted.length} ticket(s), filed nothing${notFiling ? ` (${notFiling})` : ''}`,
)

// The one routine thing that comes back to a person: what the request did not settle. It
// goes on the REQUEST issue, not on a ticket -- the requester is the only one who can
// answer, and they are not reading the tickets. One comment, every question, each with the
// default the run used meanwhile, so an unanswered question is a decision already made and
// visibly attributed rather than a silent assumption.
const CLAR_COMMENT =
  REQUEST && FILE && clarifications.length
    ? [
        `**${clarifications.length} question(s) this request does not settle.** ${RUN} carried on using the defaults below, so nothing is blocked — but if a default is wrong, the work built on it will be too.`,
        '',
        ...clarifications.map((c) => `- **${c.question}**\n  Assumed meanwhile: ${c.default_if_unanswered || 'nothing — the pipeline guessed'}`),
        '',
        tickets.filed
          ? `Answer here and re-run \`{ request: ${REQUEST} }\` to re-spec, or edit the affected ticket directly: ${filed.map((n) => '#' + n).join(', ')}.`
          : 'Answer here and re-run to file.',
      ].join('\n')
    : ''
// -- Wrap-up -------------------------------------------------------
// One agent for everything after filing: label verify+repair, the clarifications comment,
// the specs/<NNN>-<slug>/ mirror, and the run report. All of it is mechanical -- every
// body and every expected label is computed here -- and each task used to pay a full
// session context for an agent of its own.
const REPORT_PATH = opt.reportPath || `${RUNS_DIR}/prepare-tickets-${SLUG || 'run'}.md`

// The on-disk mirror of what was filed, in spec-kit's layout. specs/<NNN>-<slug>/ is
// committed, so the speckit skills (analyze, checklist, clarify, converge) can be run on
// this feature afterwards. The GitHub issues stay the contract process-issues reads; the
// mirror is written only when they exist, so a preview run leaves no trace here either.
const SPEC_MD = tickets.filed
  ? [`<!-- Filed as spec issue #${tickets.spec_issue}${REQUEST ? `, requested in #${REQUEST}` : ''} by ${RUN} -->`, '', SPEC_BODY].join('\n')
  : ''
// plan.md is the KEY to the whole mirror, not a nicety: check-prerequisites.ps1 exits 1
// without it, so every skill that gates on it -- analyze, checklist, clarify, converge,
// implement -- aborted at step 1 and the mirror was a layer nothing could read. Composed
// here from what the run already knows rather than by an agent: every line below is a fact
// this script is holding, and an agent asked to restate them would only get them wrong.
const PLAN_MD = tickets.filed
  ? [
      `# Implementation Plan: ${spec.title}`,
      '',
      '## Summary',
      '',
      synth.chosen,
      '',
      '## Technical Context',
      '',
      '**Project Type**: single project, flat layout -- a distributed Python package, not an application.',
      `**Constraints**: no new third-party dependency unless a ticket declares one in pyproject.toml. The verification harness (${HARNESS.join(', ')}) is off limits to every ticket below.`,
      `**Source of truth**: the GitHub issues under spec #${tickets.spec_issue}. This directory mirrors them; where the two disagree, the issues win.`,
      '',
      '## Constitution Check',
      '',
      critic?.verdict === 'not-ready'
        ? `FAILED. An adversarial reviewer called this design unsound; every ticket was filed behind the ${GATE_LABEL} label and a person owes it a decision before any of them starts.`
        : `Passed. .specify/memory/constitution.md was not contradicted by the chosen design, and an adversarial reviewer returned "${critic?.verdict || 'no verdict'}".`,
      '',
      '## Project Structure',
      '',
      '### Documentation (this feature)',
      '',
      '```',
      'spec.md      the filed spec, byte for byte',
      'plan.md      this file',
      'research.md  what the research lanes found, and what a skeptic did to it',
      'tasks.md     the filed tickets, in dependency order',
      '```',
      '',
      '### Source Code (repository root)',
      '',
      ...(() => {
        const files = [...new Set(order.flatMap((i) => drafted[i].files_named || []))].sort()
        return files.length ? ['```', ...files, '```'] : ['No ticket named a file path; see each issue for where its change lands.']
      })(),
      '',
      '## Complexity Tracking',
      '',
      `${order.length} ticket(s). ${order.filter((i) => gateReason[i]).length} gated behind a human decision; the rest were filed ready.`,
      synth.adr_written ? `The decision itself is recorded in \`${synth.adr_path}\`.` : 'No ADR: the decision was not architecturally significant.',
    ].join('\n')
  : ''

const RESEARCH_MD = tickets.filed
  ? [`# Research -- ${TOPIC}`, BRIEF, CONTEST_NOTE, PROTO_NOTE].filter(Boolean).join('\n\n')
  : ''
const TASKS_MD = tickets.filed
  ? [
      `# Tasks: ${spec.title}`,
      '',
      `Mirror of the tickets under spec issue #${tickets.spec_issue}; the issues are the source of truth.`,
      '',
      ...order.map((i, k) => {
        const t = drafted[i]
        const n = numberOf.get(i)
        const blockers = (t.blocked_by_index || []).map((b) => numberOf.get(b) || 0).filter((x) => x > 0)
        return [
          `- [ ] T${String(k + 1).padStart(3, '0')} ${t.title}${n ? ` (#${n})` : ''}`,
          `  - Blocked by: ${blockers.length ? blockers.map((x) => `#${x}`).join(', ') : 'nothing -- can start immediately'}`,
          gateReason[i] ? `  - Gated: ${gateReason[i]}` : null,
        ]
          .filter(Boolean)
          .join('\n')
      }),
    ].join('\n')
  : ''

// The run report. The research note holds the sources; this holds the outcome.
let REPORT_BODY = ''
if (DO_REPORT) {
  REPORT_BODY = [
    `# prepare-tickets -- ${TOPIC}`,
    '',
    `${RUN}. ${
      tickets.filed
        ? `Spec #${tickets.spec_issue} and ${filed.length} ticket(s) filed: ${filed.length - gatedCount} ready for an agent, ${gatedCount} gated behind \`${GATE_LABEL}\`.`
        : problems.length
          ? `**Nothing was filed**: the drafted bodies failed the handover check (${problems.length} problem(s), listed below).`
          : 'Nothing was filed: this was a preview run.'
    }`,
    '',
    '## Tickets',
    '',
    (tickets?.tickets || []).length ? '| # | Ticket | Delivers | Blocked by |' : 'None were drafted.',
    (tickets?.tickets || []).length ? '| --- | --- | --- | --- |' : null,
    ...(tickets?.tickets || []).map(
      (t) =>
        `| ${t.number ? '#' + t.number : 'draft'} | ${t.title} | ${t.delivers} | ${
          (t.blocked_by || []).map((n) => '#' + n).join(', ') || 'nothing -- can start now'
        } |`,
    ),
    tickets?.wide_refactor
      ? '\nThis is an expand-migrate-contract sequence, not a set of tracer bullets: the batches only promise green together.'
      : null,
    '',
    problems.length ? '## Why nothing was filed' : null,
    problems.length ? '' : null,
    problems.length
      ? [
          'The issue body is the only contract the process-issues half reads, so a body missing a section is a broken handover. These were refused:',
          '',
          ...problems.map((x) => `- ${x}`),
          '',
          'Re-run to draft them again. Nothing was created, so there is nothing to clean up.',
        ].join('\n')
      : null,
    problems.length ? '' : null,
    '## Seams',
    '',
    (spec.seams || []).length
      ? `Highest first: ${spec.seams.map((x) => `\`${x}\``).join(', ')}. These live in the spec body's Testing Decisions section, which is where the implementing workflow reads them from.`
      : 'The spec named no seams.',
    '',
    '## The design that won',
    '',
    synth.chosen,
    (synth.grafted || []).length ? `\nGrafted from the runners-up: ${synth.grafted.join('; ')}.` : null,
    (synth.risks || []).length ? `\nKnown risks: ${synth.risks.join('; ')}.` : null,
    '',
    '## The designs that lost',
    '',
    (synth.rejected || []).length
      ? synth.rejected.map((r) => `- **${r.approach}** -- ${r.reason}`).join('\n')
      : 'Only one design reached the synthesis.',
    '',
    '## Evidence',
    '',
    `${findings.length} finding(s) across ${lanes.length} research lane(s)${
      synth.adr_written
        ? `; the decision and what it costs are recorded in \`${synth.adr_path}\``
        : ` -- no ADR: ${synth.adr_skipped_why || 'the decision is not architecturally significant'}`
    }.`,
    openQuestions.length
      ? `\nStill open:\n${openQuestions.map((q) => `- ${q}`).join('\n')}`
      : '\nThe research lanes left nothing unverified.',
    '',
    '## Prototype',
    '',
    proto?.ran
      ? [
          `Question: ${protoQ}`,
          `Verdict: **${proto.verdict}**`,
          `What it showed: ${proto.what_it_showed}`,
          (proto.design_changes || []).length
            ? `The design had to change: ${proto.design_changes.join('; ')}`
            : 'The design survived intact.',
          proto.branch ? `It lives on \`${proto.branch}\` -- run it with \`${proto.run_command || 'see the branch'}\`.` : null,
        ]
          .filter((x) => x !== null)
          .join('\n\n')
      : 'None was built: no design question needed running code to settle.',
    '',
    '## Contested findings',
    '',
    contested.length
      ? refuted.length
        ? `${refuted.length} of ${contested.length} load-bearing finding(s) were **refuted** by a skeptic and are stamped at the top of the spec issue:\n\n` +
          refuted.map((r) => `- ~~${r.claim}~~\n  - ${r.why}${r.correction ? `\n  - instead: ${r.correction}` : ''}`).join('\n')
        : `All ${contested.length} load-bearing finding(s) survived a skeptic sent to refute them.`
      : 'The synthesis named no load-bearing findings, so nothing was contested.',
    '',
    '## The critic',
    '',
    `Verdict: **${critic?.verdict || 'none'}**.`,
    (critic?.must_fix || []).length
      ? `\nFolded into the spec before filing:\n${critic.must_fix.map((m) => `- ${m.problem}\n  - fix: ${m.fix}`).join('\n')}`
      : '\nNothing had to change.',
    (critic?.missing || []).length ? `\nStill missing: ${critic.missing.join('; ')}.` : null,
    (critic?.contradictions || []).length ? `\nContradicts the research: ${critic.contradictions.join('; ')}.` : null,
    '',
    '## Next',
    '',
    filed.length
      ? [
          gatedCount
            ? `- ${gatedCount} ticket(s) are gated and say why at the top of their body. Answer the question in a comment, then swap \`${GATE_LABEL}\` for \`${READY_LABEL}\`.`
            : `- Nothing is gated: every ticket is ready for an agent.`,
          `- Skim ${['#' + (tickets?.spec_issue || 0), ...filed.map((n) => '#' + n)].join(', ')} and close anything not worth doing -- the ready ones will be picked up as they stand.`,
          `- Then hand them to process-issues as \`{ issues: [${filed.join(', ')}] }\`, or drain the whole spec with run-merge-train as \`{ spec: ${tickets?.spec_issue || 0}, merge: true }\`.`,
        ].join('\n')
      : '- Nothing was filed. Read the drafted tickets above and re-run with filing enabled if they hold up.',
    critic?.verdict === 'not-ready' ? '- **The critic called the spec not-ready.** It is stamped at the top of the spec issue; read it before ungating anything.' : null,
    proto?.ran && proto.branch
      ? `- Delete the throwaway prototype branch when you are done with it: \`git branch -D ${proto.branch}\`. Nothing downstream reads it, and nothing else will clean it up.`
      : null,
  ]
    .filter((x) => x !== null)
    .join('\n')
}

const wrapTasks = []
if (filedNumbers.length) {
  wrapTasks.push(
    [
      'Verify and repair the labels on the issues below. For each: `gh issue view <n> --json number,labels`; if it already carries exactly the required label and not the forbidden one, leave it alone, otherwise `gh issue edit <n> --add-label "..." --remove-label "..."` until it does. Change nothing else on them -- no comments, no body edits, no closing.',
      ...filedNumbers.map(
        (t) => `  #${t.number}: must carry "${gateReason[t.index] ? GATE_LABEL : READY_LABEL}" and NOT "${gateReason[t.index] ? READY_LABEL : GATE_LABEL}".`,
      ),
      'These two labels are mutually exclusive. An issue carrying both is treated as gated by every reader and never gets worked.',
      'Report the FINAL labels on each issue under issues, exactly as GitHub returns them after your edits, and every repair you made under repaired.',
    ].join('\n'),
  )
}
if (CLAR_COMMENT) {
  wrapTasks.push(
    [
      `Post ONE comment on GitHub issue #${REQUEST} -- do not close it, do not relabel it, do not answer the questions yourself.`,
      `\`gh issue comment ${REQUEST} --body "<the text below, verbatim>"\``,
      '<<<COMMENT>>>',
      CLAR_COMMENT,
      '<<<END COMMENT>>>',
    ].join('\n'),
  )
}
if (SPEC_MD) {
  wrapTasks.push(
    [
      `Mirror the filed spec into the repository, in spec-kit's layout. From ${MAIN_REPO} run \`powershell -ExecutionPolicy Bypass -File .specify/scripts/powershell/create-new-feature.ps1 -Json -ShortName "${SLUG}" "<the topic, quoted safely>"\` where the topic is: ${TOPIC}. Parse the JSON it prints for the feature directory it created (specs/<NNN>-<slug>). It seeds a template spec.md there; overwrite it.`,
      'Into that directory write the four payloads below as spec.md, plan.md, research.md and tasks.md -- byte for byte, markers excluded. Report the directory, relative to the repo root, under feature_dir.',
      'All four matter: the speckit skills refuse to run on a feature directory that has no plan.md, whatever else is in it.',
      '<<<spec.md>>>',
      SPEC_MD,
      '<<<end spec.md>>>',
      '<<<plan.md>>>',
      PLAN_MD,
      '<<<end plan.md>>>',
      '<<<research.md>>>',
      RESEARCH_MD,
      '<<<end research.md>>>',
      '<<<tasks.md>>>',
      TASKS_MD,
      '<<<end tasks.md>>>',
    ].join('\n'),
  )
}
if (REPORT_BODY) {
  wrapTasks.push(
    [
      `Write the run report to ${MAIN_REPO}/${REPORT_PATH} -- create the directory if it is missing, overwrite the file if it exists. Content: everything between the two marker lines, byte for byte; the markers are NOT part of the file.`,
      '<<<REPORT-BEGINS>>>',
      REPORT_BODY,
      '<<<REPORT-ENDS>>>',
      'Then confirm the file exists and report its path and its size in bytes under notes.',
    ].join('\n'),
  )
}

let wrap = null
if (wrapTasks.length) {
  wrap = await call(
    [
      `${wrapTasks.length} task(s), in order, and nothing else. ${GH}`,
      'Every payload below is FINAL. Write the bytes as given: do not reformat, do not fix grammar, do not add headings, footers or notes about yourself.',
      ...wrapTasks.map((t, i) => `\nTASK ${i + 1} -- ${t}`),
    ].join('\n'),
    {
      label: 'wrap-up',
      phase: 'Report',
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['issues'],
        properties: {
          issues: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['number', 'labels'],
              properties: { number: { type: 'integer' }, labels: { type: 'array', items: { type: 'string' } } },
            },
          },
          repaired: { type: 'array', items: { type: 'string' } },
          feature_dir: { type: 'string', description: 'the specs/<NNN>-<slug> directory the mirror was written to, or empty' },
          notes: { type: 'string' },
        },
      },
    },
  )
  for (const r of wrap?.repaired || []) log(`label repair: ${r}`)
  if (filedNumbers.length) {
    const expected = new Map(filedNumbers.map((t) => [t.number, Boolean(gateReason[t.index])]))
    labelProblems = labelCheck((wrap?.issues || []).map((r) => ({ ...r, expected_gate: expected.get(r.number) })))
    for (const p of labelProblems) log(`label check STILL WRONG after repair: ${p}`)
  }
  if (CLAR_COMMENT) log(`posted ${clarifications.length} open question(s) to #${REQUEST}`)
  if (SPEC_MD) log(`spec mirror: ${wrap?.feature_dir || 'directory not reported -- check specs/'}`)
  if (REPORT_BODY) log(`run report: ${REPORT_PATH}`)
}

// The handover to process-issues. `tickets` is the array it takes as args.from.tickets.
// The ones marked ready can be worked as they stand -- they trace to a request a person
// filed, which is what the gate was ever protecting. The gated ones say in their own body
// which decision they are waiting on, and that refusal is the design, not a bug to work
// around.
return {
  report: DO_REPORT ? REPORT_PATH : 'not written',
  feature_dir: wrap?.feature_dir || '',
  request: REQUEST,
  topic: TOPIC,
  screened: screen?.verdict || (REQUEST ? 'not run' : 'no request to screen'),
  depth: SETTLED ? 'shallow -- the request settled the approach' : 'full',
  clarifications,
  spec_issue: tickets?.spec_issue || 0,
  spec_url: tickets?.spec_url || '',
  spec_title: spec.title,
  tickets: filed,
  ticket_detail: (tickets?.tickets || []).map((t) => ({
    number: t.number || 0,
    title: t.title,
    blocked_by: t.blocked_by || [],
  })),
  wide_refactor: tickets?.wide_refactor === true,
  chosen_design: synth.chosen,
  rejected: synth.rejected || [],
  adr: synth.adr_written ? synth.adr_path : '',
  prototype: proto?.ran ? { branch: proto.branch, verdict: proto.verdict, run: proto.run_command } : 'not built',
  open_questions: openQuestions,
  critic: { verdict: critic?.verdict || 'no verdict', must_fix: (critic?.must_fix || []).length, missing: critic?.missing || [] },
  contested: contested.length,
  refuted: refuted.map((r) => r.claim),
  body_check: problems,
  needs_human:
    critic?.verdict === 'not-ready' ||
    problems.length > 0 ||
    refuted.length > 0 ||
    tickets.filed !== true ||
    !filed.length ||
    clarifications.length > 0,
  next_step: filed.length
    ? `${filed.length - gatedCount} of ${filed.length} ticket(s) are labelled "${READY_LABEL}" and can be handed to process-issues as { from: <this result> } or { issues: [<numbers>] }, or drained together with run-merge-train as { spec: ${tickets?.spec_issue || 0}, merge: true }. ${gatedCount ? `${gatedCount} carry "${GATE_LABEL}" and say at the top of the body which decision they are waiting on; answer it, then swap the label.` : 'Nothing is gated.'} Skim them all and close anything not worth doing.`
    : 'nothing was filed -- read the drafted tickets in the return value and decide whether to re-run with file enabled',
}
