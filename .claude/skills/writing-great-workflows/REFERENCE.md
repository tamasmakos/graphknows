# Running workflows

Operational reference for dynamic workflows. Writing guidance lives in [`SKILL.md`](SKILL.md).

Items marked `†` come from hands-on reverse-engineering rather than Anthropic's docs — verify before relying on them.

[Availability](#availability) · [Triggering](#triggering) · [Permissions](#permissions) · [Watching a run](#watching-a-run) · [Resume](#resume) · [Saving and distribution](#saving-and-distribution) · [Cost](#cost) · [Where the writing guidance comes from](#where-the-writing-guidance-comes-from)

## Availability

Claude Code v2.1.154+ (`/effort ultracode` needs v2.1.203; the size guideline v2.1.202/2.1.219). All paid plans, the Anthropic API, Bedrock, Google Agent Platform and Microsoft Foundry. On Pro it ships off — turn on the **Dynamic workflows** row in `/config`. Surfaces: CLI, desktop, IDE extensions, `claude -p`, and the Agent SDK's Workflow tool (SDK v0.3.149+).

## Triggering

| Method | Notes |
|---|---|
| `ultracode` in the prompt | One-off opt-in; natural phrasing ("use a workflow", "run a workflow") works too. Pre-v2.1.160 the keyword was `workflow` |
| `/effort ultracode` | `xhigh` effort plus auto-orchestration for every substantive task in the session; `claude --effort ultracode` starts that way, the `ultracode` setting makes it permanent |
| `/deep-research <question>` | Bundled workflow: fan out searches → fetch → 3-vote adversarial verify → cited report. Needs WebSearch |
| `/<name>` | Saved and plugin workflows |
| `Workflow` tool | Programmatic |

The keyword is honored only from human-typed input — interactive prompt, IDE panel, Remote Control, or SDK input stamped `origin: { kind: "human" }`. Since v2.1.210 it is ignored from `-p`, unstamped SDK prompts, scheduled tasks and webhook/PR-comment input.

Dismiss the highlight with `Option+W` / `Alt+W`, or backspace right after it. Turn it off entirely via **Ultracode keyword trigger** in `/config`.

## Permissions

The launch prompt lists the planned phases and offers **Yes, run it** / **Yes, don't ask again for `<name>` in `<path>`** / **View raw script** (`Ctrl+G` opens it in your editor) / **No**; `Tab` edits the prompt first.

Frequency depends on mode: Auto asks on first launch only, and skips it entirely under ultracode; Manual and acceptEdits ask every run unless you chose "don't ask again"; bypassPermissions, `claude -p` and the SDK never ask.

Spawned subagents always run in `acceptEdits` and inherit your tool allowlist whatever your session mode is. File edits are auto-approved, but an unallowlisted shell command, WebFetch or MCP call still prompts mid-run — pre-allowlist what your agents will need before a long fan-out.

## Watching a run

`/workflows`, select the run, `Enter`. A one-line summary also sits in the task panel below the input box.

| Key | Action |
|---|---|
| `↑`/`↓` | Select phase or agent |
| `Enter` / `→` | Drill into a phase, then an agent (prompt, recent tool calls, result) |
| `Esc` / `←` | Back out (`Esc` only on v2.1.203–2.1.205) |
| `j`/`k` | Scroll agent detail |
| `f` | Cycle the status filter |
| `p` | Pause / resume |
| `x` | Stop the selected agent, or the whole run when focused on it |
| `r` | Restart the selected running agent |
| `s` | Save the run's script as a command |

## Resume

Two rules decide what survives a stop. An agent still running when you stopped is not cached and re-runs. Replay follows **start order**: cached results stop at the first unfinished agent, and everything that started after it re-runs even if it finished.

So stopping mid fan-out is expensive, and many small agents preserve far more progress than one long one. Resume from `/workflows` with `p`, or ask Claude to relaunch — same session only, since exiting Claude Code starts the workflow fresh.

Every run persists its script under the session directory and returns the path, so you can read, diff and edit it, then relaunch with `{ scriptPath, resumeFromRunId }`. Before diagnosing why a completed run returned something empty or odd, read `journal.jsonl` in the transcript dir — it records each agent's actual return value.

## Saving and distribution

`/workflows` → select → `s`, with `Tab` toggling location: `.claude/workflows/` is project scope, committed and shared; `~/.claude/workflows/` is personal across all projects and honors `CLAUDE_CONFIG_DIR`. It then runs as `/<name>`, with project scope winning a name collision.

In monorepos (v2.1.178+) it saves to the nearest existing `.claude/workflows/` between cwd and repo root, loads from all of them along that path, and the closest wins. A symlink guard refuses project scope if `.claude`, `.claude/workflows` or the target is a symlink, and personal scope if the target file is (v2.1.216+).

Plugins put the script in `workflows/` at plugin root, or name it in the `workflows` manifest field; it runs namespaced as `/acme-tools:release-audit`.

Args arrive as real JSON — "Run /triage-issues on issues 1024, 1025, 1030" becomes structured data the script reads from the `args` global and calls array and object methods on directly. Pass actual JSON values, never a JSON-encoded string.

## Cost

A run costs far more than the same task in conversation — Anthropic measures multi-agent research runs at roughly fifteen times a chat turn, where a single subagent sits nearer four — and counts against plan usage and rate limits. Agents use the session model unless the script routes a stage elsewhere; `CLAUDE_CODE_SUBAGENT_MODEL` overrides both, and a model your org blocks is substituted with a warning in the progress view.

An advisory **large workflow** warning appears past 25 scheduled agents or 1.5M projected tokens (the threshold follows your size guideline, and ultracode suppresses it). It pauses nothing.

The size guideline is advice to Claude, not a cap — `/config` → Dynamic workflow size, `/config workflowSizeGuideline=small`, or the settings key (v2.1.219+, which takes precedence over `/config`).

| Value | Target |
|---|---|
| `small` | < 5 agents |
| `medium` | < 15 agents (default) |
| `large` | < 50 agents |
| `unrestricted` | Claude sizes to the task |

## Settings and environment

| Key | Effect |
|---|---|
| `enableWorkflows` | Per-user on/off against the plan default; what `/config` → Dynamic workflows writes |
| `disableWorkflows` | Org-wide off in managed settings; beats `enableWorkflows: true` |
| `workflowKeywordTriggerEnabled` | `false` lets you type `ultracode` without starting a run |
| `workflowSizeGuideline` | `small` \| `medium` \| `large` \| `unrestricted` |
| `ultracode` | Start every session in ultracode |
| `CLAUDE_CODE_DISABLE_WORKFLOWS=1` | Per-session off; whichever source disables wins |
| `CLAUDE_CODE_WORKFLOW_PREFIX_STAGGER_MS` | How long same-prefix siblings are held so they read the first agent's cached prefix; default `5000`, `0` disables (v2.1.229+) |
| `CLAUDE_CODE_SUBAGENT_MODEL` | Overrides every workflow agent's model |

Disabled, the bundled workflow commands vanish and `ultracode` stops triggering and leaves the `/effort` menu. Workflow-spawned agents emit `workflow.run_id` and `workflow.name` OTel attributes.

## The Workflow tool

```typescript
type WorkflowInput = {
  script?: string;           // inline; must start with the literal `export const meta = {...}`
  name?: string;             // bundled or .claude/workflows/ name
  scriptPath?: string;       // file on disk — takes precedence over script and name
  args?: unknown;            // real JSON, not a JSON-encoded string
  resumeFromRunId?: string;  // same session only
  title?: string;            // ignored — meta sets it
  description?: string;      // ignored — meta sets it
};

type WorkflowOutput = {
  status: "async_launched" | "remote_launched";
  taskId: string;
  taskType?: "local_workflow" | "remote_agent";
  workflowName?: string;
  runId?: string;         // pass as resumeFromRunId later; absent for remote runs
  summary?: string;
  transcriptDir?: string; // subagent transcripts
  scriptPath?: string;    // persisted script — edit and re-invoke to iterate
  sessionUrl?: string;    // remote_launched only
  warning?: string;
  error?: string;         // set = syntax check failed, the run did NOT start
};
```

At least one of `script`, `name` and `scriptPath` is required. The call returns immediately and the real result arrives later as a task completion — so always check `error`, since a syntax failure still returns `status: "async_launched"` with the run never having started.

`applyFlagSettings({ ultracode: true })` turns ultracode on mid-session (TS SDK only, CC v2.1.203+).

## Choosing the primitive

| | Subagents | Skills | Agent teams | Workflows |
|---|---|---|---|---|
| What it is | Workers Claude spawns | Instructions Claude follows | A lead agent supervising peer sessions | A script the runtime executes |
| Who decides the next step | Claude, turn by turn | Claude, per the prompt | The lead agent, turn by turn | The script |
| Intermediate results | Claude's context | Claude's context | A shared task list | Script variables |
| Repeatable unit | The worker definition | The instructions | The team definition | The orchestration |
| Scale | A few per turn | Same | A handful of peers | Dozens to hundreds per run |
| On interruption | Restarts the turn | Restarts the turn | Teammates keep running | Resumable, same session |

## Sources

- https://code.claude.com/docs/en/workflows
- https://code.claude.com/docs/en/agent-sdk/typescript — Workflow tool input/output
- https://code.claude.com/docs/en/settings-reference, /env-vars, /changelog
- `†` items: https://alexop.dev/posts/claude-code-workflows-deterministic-orchestration/

## Where the writing guidance comes from

The sizing ladder, the agent-prompt contract, the barrier bottleneck and the verification caveats in [`SKILL.md`](SKILL.md) come from these; re-read them before overturning any of it.

| Source | What it supports |
|---|---|
| [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Fan-out wins on breadth and loses on shared context or inter-agent dependency (coding especially); the 1 / 2–4 / 10+ agent ladder; thin task descriptions as the top failure source; the synchronous barrier bottleneck; checkpointing and tool-failure adaptation; a single-call 0–1 rubric judging closest to human judgement; ~20 real queries as an eval set; 15× and 4× token economics. Vendor-reported internal evals, no released methodology |
| [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) (MAST) | 14 failure modes over 1,600+ traces: disobeyed task and role specs, step repetition, unawareness of termination conditions, task derailment, premature termination, absent or incorrect verification. Best prompt and topology interventions recovered only +5–14%, and verification is "the last line of defense" that cannot repair upstream design flaws |
| [Rating Roulette](https://aclanthology.org/2025.findings-emnlp.1361.pdf) · [When LLMs Agree, Are They Right?](https://arxiv.org/pdf/2607.08065) · [SAGE](https://arxiv.org/pdf/2504.07385) | Judge self-consistency is low and prompting does not fix it; agreement between votes tracks shared bias as much as truth; majority-of-three moved one benchmark 79.4% → 80.6%, while evidence-augmented judging beat self-consistency by far. Hence evidence before votes — single benchmarks, treat the direction as sound and the magnitudes as thin |
| [Agent Skills best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) · [Claude Code skills docs](https://code.claude.com/docs/en/skills) | How this skill itself is shaped: description as the whole triggering surface, body under 500 lines, references one level deep, degrees of freedom matched to how fragile each step is |
