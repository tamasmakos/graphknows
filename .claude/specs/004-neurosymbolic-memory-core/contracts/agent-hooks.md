# Contract — Agent loop hooks

**Surface**: `python -m graphknows.integrations.claude_code.hooks <verb>`. Stdin is a JSON
event; stdout is a JSON `hookSpecificOutput` block or nothing. The process is stdlib-only
and talks to the running memory service through
`graphknows/integrations/client/` (FR-036).

The governing rule is FR-031: **injection is symbol-keyed**. Something is injected only when
a symbol resolves, through the same identity path used at ingest. There is no per-prompt
fuzzy budget. When nothing resolves, nothing is injected — in 100% of such cases (SC-009).

---

## Verbs

| Verb | Event | Payload used | Call | Output |
|---|---|---|---|---|
| `context` | SessionStart | `cwd`, `session_id` | recall the working-context block for `cwd`: open decisions and current values of functional predicates | `additionalContext` (FR-032) |
| `recall` | UserPromptSubmit | `prompt` | resolve symbols from the prompt, recall facts attached to them, newest first | `additionalContext`, or **nothing** when no symbol resolves (FR-033, FR-031) |
| `preview` | PreToolUse (Edit, Write) | `tool_input.file_path` | recall facts about the target path | `additionalContext` (FR-034) |
| `observe` | PostToolUse (Read, Grep, Glob) | `tool_response` | recall facts about the symbols observed | `additionalContext` (FR-034) |
| `remember` | Stop | `transcript_path` | ingest transcript records new since the last checkpoint | none (FR-035) |
| `catchup` | PreCompact | `transcript_path` | full catch-up ingest, then flush | none (FR-035) |

`preview` runs **before** the edit; `observe` runs **after** the read. That asymmetry is the
requirement, not an implementation detail: FR-034 wants what is known about a path before it
changes, and what is known about a symbol once it has actually been seen.

---

## Checkpointing (FR-026, SC-008)

The checkpoint is the last ingested record `uuid` per transcript source. `remember` and
`catchup` read forward from it.

- Records already seen are skipped by `uuid`, incrementing `records_skipped_duplicate`.
  Re-running a hook adds nothing.
- A transcript still being appended to may end in a partial line. It is counted as
  `records_skipped_malformed` and **does not advance the checkpoint past it** — a partial
  trailing record never corrupts the checkpoint (edge case 7).
- Unknown record or block types are counted and skipped, never fatal and never silent
  (FR-027, SC-014).

## What becomes memory

Kept as segments: user prompts, assistant text and thinking, and the agent's own writes.
Sub-agent transcripts follow the same rule and are marked as side chains.

**Not segments**: tool results. They are linked as `SOURCE`s a fact may cite (FR-028). A
fact citing a source that was never ingested still resolves; recall does not break on the
missing segment (edge case 2).

Retained per record: `timestamp`, `cwd`, `gitBranch`, `isSidechain` (FR-026).

---

## Dependency and time budget (SC-010)

- p95 under 1 s for a recall verb; hard ceiling 5 s enforced by the hook's own timeout. Measured by `python -m evaluation scenario hook-latency`.
- The hook package is a source module in `.importlinter`'s `client-stdlib-only` contract, so
  it cannot import `graphknows.ingestion`, `retrieval`, `storage`, `channels` or `memory`.
- A subprocess test imports the hook module and asserts `torch`, `transformers`, `spacy`,
  `sentence_transformers` and `gliner` are absent from `sys.modules`. That is the check that
  actually catches the regression — an accidental ML import arrives transitively, where no
  reviewer sees it.

---

## Settings block

```json
{"hooks": {
  "SessionStart":     [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks context"}]}],
  "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks recall"}]}],
  "PreToolUse":       [{"matcher": "Edit|Write",      "hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks preview"}]}],
  "PostToolUse":      [{"matcher": "Read|Grep|Glob",  "hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks observe"}]}],
  "Stop":             [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks remember"}]}],
  "PreCompact":       [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks catchup"}]}]
}}
```

## Shipped guidance (FR-038)

A skill that tells the agent **when to record and when to retrieve** ships with the
integration. It is guidance for the agent, not documentation for the maintainer, and it is
the reason `memory_recall` is not simply called on every turn.

## Verify

```bash
pytest tests/integrations/claude_code -q
lint-imports
echo '{"prompt":"nothing here resolves"}' | python -m graphknows.integrations.claude_code.hooks recall   # prints nothing
```
