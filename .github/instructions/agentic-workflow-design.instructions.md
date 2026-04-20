---
description: "Use when designing, building, or reviewing agentic AI workflows, multi-agent systems, LLM pipelines, orchestration logic, MCP servers, streaming agents, or tool-calling code. Enforces production-grade agentic engineering practices from arXiv:2512.08769 and Stack Overflow coding guidelines for AI agents."
applyTo: "**/agents/**,**/workflow/**,**/orchestration/**,**/mcp/**,**/prompts/**,**/lib/agent*,**/lib/llm*,**/lib/stream*"
---

# Agentic AI Workflow Design Guide

Based on [arXiv:2512.08769](https://arxiv.org/html/2512.08769v1) — *A Practical Guide for Designing, Developing, and Deploying Production-Grade Agentic AI Workflows* — and Stack Overflow's [Coding Guidelines for AI Agents](https://stackoverflow.blog/2026/03/26/coding-guidelines-for-ai-agents-and-people-too/).

> Complexity is the #1 enemy of agentic systems. LLM reasoning already adds non-determinism — keep your scaffolding dead-simple.

---

## The 9 Core Principles

### 1. Tool Calls Over MCP

Prefer direct tool calls over MCP integrations for agent-to-service communication.

MCP adds abstraction layers that cause agents to make ambiguous tool-selection decisions, misinterpret parameters, and produce non-deterministic failures.

```python
# ✅ Direct tool call — deterministic, debuggable
result = create_github_pr(branch=branch, title=title, body=body)

# ❌ MCP integration — agent must parse protocol metadata, non-deterministic
agent.run("Create a pull request for the generated content")  # via GitHub MCP
```

Use MCP only to **expose** your workflow to external clients (Claude Desktop, VS Code, LM Studio) — not inside the workflow itself.

---

### 2. Direct Function Calls Over Tool Calls

For operations that don't require language reasoning, **bypass the LLM entirely** and call functions directly from the orchestration layer.

Operations that should always be pure function calls (never tool calls):
- Writing to databases / files
- Posting to APIs
- Committing to git / creating PRs
- Generating timestamps
- Sending notifications

```python
# ✅ Pure function — deterministic, testable, zero token overhead
commit_files_to_repo(files=artifacts, branch=branch_name)

# ❌ Tool call — LLM must reason about parameters, adds token cost + variability
pr_agent.run(f"Commit these files: {artifacts}")
```

**Rule:** If the step requires no reasoning, it is not an agent step. Execute it as code.

---

### 3. One Agent, One Tool

Never attach multiple tools to a single agent. When an agent has multiple tools, LLMs:
- Call the wrong tool
- Skip tools entirely
- Invoke tools in the wrong order
- Misformat parameters

```python
# ✅ Two agents, each with one tool
scrape_agent = Agent(tools=[scrape_markdown])
publish_agent = Agent(tools=[publish_markdown])

# ❌ One agent, two tools — causes missed calls and ordering failures
combined_agent = Agent(tools=[scrape_markdown, publish_markdown])
```

---

### 4. Single-Responsibility Agents

Each agent must do exactly **one conceptual task**. Don't mix planning, generation, validation, and side-effectful execution in the same agent.

```python
# ✅ Separate agents — clear contracts, easy to test and debug
veo_json_builder = Agent(
    instructions=load_prompt("veo_json_builder"),
    # Only produces: valid Veo-3 JSON specification
)

# Then, deterministic execution — NOT an agent:
video_path = script_to_video(veo_json)  # Pure function

# ❌ Combined — blurs planning and execution, causes hallucinated paths/status
veo_agent = Agent(
    instructions="Generate the Veo JSON AND generate the video",
)
```

**Signs an agent has too many responsibilities:**
- Its output contract is vague ("JSON plus status messages")
- It calls external APIs AND transforms data
- Its prompt exceeds ~500 tokens of task description

---

### 5. Externalize All Prompts

**Never hardcode system prompts inside source files.** Store them in a `prompts/` directory as `.md` or `.txt` files and load dynamically at runtime.

```python
# ✅ Externalized — iterable without redeployment
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()

agent = Agent(instructions=load_prompt("podcast_script_generator"))

# ❌ Hardcoded — requires code change and redeployment to iterate
agent = Agent(instructions="""You are a podcast script generator.
    Your job is to...""")
```

**Prompt files are deployable artifacts:**
- Version-control them
- Review prompt changes the same as code changes
- Enable A/B testing and rollback without touching application code
- Allow non-engineers to refine agent behavior

---

### 6. Multi-Model Consortium for Critical Outputs

For high-stakes outputs, run multiple LLMs **in parallel** and use a dedicated **reasoning agent** to consolidate.

```python
# ✅ Consortium + reasoning consolidation
drafts = await asyncio.gather(
    openai_agent.run(context),
    claude_agent.run(context),
    gemini_agent.run(context),
)
final_output = reasoning_agent.run(consolidate_prompt(drafts))

# ❌ Single model — subject to hallucination, bias, and model drift
final_output = openai_agent.run(context)
```

The **reasoning agent's** only job is consolidation: conflict resolution, deduplication, factual alignment, and removing speculation. It does **not** generate new content.

Use for: chat answers, document summaries, any output that drives business decisions.

---

### 7. Separate Workflow Logic from MCP / Transport Layer

Keep orchestration logic in `lib/` (or equivalent). Route handlers and MCP servers are **thin adapters** — they forward requests, they don't contain logic.

```
# ✅ Clean separation
app/
  api/v1/agent/route.ts    ← thin adapter: parse request → call lib → stream response
lib/
  agent-workflow.ts        ← all orchestration, tool wiring, agent definitions
  streaming.ts             ← SSE/streaming utilities
mcp/
  server.ts                ← forwards MCP tool calls to REST API, nothing else

# ❌ Workflow logic inside the route handler or MCP server
```

This allows the workflow to be invoked from HTTP, MCP clients, CLI, or tests — without duplication.

---

### 8. Containerized Deployment

Package the workflow engine, MCP server, and supporting tools as **separate containers**.

```yaml
# docker-compose.yaml — separate concerns, independent scaling
services:
  workflow:
    build: ./services/workflow
    # Handles agent orchestration, LLM calls, tool execution

  mcp-server:
    build: ./services/mcp
    # Thin adapter: forwards MCP tool calls to workflow REST API

  worker:
    build: ./services/worker
    # Handles async jobs, retries, file processing
```

Benefits: independent scaling, blue/green deployments, clear security boundaries per service.

---

### 9. KISS — Keep It Simple

Agentic workflows are not traditional enterprise software. Do not apply:
- Deep class hierarchies
- Microservice-like decomposition for every sub-task
- Abstract base agents / plugin architectures
- Multiple indirection layers between the LLM call and the business logic

**Flat, readable, function-driven orchestration** wins every time.

```python
# ✅ Flat, readable pipeline — easy to trace, test, and extend
async def run_pipeline(topic: str, urls: list[str]) -> Artifacts:
    articles = await search_agent.run(urls)
    filtered = await filter_agent.run(articles, topic)
    content = await scrape_agent.run(filtered)
    drafts = await generate_drafts_consortium(content)
    script = await reasoning_agent.run(drafts)
    audio = await tts_agent.run(script)
    commit_artifacts(script, audio)           # Pure function — no agent
    return create_github_pr(artifacts)        # Pure function — no agent

# ❌ Over-engineered — abstract factory of agent builders with config resolvers
```

---

## TypeScript-Specific Patterns

### Streaming — `AsyncIterable` for all SSE

```ts
// ✅ Standard pattern across Anthropic, OpenAI, and Vercel SDKs
async function* streamAgent(input: AgentInput): AsyncIterable<AgentEvent> {
  for await (const chunk of client.stream(input)) {
    yield parseEvent(chunk)
  }
}

// Expose abort for user cancellation
export function useAgentStream(input: AgentInput) {
  const controller = new AbortController()
  const stream = streamAgent(input, { signal: controller.signal })
  return { stream, cancel: () => controller.abort() }
}
```

### Discriminated Unions for Event Types — Never `any`

```ts
// ✅ Type-safe stream events
type AgentEvent =
  | { type: 'token';     data: string }
  | { type: 'reasoning'; data: string }
  | { type: 'citation';  data: Citation }
  | { type: 'done';      data: AgentResult }
  | { type: 'error';     data: string }

// ❌ Untyped
const event: any = parseSSEChunk(raw)
```

### Zod as Single Source of Truth for LLM Schemas

```ts
// ✅ Define once, use for LLM structured output + runtime validation
import { z } from 'zod'

const AgentOutputSchema = z.object({
  summary: z.string(),
  citations: z.array(z.object({ url: z.string().url(), title: z.string() })),
  confidence: z.number().min(0).max(1),
})

type AgentOutput = z.infer<typeof AgentOutputSchema>  // Inferred — no duplicate types

// Validate LLM response at the boundary
const output = AgentOutputSchema.parse(llmResponse)
```

---

## Writing Agent Instructions (Prompts)

Follow the Stack Overflow guidelines for agent-readable documentation:

- **Be explicit, not tacit** — agents don't absorb conventions. Every constraint must be spelled out.
- **Show correct AND incorrect examples** — agents pattern-match; anti-patterns are as important as correct examples.
- **One concern per prompt file** — don't mix generation, validation, and formatting instructions.
- **State the output contract at the top** — begin with "Your output must be: ..."
- **Use failures as feedback** — when an agent violates a guideline, update the prompt file immediately.

```markdown
<!-- prompts/veo_json_builder.md -->
# Veo-3 JSON Builder

Your output must be: a single, valid JSON object conforming to the Veo-3 schema.

## Rules
- Output ONLY the JSON object. No explanatory text before or after.
- Every scene must have: `duration_seconds`, `visual_description`, `audio_cue`.
- Do not include narration or script text in the JSON.

## Correct output
{ "scenes": [{ "duration_seconds": 5, "visual_description": "...", "audio_cue": "..." }] }

## Incorrect output (never do this)
Here is the JSON you requested:
{ ... }
```

---

## Multi-Agent Architecture Patterns

| Pattern | Use when | Watch out for |
|---|---|---|
| **Sequential pipeline** | Data flows one-way, each step feeds the next | Single point of failure at each stage |
| **Parallel / Consortium** | Independent tasks, exploration, creative output | Requires reasoning consolidation step |
| **Supervisor** | Tight orchestration, sequential subtask delegation | Bottleneck for open-ended queries |
| **Hierarchical** | Complex planning with sub-delegation needed | High complexity — apply KISS first |

For **open-ended, exploratory** tasks (e.g., graph queries, research), prefer parallel/swarm over supervisor — supervisor architectures become bottlenecks when queries are unpredictable.

---

## Decision Checklist

Before adding LLM reasoning to any step, ask:

- [ ] Does this step require language understanding or judgment? → If no, use a pure function.
- [ ] Does this agent have more than one tool? → Split into multiple agents.
- [ ] Is the prompt hardcoded in source? → Move to `prompts/` directory.
- [ ] Is the output high-stakes? → Add a reasoning consolidation step over multiple models.
- [ ] Is workflow logic inside the route handler or MCP server? → Move to `lib/`.
- [ ] Does this abstraction add clarity or complexity? → If complexity, remove it.
