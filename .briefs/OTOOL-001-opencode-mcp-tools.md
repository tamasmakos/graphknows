# OTOOL-001: OpenCode MCP Tool Wrappers for GraphKnows Memory

## First actions (mandatory)
At the top of your final reply, explicitly state:
1. current working directory
2. what tools/capabilities you actually have in this session
3. whether you can read repo files and run local python AND bun/typescript

If cwd is not `/root/graphknows`, stop and fix it first.

## Hard scope
Create only new files under `.opencode/tools/`. Do NOT modify existing `.opencode/tools/` files.
Do NOT modify any production code under `graphknows/`.
Do NOT commit, push, or open a PR.

## Project rules
- Repo root: `/root/graphknows`
- OpenCode tools live at `.opencode/tools/` — follow existing TypeScript pattern
- Existing tools: `memgraph-query.ts`, `neo4j-query.ts`, `pgvector-query.ts`, `neo4j-status.ts` — study these for the pattern
- The bridge script: `.opencode/tools/internal/db_debug.py` — study dispatch pattern

## Context — what we're doing

GraphKnows has an MCP server running at `/root/graphknows/.venv/bin/graphknows-mcp` (PID 158299). It exposes memory tools via stdin/stdout JSON-RPC 2.0:
- `memory_recall` — hybrid RRF search (STM/LTM/both)
- `memory_ingest` — ingest text into STM
- `memory_flush` — promote STM session to LTM
- `memory_doctor` — backend connectivity check
- `memory_stats` — STM/LTM counts
- Plus 6 LTM tools (ltm_query, ltm_entity, ltm_topics, recall_by_document, recall_by_dialog, recall_by_topic)

OpenCode doesn't speak MCP natively. We create thin TypeScript tool wrappers that shell out to the MCP server via stdin JSON-RPC probe — one-shot call, parse response, return result.

This is a DIFFERENT pattern from the existing DB tools. The DB tools call `db_debug.py` directly. These new tools call the MCP server process via stdin pipe.

## Read before coding

1. `.opencode/tools/memgraph-query.ts` — study the TypeScript tool pattern (import, description, args schema, execute)
2. `.opencode/tools/internal/db_debug.py` — understand the bridge dispatch pattern (we'll use a different approach)

## MCP JSON-RPC stdin probe pattern

The MCP protocol uses JSON-RPC 2.0 over stdin/stdout. To call a tool:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"memory_recall","arguments":{"query":"test query","session_id":"test","top_k":3}}}' | /root/graphknows/.venv/bin/graphknows-mcp 2>/dev/null
```

The response is a JSON-RPC response with the tool result in `result.content[0].text`.

Important: the MCP server requires an `initialize` handshake before `tools/call`. A complete probe:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"opencode","version":"1.0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_recall","arguments":{"query":"test query","session_id":"test","top_k":3}}}' \
  | /root/graphknows/.venv/bin/graphknows-mcp 2>/dev/null | tail -1
```

The `tail -1` gets the last JSON line (the tools/call response).

## What to build — 4 TypeScript tools

### Tool 1: `graphknows-recall.ts`

```
Description: Search GraphKnows memory (STM, LTM, or both) using hybrid RRF retrieval.
Args:
  - query: string (required) — natural language query
  - session_id: string (required) — session namespace
  - scope: enum["stm","ltm","both"] (default: "stm")
  - top_k: number (default: 5, min:1, max:20)
Returns: formatted memory hits with text, score, entities, provenance
```

### Tool 2: `graphknows-ingest.ts`

```
Description: Ingest text into GraphKnows short-term memory. GLiNER2 extracts entities, text is chunked and embedded.
Args:
  - text: string (required) — content to ingest
  - session_id: string (required) — session namespace
  - title: string (optional) — document title for provenance
Returns: ingest receipt with file_id, chunks, entities, elapsed_s
```

### Tool 3: `graphknows-entity.ts`

```
Description: Look up a named entity in the GraphKnows LTM knowledge graph.
Args:
  - name: string (required) — entity name (case-sensitive)
Returns: entity properties including community, pagerank, betweenness, segment count
```

### Tool 4: `graphknows-flush.ts`

```
Description: Promote a session from short-term to long-term memory (entity merging, optional topic summarization).
Args:
  - session_id: string (required) — session to promote
Returns: flush receipt with nodes, edges, chunks, turns, elapsed_s
```

## Implementation pattern

Each `.ts` file follows this exact pattern:

```typescript
import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Search GraphKnows memory using hybrid RRF retrieval...",
  args: {
    query: tool.schema.string().describe("Natural language query to recall against"),
    session_id: tool.schema.string().describe("Session namespace to scope the search"),
    scope: tool.schema.string().default("stm").describe("stm, ltm, or both"),
    top_k: tool.schema.number().int().min(1).max(20).default(5).describe("Max hits to return"),
  },
  async execute(args, context) {
    const mcpBinary = "/root/graphknows/.venv/bin/graphknows-mcp"
    
    // Build JSON-RPC initialize + tools/call
    const initRpc = JSON.stringify({
      jsonrpc: "2.0", id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "opencode", version: "1.0" } }
    })
    const callRpc = JSON.stringify({
      jsonrpc: "2.0", id: 2,
      method: "tools/call",
      params: { name: "memory_recall", arguments: { query: args.query, session_id: args.session_id, scope: args.scope, top_k: args.top_k } }
    })
    
    const input = initRpc + "\n" + callRpc + "\n"
    
    try {
      // Pipe to MCP server, capture last line (tools/call response)
      const result = await Bun.$`printf ${input} | ${mcpBinary} 2>/dev/null | tail -1`.text()
      const parsed = JSON.parse(result.trim())
      
      // Extract text content from MCP response
      if (parsed.result && parsed.result.content && parsed.result.content[0]) {
        return parsed.result.content[0].text
      }
      return JSON.stringify(parsed, null, 2)
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      return JSON.stringify({ error: "graphknows-recall failed", detail }, null, 2)
    }
  },
})
```

## Tool-to-MCP mapping

| TypeScript tool | MCP tool name | MCP arguments |
|----------------|---------------|---------------|
| graphknows-recall | memory_recall | query, session_id, top_k, scope |
| graphknows-ingest | memory_ingest | text, session_id, title |
| graphknows-entity | ltm_entity | name |
| graphknows-flush | memory_flush | session_id |

## Acceptance criteria

- [ ] 4 new `.ts` files created in `.opencode/tools/`
- [ ] Each follows the existing TypeScript tool pattern (import, description, args schema, execute)
- [ ] Each uses stdin JSON-RPC probe to `graphknows-mcp` binary
- [ ] No existing files modified
- [ ] No production code under `graphknows/` modified
- [ ] Error handling: catch failures, return structured error JSON

## Notes

- The `Bun.$` template literal with printf approach may need escaping. If printf is problematic, use `echo` with newlines or write input to a temp file.
- The MCP server must be running. If it's not, tools will return error JSON (this is acceptable — the agent can detect it).
- Do NOT hardcode any API keys or credentials.
- Keep tool descriptions concise — these go into the OpenCode agent's context window.

## Git — do NOT commit

Work in the working tree. Do not commit. The principal engineer will review and integrate.