import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Search GraphKnows memory (STM, LTM, or both) using hybrid RRF retrieval",
  args: {
    query: tool.schema.string().describe("Natural language query to recall against"),
    session_id: tool.schema.string().describe("Session namespace to scope the search"),
    scope: tool.schema.string().default("stm").describe("stm, ltm, or both"),
    top_k: tool.schema.number().int().min(1).max(20).default(5).describe("Max hits to return"),
  },
  async execute(args, _context) {
    const mcpBinary = "/root/graphknows/.venv/bin/graphknows-mcp"

    const initRpc = JSON.stringify({
      jsonrpc: "2.0", id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "opencode", version: "1.0" } },
    })
    const callRpc = JSON.stringify({
      jsonrpc: "2.0", id: 2,
      method: "tools/call",
      params: { name: "memory_recall", arguments: { query: args.query, session_id: args.session_id, scope: args.scope, top_k: args.top_k } },
    })

    const input = initRpc + "\n" + callRpc + "\n"

    try {
      const result = await Bun.$`printf ${input} | ${mcpBinary} 2>/dev/null | tail -1`.text()
      const parsed = JSON.parse(result.trim())

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
