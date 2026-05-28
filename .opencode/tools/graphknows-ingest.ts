import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Ingest text into GraphKnows short-term memory. GLiNER2 extracts entities, text is chunked and embedded",
  args: {
    text: tool.schema.string().describe("Content to ingest"),
    session_id: tool.schema.string().describe("Session namespace"),
    title: tool.schema.string().optional().describe("Document title for provenance"),
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
      params: { name: "memory_ingest", arguments: { text: args.text, session_id: args.session_id, title: args.title || "" } },
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
      return JSON.stringify({ error: "graphknows-ingest failed", detail }, null, 2)
    }
  },
})
