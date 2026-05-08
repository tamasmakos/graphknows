import { tool } from "@opencode-ai/plugin"

declare const Bun: any

export default tool({
  description: "Check Neo4j connection status by reading environment variables",
  args: {},
  async execute(_args, context) {
    const script = `${context.worktree}/.opencode/tools/internal/db_debug.py`
    try {
      const result = await Bun.$`python3 ${script} neo4j-status`.text()
      return result.trim()
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      return JSON.stringify({ error: "neo4j-status failed", detail }, null, 2)
    }
  },
})
