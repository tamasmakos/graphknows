import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description: "Run read-only SQL queries against the pgvector PostgreSQL database",
  args: {
    query: tool.schema.string().describe("Read-only SQL query to execute"),
    max_rows: tool.schema.number().int().min(1).max(500).default(25).describe("Maximum rows to return"),
  },
  async execute(args, context) {
    const script = path.join(context.worktree, ".opencode/tools/internal/db_debug.py")
    try {
      const output = await Bun.$`python3 ${script} pgvector-query ${args.query} ${String(args.max_rows)}`.text()
      return output.trim()
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      return JSON.stringify({ error: "pgvector-query failed", detail }, null, 2)
    }
  },
})
