import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description: "Run read-only Cypher queries against the Neo4j database",
  args: {
    query: tool.schema.string().describe("Read-only Cypher query to execute"),
    database: tool.schema.string().default("neo4j").describe("Neo4j database name"),
    max_rows: tool.schema.number().int().min(1).max(200).default(25).describe("Maximum rows to return"),
  },
  async execute(args, context) {
    const script = path.join(context.worktree, ".opencode/tools/internal/db_debug.py")
    try {
      const output = await Bun.$`python3 ${script} neo4j-query ${args.query} ${args.database} ${String(args.max_rows)}`.text()
      return output.trim()
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      return JSON.stringify({ error: "neo4j-query failed", detail }, null, 2)
    }
  },
})
