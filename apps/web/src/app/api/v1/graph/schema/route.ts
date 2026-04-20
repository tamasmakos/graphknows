import { GRAPHRAG_URL } from "@/lib/env";

/**
 * GET /api/v1/graph/schema
 */
export async function GET(req: Request) {
    const { searchParams } = new URL(req.url);
    const db = searchParams.get("database") ?? "neo4j";

    const res = await fetch(`${GRAPHRAG_URL}/schema?database=${db}`);
    const data = await res.json();
    return Response.json(data, { status: res.status });
}
