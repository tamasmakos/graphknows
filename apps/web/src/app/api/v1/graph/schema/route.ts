import { GRAPHRAG_URL } from "@/lib/env";

/**
 * GET /api/v1/graph/schema
 */
export async function GET(req: Request) {
    const { searchParams } = new URL(req.url);
    const db = searchParams.get("database") ?? "neo4j";

    try {
        const res = await fetch(`${GRAPHRAG_URL}/schema?database=${db}`, {
            signal: AbortSignal.timeout(5000),
        });
        const data = await res.json();
        return Response.json(data, { status: res.status });
    } catch {
        return Response.json(
            { database: db, node_labels: [], relationship_types: [], property_keys: [] },
            { status: 503 }
        );
    }
}
