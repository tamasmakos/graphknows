import { GRAPHGEN_URL } from "@/lib/env";

/**
 * GET /api/v1/analytics
 * Returns aggregate node counts from graphgen.
 */
export async function GET() {
    try {
        const res = await fetch(`${GRAPHGEN_URL}/analytics`);
        if (!res.ok) throw new Error("graphgen error");
        const data = await res.json();
        return Response.json(data);
    } catch {
        // Graceful degradation
        return Response.json(
            { documents: 0, chunks: 0, entities: 0, relationships: 0 },
            { status: 200 }
        );
    }
}
