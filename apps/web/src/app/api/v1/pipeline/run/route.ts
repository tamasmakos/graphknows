import { GRAPHGEN_URL } from "@/lib/env";

/**
 * POST /api/v1/pipeline/run
 * Triggers the KG extraction pipeline in graphgen.
 */
export async function POST() {
    try {
        const res = await fetch(`${GRAPHGEN_URL}/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clean_database: false, skip_communities: false }),
            signal: AbortSignal.timeout(10_000),
        });
        const data = await res.json();
        return Response.json(data, { status: res.status });
    } catch {
        return Response.json(
            { error: "Pipeline service unavailable" },
            { status: 503 }
        );
    }
}
