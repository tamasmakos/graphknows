import { NextRequest } from "next/server";
import { GRAPHGEN_URL } from "@/lib/env";

/**
 * GET  /api/v1/documents  — list all documents
 * POST /api/v1/documents  — upload a file and trigger ingestion
 */
const TIMEOUT_MS = 5000;

export async function GET() {
    try {
        const res = await fetch(`${GRAPHGEN_URL}/documents`, {
            signal: AbortSignal.timeout(TIMEOUT_MS),
        });
        const data = await res.json();
        return Response.json(data, { status: res.status });
    } catch {
        return Response.json(
            { documents: [], error: "graphgen service unreachable" },
            { status: 503 }
        );
    }
}

export async function POST(req: NextRequest) {
    const formData = await req.formData();
    try {
        const res = await fetch(`${GRAPHGEN_URL}/documents`, {
            method: "POST",
            body: formData,
            signal: AbortSignal.timeout(60_000), // uploads may take longer
        });
        const text = await res.text();
        let data: unknown;
        try { data = JSON.parse(text); } catch { data = { detail: text }; }
        return Response.json(data, { status: res.status });
    } catch {
        return Response.json(
            { detail: "graphgen service unreachable" },
            { status: 503 }
        );
    }
}
