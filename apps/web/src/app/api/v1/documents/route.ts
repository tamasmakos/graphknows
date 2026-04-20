import { NextRequest } from "next/server";
import { GRAPHGEN_URL } from "@/lib/env";

/**
 * GET  /api/v1/documents  — list all documents
 * POST /api/v1/documents  — upload a file and trigger ingestion
 */
export async function GET() {
    const res = await fetch(`${GRAPHGEN_URL}/documents`);
    const data = await res.json();
    return Response.json(data, { status: res.status });
}

export async function POST(req: NextRequest) {
    const formData = await req.formData();
    const res = await fetch(`${GRAPHGEN_URL}/documents`, {
        method: "POST",
        body: formData,
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
}
