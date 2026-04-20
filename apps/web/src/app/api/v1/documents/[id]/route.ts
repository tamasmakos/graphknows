import { NextRequest } from "next/server";
import { GRAPHGEN_URL } from "@/lib/env";

/**
 * DELETE /api/v1/documents/[id]
 * POST   /api/v1/documents/[id]/reprocess (handled via search params)
 */
export async function DELETE(
    _req: NextRequest,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;
    const res = await fetch(`${GRAPHGEN_URL}/documents/${id}`, {
        method: "DELETE",
    });
    return new Response(null, { status: res.status });
}
