import { NextRequest } from "next/server";
import { GRAPHRAG_URL } from "@/lib/env";

/**
 * POST /api/v1/chat
 * Proxies to graphrag /chat and streams the SSE response back to the browser.
 */
export async function POST(req: NextRequest) {
    const body = await req.json();

    const upstream = await fetch(`${GRAPHRAG_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!upstream.ok) {
        return new Response(await upstream.text(), { status: upstream.status });
    }

    // Pass through the SSE stream
    return new Response(upstream.body, {
        headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
        },
    });
}
