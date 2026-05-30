"use client";

import { useRef, useCallback, useEffect } from "react";

export function useAgentWebSocket(
    sessionId: string,
    onMessage?: (msg: Record<string, unknown>) => void,
) {
    const onMessageRef = useRef(onMessage);
    onMessageRef.current = onMessage;

    const wsRef = useRef<WebSocket | null>(null);

    const connect = useCallback((query: string) => {
        const baseUrl = process.env.NEXT_PUBLIC_AGENT_WS_URL ?? "ws://localhost:8002/api/v1";
        const ws = new WebSocket(`${baseUrl}/ws/chat/${sessionId}`);
        wsRef.current = ws;

        ws.onopen = () => {
            ws.send(JSON.stringify({ query }));
        };

        ws.onmessage = (evt: MessageEvent<string>) => {
            try {
                const msg = JSON.parse(evt.data) as Record<string, unknown>;
                onMessageRef.current?.(msg);
            } catch {
                console.warn("Failed to parse WS message", evt.data);
            }
        };

        ws.onerror = () => {
            console.warn("Agent WS error (non-fatal)");
        };

        ws.onclose = () => {
            console.log("Agent WS closed");
        };
    }, [sessionId]);

    const disconnect = useCallback(() => {
        wsRef.current?.close();
        wsRef.current = null;
    }, []);

    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return { connect, disconnect };
}
