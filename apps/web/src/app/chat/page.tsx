"use client";

import { useRef, useState } from "react";
import type { Citation, SSEChunk } from "@graphknows/types";

interface Message {
    role: "user" | "assistant";
    content: string;
    citations?: Citation[];
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [streaming, setStreaming] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    async function send() {
        const query = input.trim();
        if (!query || streaming) return;

        setInput("");
        const userMsg: Message = { role: "user", content: query };
        setMessages((prev) => [...prev, userMsg]);
        setStreaming(true);

        const assistantMsg: Message = { role: "assistant", content: "", citations: [] };
        setMessages((prev) => [...prev, assistantMsg]);

        try {
            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    messages: messages.map((m) => ({ role: m.role, content: m.content })),
                }),
            });

            if (!res.body) throw new Error("No response body");
            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const raw = decoder.decode(value);
                const lines = raw.split("\n");

                for (const line of lines) {
                    if (!line.startsWith("data:")) continue;
                    try {
                        const chunk = JSON.parse(line.slice(5).trim()) as SSEChunk;
                        if (chunk.type === "delta") {
                            setMessages((prev) => {
                                const copy = [...prev];
                                copy[copy.length - 1] = {
                                    ...copy[copy.length - 1],
                                    content: copy[copy.length - 1].content + (chunk.data as string),
                                };
                                return copy;
                            });
                        } else if (chunk.type === "citations") {
                            setMessages((prev) => {
                                const copy = [...prev];
                                copy[copy.length - 1] = {
                                    ...copy[copy.length - 1],
                                    citations: chunk.data as Citation[],
                                };
                                return copy;
                            });
                        }
                    } catch {
                        // partial JSON — skip
                    }
                }
            }
        } catch (err) {
            setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = {
                    ...copy[copy.length - 1],
                    content: `Error: ${err instanceof Error ? err.message : String(err)}`,
                };
                return copy;
            });
        } finally {
            setStreaming(false);
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }

    return (
        <div className="flex flex-col h-full">
            {/* Message list */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && (
                    <p className="text-center mt-20" style={{ color: "var(--text-muted)" }}>
                        Ask anything about your documents…
                    </p>
                )}
                {messages.map((msg, i) => (
                    <div key={i} className={msg.role === "user" ? "flex justify-end" : "flex justify-start"}>
                        <div
                            className="max-w-2xl rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap"
                            style={{
                                backgroundColor: msg.role === "user" ? "var(--accent)" : "var(--surface)",
                                color: msg.role === "user" ? "#fff" : "var(--text)",
                                border: msg.role === "assistant" ? "1px solid var(--border)" : "none",
                            }}
                        >
                            {msg.content}
                            {msg.citations && msg.citations.length > 0 && (
                                <details className="mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
                                    <summary className="cursor-pointer">
                                        {msg.citations.length} source{msg.citations.length !== 1 ? "s" : ""}
                                    </summary>
                                    <ol className="mt-2 space-y-1 list-decimal list-inside">
                                        {msg.citations.map((c, ci) => (
                                            <li key={ci} title={c.text_excerpt}>
                                                <strong>{c.doc_title}</strong>
                                                {c.heading_path?.length > 0 && ` › ${c.heading_path.join(" › ")}`}
                                                {` (score: ${c.score.toFixed(2)})`}
                                            </li>
                                        ))}
                                    </ol>
                                </details>
                            )}
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>

            {/* Input row */}
            <div
                className="border-t p-4 flex gap-2"
                style={{ borderColor: "var(--border)", backgroundColor: "var(--surface)" }}
            >
                <input
                    className="flex-1 rounded-lg border px-4 py-2 text-sm outline-none focus:ring-2"
                    style={{
                        backgroundColor: "var(--bg)",
                        borderColor: "var(--border)",
                        color: "var(--text)",
                    }}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
                    placeholder="Ask a question…"
                    disabled={streaming}
                />
                <button
                    onClick={send}
                    disabled={streaming || !input.trim()}
                    className="rounded-lg px-5 py-2 text-sm font-medium transition-colors disabled:opacity-40"
                    style={{ backgroundColor: "var(--accent)", color: "#fff" }}
                >
                    {streaming ? "…" : "Send"}
                </button>
            </div>
        </div>
    );
}
