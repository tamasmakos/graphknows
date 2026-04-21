"use client";

import { useRef, useState } from "react";
import { DocumentsPane } from "@/components/DocumentsPane";
import { GraphVisualizer } from "@/components/GraphVisualizer";
import { Message, SSEChunk } from "@/lib/types";
import { Send, User as UserIcon, Bot, Loader2 } from "lucide-react";

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [streaming, setStreaming] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const [graphData, setGraphData] = useState<{ nodes: any[], edges: any[] } | null>(null);

    async function send() {
        const query = input.trim();
        if (!query || streaming) return;

        setInput("");
        setGraphData(null); // Clear graph for new question
        const userMsg: Message = { role: "user", content: query };
        setMessages((prev) => [...prev, userMsg]);
        setStreaming(true);

        const assistantMsg: Message = { role: "assistant", content: "", citations: [], toolCalls: [] };
        setMessages((prev) => [...prev, assistantMsg]);

        try {
            const res = await fetch("/api/v1/chat/stream", {
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
                        const chunkStr = line.slice(5).trim();
                        if (!chunkStr) continue;
                        
                        const chunk = JSON.parse(chunkStr) as SSEChunk;
                        
                        if (chunk.type === "token") {
                            setMessages((prev) => {
                                const copy = [...prev];
                                copy[copy.length - 1] = {
                                    ...copy[copy.length - 1],
                                    content: copy[copy.length - 1].content + (chunk.data as string),
                                };
                                return copy;
                            });
                        } else if (chunk.type === "citation") {
                            setMessages((prev) => {
                                const copy = [...prev];
                                const currentCitations = copy[copy.length - 1].citations || [];
                                copy[copy.length - 1] = {
                                    ...copy[copy.length - 1],
                                    citations: [...currentCitations, chunk.data],
                                };
                                return copy;
                            });
                        } else if (chunk.type === "tool_call") {
                             setMessages((prev) => {
                                const copy = [...prev];
                                const currentTools = copy[copy.length - 1].toolCalls || [];
                                copy[copy.length - 1] = {
                                    ...copy[copy.length - 1],
                                    toolCalls: [...currentTools, chunk.data],
                                };
                                return copy;
                            });
                        } else if (chunk.type === "graph") {
                            // Extract Neo4j subgraph contexts to explicitly draw the user visual focus
                            setGraphData(prev => {
                                if (!prev) return chunk.data;
                                // Basic merge of incoming Graph node data (e.g., from multiple tools)
                                const existingNodes = new Set(prev.nodes.map((n:any) => n.id));
                                const existingEdges = new Set(prev.edges.map((e:any) => `${e.source}-${e.target}-${e.type}`));
                                
                                const newNodes = chunk.data.nodes.filter((n:any) => !existingNodes.has(n.id));
                                const newEdges = chunk.data.edges.filter((e:any) => !existingEdges.has(`${e.source}-${e.target}-${e.type}`));
                                
                                return {
                                    nodes: [...prev.nodes, ...newNodes],
                                    edges: [...prev.edges, ...newEdges]
                                }
                            });
                        }
                    } catch (e) {
                        console.error("SSE parse error", e, line);
                    }
                }
                bottomRef.current?.scrollIntoView({ behavior: "smooth" });
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
        <div className="flex h-screen w-full overflow-hidden bg-white dark:bg-zinc-950 font-sans">
            {/* Left Pane: Documents & Upload */}
            <div className="w-80 flex-shrink-0">
                <DocumentsPane />
            </div>

            {/* Center Pane: Chat Interface */}
            <div className="flex-1 flex flex-col min-w-0 border-r dark:border-zinc-800">
                <div className="p-4 border-b dark:border-zinc-800 bg-gray-50/30 flex justify-between items-center">
                    <h1 className="font-semibold px-2">Agent Interface</h1>
                </div>

                <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
                    {messages.length === 0 ? (
                        <div className="flex flex-col flex-1 h-full items-center justify-center text-center max-w-sm mx-auto space-y-4 opacity-70">
                           <Bot className="w-12 h-12 text-primary/60"/>
                           <p className="text-sm font-medium">Hello. Upload a document on the left, then ask me a question.</p>
                        </div>
                    ) : (
                        messages.map((m, i) => (
                            <div key={i} className={`flex gap-4 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                {m.role === 'assistant' && (
                                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-1 text-primary">
                                        <Bot className="w-5 h-5"/>
                                    </div>
                                )}
                                <div className={`flex flex-col gap-2 max-w-[85%] ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                                    <div className={`px-4 py-3 rounded-2xl text-[15px] leading-relaxed shadow-sm ${
                                        m.role === 'user' 
                                          ? 'bg-primary text-primary-foreground rounded-tr-sm' 
                                          : 'bg-white border rounded-tl-sm text-gray-800 dark:bg-zinc-900 dark:border-zinc-700 dark:text-zinc-200'
                                    }`}>
                                        {m.content || (streaming && i === messages.length - 1 ? <Loader2 className="w-4 h-4 animate-spin my-1"/> : "")}
                                    </div>
                                    
                                    {m.role === "assistant" && m.toolCalls && m.toolCalls.length > 0 && (
                                        <div className="text-xs text-gray-500 font-mono bg-gray-50 border rounded p-2 max-w-full overflow-x-auto dark:bg-zinc-900 dark:border-zinc-800">
                                            {m.toolCalls.map((tc, idx) => (
                                                <div key={idx} className="flex gap-2 items-center">
                                                    <span className="text-blue-500 hover:underline cursor-pointer">{tc.tool_name}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {m.role === "assistant" && m.citations && m.citations.length > 0 && (
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {m.citations.map((cit, idx) => (
                                                <span key={idx} className="text-[10px] bg-slate-100 border text-slate-600 px-2 py-0.5 rounded cursor-pointer hover:bg-slate-200 dark:bg-zinc-800 dark:border-zinc-700 dark:text-zinc-400">
                                                    [{idx + 1}] {cit.doc_id ? cit.doc_id.substring(0, 8) : "Source"}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    <div ref={bottomRef} className="h-px bg-transparent"></div>
                </div>

                <div className="p-4 bg-white dark:bg-zinc-950">
                    <div className="relative max-w-3xl mx-auto flex items-center shadow-sm rounded-xl border dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-hidden focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all">
                        <input 
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
                            placeholder="Ask a question..."
                            className="flex-1 bg-transparent py-4 pl-5 outline-none text-[15px] text-gray-800 dark:text-zinc-100 placeholder:text-gray-400"
                        />
                        <button 
                            onClick={send}
                            disabled={!input.trim() || streaming}
                            className="bg-primary/10 hover:bg-primary text-primary hover:text-white p-2.5 mr-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Right Pane: Graph Explorer (Force Graph) */}
            <div className="w-[33%] hidden lg:block bg-gray-50 flex-shrink-0 relative">
                 <GraphVisualizer graphData={graphData} isLoading={streaming && graphData === null} />
            </div>
        </div>
    );
}
