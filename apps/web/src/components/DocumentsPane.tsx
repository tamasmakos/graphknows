"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, FileBox, RefreshCw, Trash2, CheckCircle2, Clock, AlertCircle, Terminal, Cpu } from "lucide-react";

interface UploadedDoc {
    id: string;
    title: string;
    status: "pending" | "processing" | "complete" | "error";
    chunk_count: number;
    entity_count: number;
}

interface LogEntry {
    id: string;
    timestamp: Date;
    message: string;
    type: "info" | "success" | "error";
}

export function DocumentsPane() {
    const [documents, setDocuments] = useState<UploadedDoc[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [isPipelining, setIsPipelining] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const logsEndRef = useRef<HTMLDivElement>(null);

    const addLog = useCallback((message: string, type: "info" | "success" | "error" = "info") => {
        setLogs(prev => [...prev, { id: Math.random().toString(36).slice(2), timestamp: new Date(), message, type }]);
    }, []);

    // Auto-scroll logs
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs]);

    const fetchDocuments = useCallback(async (silent = true) => {
        try {
            if (!silent) addLog("Fetching latest documents...", "info");
            const res = await fetch("/api/v1/documents");
            if (res.ok) {
                const data = await res.json();
                // The API returns { documents: [...] }
                const docs = data.documents || data || [];
                setDocuments(docs);
                if (!silent) addLog(`Found ${docs.length} document(s).`, "success");
            } else {
                if (!silent) addLog(`Failed to fetch documents: ${res.statusText}`, "error");
            }
        } catch (e) {
            if (!silent) addLog("Error fetching documents connection failed.", "error");
            console.error("Failed to fetch docs", e);
        }
    }, [addLog]);

    useEffect(() => {
        fetchDocuments(false);
        const interval = setInterval(() => fetchDocuments(true), 5000);
        return () => clearInterval(interval);
    }, [fetchDocuments]);

    const onDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };
    const onDragLeave = () => setIsDragging(false);

    const handleFiles = async (files: FileList | null) => {
        if (!files || files.length === 0) return;

        setIsUploading(true);
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append("file", file);

            addLog(`Started uploading "${file.name}"...`, "info");

            try {
                const res = await fetch("/api/v1/documents", {
                    method: "POST",
                    body: formData,
                });

                if (res.ok) {
                    const data = await res.json();
                    addLog(`Successfully uploaded "${file.name}". Created ${data.chunks || 0} chunks.`, "success");
                } else {
                    const errText = await res.text();
                    addLog(`Failed to upload "${file.name}": ${errText}`, "error");
                }
            } catch (e) {
                addLog(`Error uploading "${file.name}".`, "error");
                console.error("Upload failed", e);
            }
        }
        setIsUploading(false);
        fetchDocuments(false);
    };

    const runPipeline = async () => {
        setIsPipelining(true);
        addLog("Starting KG extraction pipeline...", "info");
        try {
            const res = await fetch("/api/v1/pipeline/run", { method: "POST" });
            const data = await res.json();
            if (res.ok) {
                addLog(`Pipeline started: ${data.message || "running in background."}`, "success");
            } else if (res.status === 409) {
                addLog("Pipeline is already running.", "info");
            } else {
                addLog(`Pipeline error: ${data.detail || data.error || res.statusText}`, "error");
            }
        } catch {
            addLog("Failed to contact pipeline service.", "error");
        } finally {
            setIsPipelining(false);
        }
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
    };

    function getStatusIcon(status: string) {
        if (status === "complete") return <CheckCircle2 className="w-4 h-4 text-green-500" />;
        if (status === "processing" || status === "pending") return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
        if (status === "error") return <AlertCircle className="w-4 h-4 text-red-500" />;
        return <Clock className="w-4 h-4 text-gray-500" />;
    }

    return (
        <div className="flex flex-col h-full bg-gray-50/50 border-r dark:bg-zinc-920 dark:border-zinc-800">
            <div className="p-4 border-b dark:border-zinc-800 bg-white dark:bg-zinc-950">
                <h2 className="font-semibold flex items-center gap-2">
                    <FileBox className="w-5 h-5 text-primary" />
                    Knowledge Base
                </h2>
            </div>
            <div className="px-4 py-2 border-b dark:border-zinc-800 bg-white dark:bg-zinc-950">
                <button
                    onClick={runPipeline}
                    disabled={isPipelining || documents.length === 0}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                    <Cpu className={`w-4 h-4 ${isPipelining ? 'animate-pulse' : ''}`} />
                    {isPipelining ? 'Building Graph...' : 'Build Graph'}
                </button>
            </div>

            <div className="p-4 space-y-4 flex-1 overflow-auto">
                <div
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                    className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-sm transition-all duration-200 cursor-pointer
            ${isDragging ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-gray-200 dark:border-zinc-700 hover:border-primary/50 hover:bg-gray-100 dark:hover:bg-zinc-800/50'}
          `}
                    onClick={() => document.getElementById('file-upload')?.click()}
                >
                    <Upload className={`w-8 h-8 mb-3 transition-colors ${isDragging ? 'text-primary' : 'text-gray-400'}`} />
                    <p className="font-medium">Drop documents here</p>
                    <p className="text-gray-500 text-xs text-center mt-1 mb-2">
                        PDF, DOCX, PPTX, CSV, TXT, HTML
                    </p>
                    <input
                        type="file"
                        multiple
                        className="hidden"
                        id="file-upload"
                        onChange={(e) => handleFiles(e.target.files)}
                    />
                    <span className="mt-2 px-4 py-1.5 bg-white dark:bg-zinc-900 border shadow-sm rounded-md text-xs font-semibold text-gray-700 dark:text-zinc-300">
                        {isUploading ? 'Uploading...' : 'Browse files'}
                    </span>
                </div>

                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3 px-1">Processed Files</h3>
                    {documents.length === 0 ? (
                        <p className="text-sm text-gray-500 text-center py-6 bg-white dark:bg-zinc-900 rounded-xl border border-dashed shadow-sm">No documents yet</p>
                    ) : (
                        documents.map((doc: any) => (
                            <div key={doc.doc_id || doc.id} className="bg-white dark:bg-zinc-950 p-3.5 rounded-xl border shadow-sm flex items-start gap-3 hover:border-primary/30 transition-colors">
                                <div className="mt-0.5">
                                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <p className="font-medium text-sm truncate text-gray-800 dark:text-zinc-200" title={doc.title || doc.source_path}>{doc.title || doc.source_path}</p>
                                    <p className="text-xs text-gray-500 flex gap-2 mt-1.5 items-center">
                                        <span className="bg-gray-100 dark:bg-zinc-800 px-2 py-0.5 rounded-md">{doc.chunk_count || 0} chunks</span>
                                        <span>&bull;</span>
                                        <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                                    </p>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Logs Terminal Pane */}
            <div className="h-48 border-t dark:border-zinc-800 bg-[#0A0A0A] flex flex-col text-xs font-mono">
                <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800/50 bg-[#111] text-zinc-400">
                    <Terminal className="w-3.5 h-3.5" />
                    <span className="font-semibold tracking-wider text-[10px] uppercase">Activity Log</span>
                </div>
                <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
                    {logs.length === 0 ? (
                        <span className="text-zinc-600 italic">Waiting for activity...</span>
                    ) : (
                        logs.map(log => (
                            <div key={log.id} className="flex gap-3 leading-relaxed">
                                <span className="text-zinc-500 shrink-0">
                                    {log.timestamp.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </span>
                                <span className={`
                  ${log.type === 'error' ? 'text-red-400' : ''}
                  ${log.type === 'success' ? 'text-emerald-400' : ''}
                  ${log.type === 'info' ? 'text-zinc-300' : ''}
                `}>
                                    {log.message}
                                </span>
                            </div>
                        ))
                    )}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>
    );
}
