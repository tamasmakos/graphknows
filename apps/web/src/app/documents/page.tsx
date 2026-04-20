"use client";

import { useEffect, useState } from "react";

interface Document {
    doc_id: string;
    title: string;
    source_path: string;
    created_at: string;
    chunk_count: number;
}

export default function DocumentsPage() {
    const [docs, setDocs] = useState<Document[]>([]);
    const [loading, setLoading] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function loadDocs() {
        try {
            const res = await fetch("/api/v1/documents");
            const data = await res.json();
            setDocs(data.documents ?? []);
        } catch {
            setError("Failed to load documents");
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        void loadDocs();
    }, []);

    async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
        const file = e.target.files?.[0];
        if (!file) return;
        setUploading(true);
        try {
            const form = new FormData();
            form.append("file", file);
            const res = await fetch("/api/v1/documents", { method: "POST", body: form });
            if (!res.ok) throw new Error(await res.text());
            await loadDocs();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setUploading(false);
            e.target.value = "";
        }
    }

    async function deleteDoc(doc_id: string) {
        await fetch(`/api/v1/documents/${doc_id}`, { method: "DELETE" });
        await loadDocs();
    }

    return (
        <div className="p-6 max-w-4xl">
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-xl font-semibold">Documents</h1>
                <label
                    className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium"
                    style={{ backgroundColor: "var(--accent)", color: "#fff", opacity: uploading ? 0.5 : 1 }}
                >
                    {uploading ? "Uploading…" : "Upload document"}
                    <input
                        type="file"
                        className="hidden"
                        accept=".txt,.md,.pdf,.docx,.pptx,.xlsx,.html,.png,.jpg"
                        onChange={handleUpload}
                        disabled={uploading}
                    />
                </label>
            </div>

            {error && (
                <div
                    className="rounded-lg px-4 py-3 mb-4 text-sm"
                    style={{ backgroundColor: "#3f1818", color: "#fca5a5" }}
                >
                    {error}
                    <button className="ml-3 underline" onClick={() => setError(null)}>
                        dismiss
                    </button>
                </div>
            )}

            {loading ? (
                <p style={{ color: "var(--text-muted)" }}>Loading…</p>
            ) : docs.length === 0 ? (
                <p style={{ color: "var(--text-muted)" }}>No documents yet. Upload one to get started.</p>
            ) : (
                <table className="w-full text-sm">
                    <thead>
                        <tr style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
                            <th className="text-left py-2 pr-4">Title</th>
                            <th className="text-left py-2 pr-4">Chunks</th>
                            <th className="text-left py-2 pr-4">Ingested</th>
                            <th />
                        </tr>
                    </thead>
                    <tbody>
                        {docs.map((doc) => (
                            <tr
                                key={doc.doc_id}
                                style={{ borderBottom: "1px solid var(--border)" }}
                                className="hover:bg-white/5"
                            >
                                <td className="py-2 pr-4 font-medium">{doc.title || doc.source_path}</td>
                                <td className="py-2 pr-4" style={{ color: "var(--text-muted)" }}>
                                    {doc.chunk_count}
                                </td>
                                <td className="py-2 pr-4" style={{ color: "var(--text-muted)" }}>
                                    {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : "—"}
                                </td>
                                <td className="py-2 text-right">
                                    <button
                                        onClick={() => void deleteDoc(doc.doc_id)}
                                        className="text-xs underline hover:no-underline"
                                        style={{ color: "var(--text-muted)" }}
                                    >
                                        delete
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}
