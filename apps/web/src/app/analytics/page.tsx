"use client";

import { useEffect, useState } from "react";

interface Stats {
    documents: number;
    chunks: number;
    entities: number;
    relationships: number;
}

export default function AnalyticsPage() {
    const [stats, setStats] = useState<Stats | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/v1/graph/schema")
            .then((r) => r.json())
            .then(() => {
                // Fetch actual counts from graphgen
                return fetch("/api/v1/analytics");
            })
            .then((r) => r.json())
            .then(setStats)
            .catch(() => setStats(null))
            .finally(() => setLoading(false));
    }, []);

    const cards = stats
        ? [
            { label: "Documents", value: stats.documents },
            { label: "Chunks", value: stats.chunks },
            { label: "Entities", value: stats.entities },
            { label: "Relationships", value: stats.relationships },
        ]
        : [];

    return (
        <div className="p-6">
            <h1 className="text-xl font-semibold mb-6">Analytics</h1>
            {loading ? (
                <p style={{ color: "var(--text-muted)" }}>Loading…</p>
            ) : stats === null ? (
                <p style={{ color: "var(--text-muted)" }}>Stats unavailable.</p>
            ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {cards.map((c) => (
                        <div
                            key={c.label}
                            className="rounded-xl p-5"
                            style={{ backgroundColor: "var(--surface)", border: "1px solid var(--border)" }}
                        >
                            <p className="text-3xl font-bold" style={{ color: "var(--accent)" }}>
                                {c.value.toLocaleString()}
                            </p>
                            <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                                {c.label}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
