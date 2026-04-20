"use client";

import { useEffect, useState } from "react";
import type { SchemaResponse } from "@graphknows/types";

export default function GraphPage() {
    const [schema, setSchema] = useState<SchemaResponse | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/v1/graph/schema")
            .then((r) => r.json())
            .then(setSchema)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="p-6 max-w-3xl">
            <h1 className="text-xl font-semibold mb-6">Graph Schema</h1>
            {loading ? (
                <p style={{ color: "var(--text-muted)" }}>Loading…</p>
            ) : !schema ? (
                <p style={{ color: "var(--text-muted)" }}>Failed to load schema.</p>
            ) : (
                <div className="space-y-6 text-sm">
                    <Section title="Node Labels" items={schema.node_labels} color="#6366f1" />
                    <Section title="Relationship Types" items={schema.relationship_types} color="#22d3ee" />
                    <Section title="Property Keys" items={schema.property_keys} color="#a3e635" />
                </div>
            )}
        </div>
    );
}

function Section({
    title,
    items,
    color,
}: {
    title: string;
    items: string[];
    color: string;
}) {
    return (
        <div>
            <h2 className="font-medium mb-2" style={{ color }}>
                {title}
            </h2>
            <div className="flex flex-wrap gap-2">
                {items.map((item) => (
                    <span
                        key={item}
                        className="rounded-full px-3 py-1 text-xs"
                        style={{ backgroundColor: "var(--surface)", border: "1px solid var(--border)" }}
                    >
                        {item}
                    </span>
                ))}
            </div>
        </div>
    );
}
