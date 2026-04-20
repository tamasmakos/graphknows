import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "GraphKnows",
    description: "Knowledge-graph RAG platform",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body>
                <div className="flex h-screen overflow-hidden">
                    <nav
                        className="flex flex-col w-52 shrink-0 border-r p-4 gap-2"
                        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface)" }}
                    >
                        <span className="text-lg font-bold mb-4" style={{ color: "var(--accent)" }}>
                            GraphKnows
                        </span>
                        {[
                            { href: "/chat", label: "Chat" },
                            { href: "/documents", label: "Documents" },
                            { href: "/graph", label: "Graph" },
                            { href: "/analytics", label: "Analytics" },
                        ].map(({ href, label }) => (
                            <a
                                key={href}
                                href={href}
                                className="rounded px-3 py-2 transition-colors hover:bg-white/5"
                                style={{ color: "var(--text-muted)" }}
                            >
                                {label}
                            </a>
                        ))}
                    </nav>
                    <main className="flex-1 overflow-auto">{children}</main>
                </div>
            </body>
        </html>
    );
}
