import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
    title: "GraphKnows",
    description: "Knowledge-graph RAG platform",
};

const NAV_ITEMS = [
    { href: "/chat", label: "Chat", icon: "💬" },
    { href: "/documents", label: "Documents", icon: "📄" },
    { href: "/analytics", label: "Analytics", icon: "📊" },
    { href: "/graph", label: "Graph", icon: "🕸️" },
];

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body>
                <div className="flex h-screen w-full overflow-hidden">
                    {/* Sidebar navigation */}
                    <nav className="flex flex-col w-14 flex-shrink-0 border-r" style={{ backgroundColor: "var(--surface)", borderColor: "var(--border)" }}>
                        <div className="flex items-center justify-center h-14 border-b" style={{ borderColor: "var(--border)" }}>
                            <span className="text-lg font-bold" style={{ color: "var(--accent)" }}>⬡</span>
                        </div>
                        <div className="flex flex-col gap-1 p-2 flex-1">
                            {NAV_ITEMS.map((item) => (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    title={item.label}
                                    className="flex items-center justify-center w-10 h-10 rounded-lg text-lg transition-colors hover:opacity-80"
                                    style={{ backgroundColor: "transparent" }}
                                >
                                    {item.icon}
                                </Link>
                            ))}
                        </div>
                    </nav>
                    {/* Page content */}
                    <main className="flex-1 overflow-hidden">
                        {children}
                    </main>
                </div>
            </body>
        </html>
    );
}
