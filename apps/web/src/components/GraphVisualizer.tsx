"use client";

import dynamic from "next/dynamic";
import { useEffect, useState, useCallback, useMemo } from "react";

// react-force-graph-2d uses canvas which requires window object. Dynamic import avoids SSR issues in Next.js
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

interface GraphVisualizerProps {
    graphData: { nodes: any[]; edges: any[] } | null;
    isLoading?: boolean;
}

export function GraphVisualizer({ graphData, isLoading = false }: GraphVisualizerProps) {
    const [dimensions, setDimensions] = useState({ width: 600, height: 600 });
    const [data, setData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });

    useEffect(() => {
        setDimensions({ width: window.innerWidth * 0.33, height: window.innerHeight });
        function handleResize() {
            const container = document.getElementById("graph-container");
            if (container) {
                setDimensions({
                    width: container.clientWidth,
                    height: container.clientHeight,
                });
            }
        }

        window.addEventListener("resize", handleResize);
        handleResize(); // Initial measurement

        return () => window.removeEventListener("resize", handleResize);
    }, []);

    useEffect(() => {
        if (graphData && graphData.nodes?.length > 0) {
            // Map edges to links structure expected by force-graph
            const links = graphData.edges.map((e: any) => ({
                source: e.source,
                target: e.target,
                type: e.type,
            }));
            setData({ nodes: graphData.nodes, links });
        } else {
            setData({ nodes: [], links: [] });
        }
    }, [graphData]);

    // Color logic for node clustering
    const getNodeColor = useCallback((node: any) => {
        switch (node.label) {
            case "DOCUMENT": return "#3b82f6"; // blue
            case "CHUNK": return "#8b5cf6"; // lighter blue
            case "ENTITY": return "#f59e0b"; // purple
            default: return "#9ca3af"; // gray
        }
    }, []);

    return (
        <div id="graph-container" className="h-full w-full bg-gray-50 border-l dark:bg-zinc-900/50 dark:border-zinc-800 relative overflow-hidden flex flex-col">
            <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur-sm p-3 rounded-md shadow-[0_0_10px_rgba(0,0,0,0.05)] border font-mono text-xs dark:bg-zinc-950/90 dark:border-zinc-800">
                <div className="font-semibold mb-2">Live Context</div>
                <div className="flex flex-col gap-2">
                    <div className="flex gap-2 items-center"><div className="w-3 h-3 rounded-full bg-blue-500"></div> Document</div>
                    <div className="flex gap-2 items-center"><div className="w-3 h-3 rounded-full bg-indigo-400"></div> Chunk</div>
                    <div className="flex gap-2 items-center"><div className="w-3 h-3 rounded-full bg-orange-400"></div> Entity</div>
                </div>
            </div>

            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/20 backdrop-blur-[1px] z-20">
                    <div className="flex flex-col items-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                        <p className="text-sm font-medium mt-4 text-gray-500">Retrieving context subgraph...</p>
                    </div>
                </div>
            )}

            {data.nodes.length === 0 && !isLoading ? (
                <div className="flex-1 flex flex-col items-center justify-center opacity-50 p-6 text-center text-sm font-medium h-full w-full">
                    Ask a question to see the AI's traversal path and retrieved context mapped here.
                </div>
            ) : (
                <ForceGraph2D
                    width={dimensions.width}
                    height={dimensions.height}
                    graphData={data}
                    nodeLabel="name"
                    nodeColor={getNodeColor}
                    nodeRelSize={6}
                    linkColor={() => "#cbd5e1"}
                    linkWidth={1.5}
                    linkDirectionalArrowLength={3.5}
                    linkDirectionalArrowRelPos={1}
                    onNodeClick={(node) => {
                        // Optional: Handle interaction
                        console.log(node);
                    }}
                />
            )}
        </div>
    );
}
