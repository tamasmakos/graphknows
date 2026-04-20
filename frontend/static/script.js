<<<<<<< HEAD:frontend/static/script.js
const AGENT_API = "http://127.0.0.1:8010";
const MGMT_API = "";

// D3 Global State
let simulation = null;
let svg = null;
let container = null;
let graphData = { nodes: [], links: [] };
let nodeElements = null;
let linkElements = null;
let linkLabelElements = null;
let textElements = null;
let transformState = d3.zoomIdentity;

// Selection State
let selectedNodes = new Set(); // Stores IDs

// --- Colors ---
const COLORS = {
    "TOPIC": "#FF7043",
    "SUBTOPIC": "#FFA726",
    "ENTITY_CONCEPT": "#AB47BC",
    "ONTOLOGY_CLASS": "#7E57C2",
    "PLACE": "#42A5F5",
    "CONTEXT": "#26C6DA",
    "CHUNK": "#78909C",
    "SEGMENT": "#66BB6A",
    "EPISODE": "#9CCC65",
    "CONVERSATION": "#EC407A",
    "DAY": "#8D6E63",
    "DEFAULT": "#8e918f"
};

function getColor(d) {
    const label = d.labels ? d.labels[0] : "Unknown";
    return COLORS[label] || COLORS.DEFAULT;
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    initD3Graph();
    // fetchSampleGraph(); // Disabled preloading
    fetchNodeTypes();
    fetchPgStats();
});

// --- Explorer & Controls ---

function toggleExplorer() {
    const panel = document.getElementById('explorer-panel');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        panel.classList.add('flex');
    } else {
        panel.classList.add('hidden');
        panel.classList.remove('flex');
    }
}

async function fetchPgStats() {
    try {
        const res = await fetch(`/api/graph/stats/pgvector`);
        const data = await res.json();

        if (data.error) {
            console.error("PG Stats error:", data.error);
            return;
        }

        document.getElementById('pg-count').innerText = data.row_count || 0;
        document.getElementById('pg-size').innerText = data.table_size || '0 B';

    } catch (e) {
        console.error("Failed to fetch PG stats", e);
    }
}

async function fetchNodeTypes() {
    try {
        const container = document.getElementById('type-filters');
        if (!container) return;

        const res = await fetch(`/api/graph/labels`);
        const data = await res.json();

        if (data.labels && data.labels.length > 0) {
            container.innerHTML = data.labels.map(label => `
                <label class="flex items-center gap-2 cursor-pointer group">
                    <input type="checkbox" value="${label}" class="accent-[#a8c7fa] rounded-sm bg-[#333537] border-[#444746]" checked>
                    <span class="text-xs text-[#c4c7c5] group-hover:text-[#e3e3e3]">${label}</span>
                </label>
            `).join('');
        } else {
            container.innerHTML = '<div class="text-xs text-[#8e918f]">No types found</div>';
        }
    } catch (e) {
        console.error("Failed to fetch types", e);
        const container = document.getElementById('type-filters');
        if (container) container.innerHTML = '<div class="text-xs text-red-400">Error loading types</div>';
    }
}

async function fetchSampleWithFilters() {
    const checkboxes = document.querySelectorAll('#type-filters input[type="checkbox"]:checked');
    const types = Array.from(checkboxes).map(cb => cb.value).join(',');

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const url = `${MGMT_API}/api/graph/sample?limit=50${types ? `&types=${encodeURIComponent(types)}` : ''}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (data.edges) data.links = data.edges;

        updateGraphData(data);
    } catch (e) {
        console.error("Fetch sample failed", e);
        alert("Failed to fetch sample: " + e.message);
    }
}



async function runCypherQuery(query) {
    if (!query) return;

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const res = await fetch(`/api/graph/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        // Process results to match graph format
        // Result is list of dicts. We need to extract nodes and rels.

        const nodes = new Map();
        const links = [];

        data.result.forEach(row => {
            Object.values(row).forEach(item => {
                if (!item) return;

                // If item looks like a node (has labels, properties, id)
                if (item.labels && item.id !== undefined) {
                    // Normalize
                    const nid = item.properties && item.properties.id ? item.properties.id : item.id.toString();
                    nodes.set(nid, {
                        id: nid,
                        labels: item.labels,
                        properties: item.properties || {},
                        element_id: item.id.toString()
                    });
                }

                // If item looks like a relationship (has relation, src_node, dest_node)
                if (item.relation && item.src_node !== undefined) {
                    links.push(item);
                }
            });
        });

        // Second pass for links to resolve IDs
        const finalLinks = [];
        links.forEach(l => {
            // Find source/target strings from nodes map based on internal IDs
            let source = null;
            let target = null;

            for (const [nid, node] of nodes.entries()) {
                if (node.element_id === l.src_node.toString()) source = nid;
                if (node.element_id === l.dest_node.toString()) target = nid;
            }

            if (source && target) {
                finalLinks.push({
                    id: l.id.toString(),
                    source: source,
                    target: target,
                    type: l.relation,
                    properties: l.properties || {}
                });
            }
        });

        const graphData = {
            nodes: Array.from(nodes.values()),
            edges: finalLinks
        };

        updateGraphData(graphData);
        alert(`Loaded ${graphData.nodes.length} nodes and ${graphData.edges.length} edges.`);

    } catch (e) {
        console.error("Query failed", e);
        alert("Query failed: " + e.message);
    }
}

// function runLimitedSample() { ... } replaced below
async function runLimitedSample() {
    const limit = document.getElementById('query-limit').value || 50;

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const res = await fetch(`/api/graph/sample?limit=${limit}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (data.edges) data.links = data.edges;

        updateGraphData(data);

        // Optional: show stats in UI instead of alert
        // document.getElementById('stat-nodes').innerText = data.nodes.length; // updateGraphData does this

    } catch (e) {
        console.error("Failed to load sample", e);
        alert("Failed to load sample: " + e.message);
    }
}

// "Find Path" removed to simplify UI as requested, 
// keeping backend support for future use.

// --- Chat Logic ---
let chatHistory = [];

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const status = document.getElementById('chat-status');
    const text = input.value.trim();
    if (!text) return;

    addMessage('user', text);
    input.value = '';

    status.classList.remove('hidden');
    status.classList.add('flex');
    const useAgent = document.getElementById('use-agent').checked;

    try {
        const payload = {
            query: text,
            messages: chatHistory,
            use_agent: useAgent
        };

        const response = await fetch(`${AGENT_API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("API Request Failed");
        const data = await response.json();

        addMessage('assistant', data.answer, data);

        if (data.graph_data && data.graph_data.nodes.length > 0) {
            // Clear previous graph and show ONLY the context
            updateGraphData(data.graph_data, true);
        }

        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });

    } catch (e) {
        addMessage('assistant', `Error: ${e.message}`);
    } finally {
        status.classList.add('hidden');
        status.classList.remove('flex');
    }
}

function addMessage(role, content, data = null) {
    const container = document.getElementById('chat-messages');

    if (container.children.length === 1 && container.children[0].innerText.includes("explore")) {
        container.innerHTML = '';
    }

    const wrapper = document.createElement('div');
    wrapper.className = `flex w-full mb-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;

    const isUser = role === 'user';
    const bubbleClass = isUser
        ? 'bg-[#0842a0] text-[#d6e3ff] rounded-2xl rounded-tr-sm'
        : 'bg-[#333537] text-[#e3e3e3] rounded-2xl rounded-tl-sm border border-[#444746]';

    let html = `
        <div class="max-w-[90%] px-4 py-3 shadow-sm ${bubbleClass} text-sm leading-relaxed">
            <div>${content}</div>
    `;

    if (data) {
        html += `<div class="mt-2 pt-2 border-t border-[#444746]/50 space-y-2">`;

        // Execution Time & Detailed Timings
        if (data.step_timings) {
            const totalTime = data.step_timings.synthesize_answer
                ? (data.execution_time || Object.values(data.step_timings).reduce((a, b) => a + b, 0))
                : data.execution_time;

            html += `
                <div class="text-[10px] text-[#c4c7c5] flex flex-col gap-2 mt-2 bg-[#28292a] p-2 rounded border border-[#444746]/50">
                    <div class="flex items-center justify-between pb-1 border-b border-[#444746]/50 mb-1">
                        <span class="font-bold text-[#e3e3e3] flex items-center gap-1"><span class="material-icons-round text-[12px]">timer</span> Performance Metrics</span>
                        <span class="font-mono text-[#a8c7fa]">${totalTime ? totalTime.toFixed(2) : '0.00'}s</span>
                    </div>
                    <div class="space-y-1.5">
             `;

            // Sort by time desc
            const sortedTimings = Object.entries(data.step_timings).sort((a, b) => b[1] - a[1]);
            const maxTime = sortedTimings.length > 0 ? sortedTimings[0][1] : 1;

            sortedTimings.forEach(([step, time]) => {
                const width = (time / totalTime) * 100;
                // Color mapping for known steps
                let barColor = '#444746';
                if (step.includes('extract')) barColor = '#FF7043';
                if (step.includes('seed')) barColor = '#AB47BC';
                if (step.includes('expand')) barColor = '#42A5F5';
                if (step.includes('synthesize')) barColor = '#9CCC65';

                html += `
                <div class="flex flex-col gap-0.5">
                    <div class="flex justify-between items-center text-[9px] text-[#8e918f]">
                        <span class="uppercase tracking-wider">${step.replace(/_/g, ' ')}</span>
                        <span class="font-mono text-[#c4c7c5]">${time.toFixed(3)}s</span>
                    </div>
                    <div class="h-1.5 w-full bg-[#1e1f20] rounded-full overflow-hidden">
                        <div class="h-full rounded-full" style="width: ${Math.min(width, 100)}%; background-color: ${barColor}"></div>
                    </div>
                </div>`;
            });
            html += `</div></div>`;
        } else if (data.execution_time) {
            html += `<div class="text-[10px] text-[#c4c7c5] flex items-center gap-1"><span class="material-icons-round text-[10px]">timer</span> ${data.execution_time.toFixed(2)}s</div>`;
        }

        // Seeds
        if ((data.seed_entities && data.seed_entities.length > 0) || (data.seed_topics && data.seed_topics.length > 0)) {
            html += `<div class="flex flex-wrap gap-1 mt-1">`;
            if (data.seed_topics && data.seed_topics.length > 0) {
                html += data.seed_topics.map(s => `<span class="bg-[#FF7043]/20 text-[#FF7043] px-1.5 py-0.5 rounded text-[9px] font-bold border border-[#FF7043]/30">TOPIC: ${s}</span>`).join('');
            }
            if (data.seed_entities && data.seed_entities.length > 0) {
                html += data.seed_entities.map(s => `<span class="bg-[#AB47BC]/20 text-[#AB47BC] px-1.5 py-0.5 rounded text-[9px] font-bold border border-[#AB47BC]/30">ENTITY: ${s}</span>`).join('');
            }
            html += `</div>`;
        }

        if (data.reasoning_chain && data.reasoning_chain.length > 0) {
            html += `
                <div class="bg-[#1e1f20]/50 rounded p-2 text-[10px] text-[#c4c7c5] border border-[#444746]/30">
                    <div class="font-bold text-[#a8c7fa] mb-1 text-[9px]">REASONING</div>
                    ${data.reasoning_chain.map(s => `<div class="truncate">• ${s}</div>`).join('')}
                </div>
            `;
        }

        // Full Context Display
        if (data.context) {
            const contextId = `ctx-${Date.now()}`;
            html += `
                <div class="mt-2">
                    <button onclick="document.getElementById('${contextId}').classList.toggle('hidden')" class="text-[10px] text-[#a8c7fa] hover:text-[#d6e3ff] flex items-center gap-1 cursor-pointer w-full text-left">
                        <span class="material-icons-round text-[12px]">description</span> View Retrieved Context
                    </button>
                    <div id="${contextId}" class="hidden mt-2 p-2 bg-[#1e1f20] rounded border border-[#444746] text-[10px] font-mono text-[#c4c7c5] whitespace-pre-wrap overflow-x-auto max-h-60 scrollbar-thin">
                        ${data.context.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                    </div>
                </div>
             `;
        }

        if (data.full_prompt) {
            const promptId = `prompt-${Date.now()}`;
            html += `
                <div class="mt-1">
                    <button onclick="document.getElementById('${promptId}').classList.toggle('hidden')" class="text-[10px] text-[#8e918f] hover:text-[#c4c7c5] flex items-center gap-1 cursor-pointer w-full text-left">
                        <span class="material-icons-round text-[12px]">code</span> View Full Prompt
                    </button>
                    <div id="${promptId}" class="hidden mt-2 p-2 bg-[#1e1f20] rounded border border-[#444746] text-[10px] font-mono text-[#c4c7c5] whitespace-pre-wrap overflow-x-auto max-h-60 scrollbar-thin">
                        ${data.full_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                    </div>
                </div>
             `;
        }

        html += `</div>`;
    }

    html += `</div>`;
    wrapper.innerHTML = html;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}


// --- D3.js Implementation ---

function initD3Graph() {
    const containerEl = document.getElementById('graph-container');
    const width = containerEl.clientWidth;
    const height = containerEl.clientHeight;

    const zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
            transformState = event.transform;
            container.attr("transform", event.transform);
            ticked();
        });

    svg = d3.select("#graph-container").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("background-color", "#121212")
        .call(zoom)
        .on("dblclick.zoom", null)
        .on("click", (e) => {
            if (e.target.tagName === 'svg' && !e.shiftKey) {
                clearSelection();
            }
        });

    container = svg.append("g");

    simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(100)) // Increased distance for labels
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(30).iterations(2));

    container.append("g").attr("class", "links");
    container.append("g").attr("class", "link-labels"); // New group for edge labels
    container.append("g").attr("class", "nodes");
    container.append("g").attr("class", "labels");

    window.addEventListener('resize', () => {
        const w = containerEl.clientWidth;
        const h = containerEl.clientHeight;
        svg.attr("width", w).attr("height", h).attr("viewBox", [0, 0, w, h]);
        simulation.force("center", d3.forceCenter(w / 2, h / 2));
        simulation.alpha(0.3).restart();
    });
}

function getLinkId(l) {
    const sid = typeof l.source === 'object' ? l.source.id : l.source;
    const tid = typeof l.target === 'object' ? l.target.id : l.target;
    return `${sid}-${tid}-${l.type || 'rel'}`; // Include type in ID for uniqueness
}

function updateGraphData(data, reset = false) {
    if (reset) {
        graphData = { nodes: [], links: [] };
        simulation.nodes([]);
        simulation.force("link").links([]);
        // Clear all visuals
        container.select(".nodes").selectAll("*").remove();
        container.select(".links").selectAll("*").remove();
        container.select(".labels").selectAll("*").remove();
        container.select(".link-labels").selectAll("*").remove();
    }

    if (data.edges) data.links = data.edges;

    const nodeMap = new Map(graphData.nodes.map(n => [n.id, n]));

    let newNodesCount = 0;
    data.nodes.forEach(n => {
        if (!n || !n.id) return;
        if (!nodeMap.has(n.id)) {
            // Init pos: Random scatter near center to help simulation
            n.x = (Math.random() - 0.5) * 50;
            n.y = (Math.random() - 0.5) * 50;
            nodeMap.set(n.id, n);
            newNodesCount++;
        }
    });

    const linkMap = new Map();
    graphData.links.forEach(l => linkMap.set(getLinkId(l), l));

    data.links.forEach(l => {
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;

        if (!sid || !tid) return;

        if (nodeMap.has(sid) && nodeMap.has(tid)) {
            const id = `${sid}-${tid}-${l.type || 'rel'}`;
            if (!linkMap.has(id)) {
                l.source = nodeMap.get(sid);
                l.target = nodeMap.get(tid);
                linkMap.set(id, l);
            }
        }
    });

    graphData.nodes = Array.from(nodeMap.values());
    graphData.links = Array.from(linkMap.values());

    document.getElementById('stat-nodes').innerText = graphData.nodes.length;
    document.getElementById('stat-edges').innerText = graphData.links.length;

    renderGraph();

    if (newNodesCount > 0 || reset) {
        simulation.alpha(1).restart();
    }
}

function renderGraph() {
    // Links
    const linksGroup = container.select(".links");
    linkElements = linksGroup.selectAll("line")
        .data(graphData.links, d => getLinkId(d));

    linkElements.exit().remove();

    const linkEnter = linkElements.enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", 1)
        .attr("stroke", "#444746")
        .on("click", (event, d) => handleEdgeClick(event, d));

    linkElements = linkEnter.merge(linkElements);

    // Link Labels
    const linkLabelsGroup = container.select(".link-labels");
    linkLabelElements = linkLabelsGroup.selectAll("text")
        .data(graphData.links, d => getLinkId(d));

    linkLabelElements.exit().remove();

    const linkLabelEnter = linkLabelElements.enter().append("text")
        .text(d => d.type)
        .attr("text-anchor", "middle")
        .attr("font-size", "8px")
        .attr("fill", "#8e918f")
        .attr("dy", -3); // Offset above line

    linkLabelElements = linkLabelEnter.merge(linkLabelElements);

    // Nodes
    const nodesGroup = container.select(".nodes");
    nodeElements = nodesGroup.selectAll("circle")
        .data(graphData.nodes, d => d.id);

    nodeElements.exit().transition().duration(500).attr("r", 0).remove();

    const nodeEnter = nodeElements.enter().append("circle")
        .attr("class", "node")
        .attr("r", 0)
        .attr("fill", d => getColor(d))
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("click", (event, d) => handleNodeClick(event, d))
        .on("mouseover", (event, d) => showTooltip(event, d))
        .on("mouseout", hideTooltip);

    nodeEnter.transition().duration(500).attr("r", 6);

    nodeElements = nodeEnter.merge(nodeElements);

    updateSelectionVisuals();

    // Labels (Nodes)
    const labelsGroup = container.select(".labels");
    textElements = labelsGroup.selectAll("text")
        .data(graphData.nodes, d => d.id);

    textElements.exit().remove();

    const textEnter = textElements.enter().append("text")
        .attr("dy", 15)
        .attr("text-anchor", "middle")
        .attr("fill", "#e3e3e3")
        .attr("font-size", "10px")
        .attr("opacity", 0)
        .text(d => d.properties.name || d.id);

    textElements = textEnter.merge(textElements);

    // Restart
    simulation.nodes(graphData.nodes).on("tick", ticked);
    simulation.force("link").links(graphData.links);
}

function ticked() {
    linkElements
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    // Update Edge Labels (Center of line)
    linkLabelElements
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2)
        .attr("transform", d => {
            // Optional: Rotate text to align with line
            // For now, keep horizontal for readability
            return "";
        })
        .attr("opacity", transformState.k > 1.2 ? 1 : 0); // Hide when zoomed out

    nodeElements
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

    const k = transformState.k;
    textElements
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("opacity", k > 1.2 ? 1 : 0);
}

// --- Drag ---
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    hideTooltip();
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// --- Interactions ---

function handleNodeClick(event, d) {
    if (event.shiftKey) {
        if (selectedNodes.has(d.id)) {
            selectedNodes.delete(d.id);
        } else {
            selectedNodes.add(d.id);
        }
        updateSelectionVisuals();
        return;
    }

    selectedNodes.clear();
    selectedNodes.add(d.id);
    updateSelectionVisuals();

    showNodeDetails(d);
}

function handleEdgeClick(event, d) {
    selectedNodes.clear();
    updateSelectionVisuals();
    showEdgeDetails(d);
}

function clearSelection() {
    selectedNodes.clear();
    updateSelectionVisuals();
    closeDetails();
}

function updateSelectionVisuals() {
    if (!nodeElements) return;
    nodeElements
        .attr("stroke", d => selectedNodes.has(d.id) ? "#fff" : "#121212")
        .attr("stroke-width", d => selectedNodes.has(d.id) ? 2.5 : 1.5)
        .attr("r", d => selectedNodes.has(d.id) ? 8 : 6);
}

function showNodeDetails(d) {
    const panel = document.getElementById('node-details');
    panel.classList.remove('translate-x-[110%]');
    document.getElementById('expand-button-container').classList.remove('hidden');

    document.getElementById('detail-title').innerText = d.properties.name || d.id;
    const content = document.getElementById('detail-content');
    const type = d.properties.ontology_class || d.labels.join(', ');

    let html = `<div class="mb-4"><span class="px-2 py-1 rounded bg-[#333537] text-[#a8c7fa] text-xs font-bold uppercase tracking-wider">${type}</span></div>`;

    Object.entries(d.properties).forEach(([k, v]) => {
        if (!['embedding', 'embeddings'].includes(k.toLowerCase())) {
            html += `
                <div class="mb-3 border-b border-[#444746]/50 pb-2 last:border-0">
                    <div class="text-[#8e918f] text-[10px] uppercase font-bold mb-1">${k}</div>
                    <div class="text-[#e3e3e3] text-sm break-words">${typeof v === 'object' ? JSON.stringify(v) : v}</div>
                </div>`;
        }
    });
    content.innerHTML = html;
    window._currentNodeId = d.id;
}

function showEdgeDetails(d) {
    const panel = document.getElementById('node-details');
    panel.classList.remove('translate-x-[110%]');
    document.getElementById('expand-button-container').classList.add('hidden');

    document.getElementById('detail-title').innerText = d.type || 'Relationship';
    const content = document.getElementById('detail-content');

    let html = `
        <div class="mb-4 flex flex-col gap-2 p-2 bg-[#28292a] rounded border border-[#444746]">
            <div class="flex flex-col">
                <span class="text-[9px] uppercase font-bold text-[#8e918f]">Source</span>
                <span class="text-xs text-[#a8c7fa] truncate">${d.source.properties?.name || d.source.id}</span>
            </div>
            <div class="flex items-center justify-center -my-1">
                <span class="material-icons-round text-xs text-[#444746]">arrow_downward</span>
            </div>
            <div class="flex flex-col">
                <span class="text-[9px] uppercase font-bold text-[#8e918f]">Target</span>
                <span class="text-xs text-[#a8c7fa] truncate">${d.target.properties?.name || d.target.id}</span>
            </div>
        </div>
    `;

    if (d.properties) {
        Object.entries(d.properties).forEach(([k, v]) => {
            if (!['embedding', 'embeddings'].includes(k.toLowerCase())) {
                html += `
                    <div class="mb-3 border-b border-[#444746]/50 pb-2 last:border-0">
                        <div class="text-[#8e918f] text-[10px] uppercase font-bold mb-1">${k}</div>
                        <div class="text-[#e3e3e3] text-sm break-words">${typeof v === 'object' ? JSON.stringify(v) : v}</div>
                    </div>`;
            }
        });
    }
    content.innerHTML = html;
}

function closeDetails() {
    document.getElementById('node-details').classList.add('translate-x-[110%]');
}

async function expandSelectedNode() {
    const nodeId = window._currentNodeId;
    if (!nodeId) return;
    try {
        const res = await fetch(`/api/graph/node/${encodeURIComponent(nodeId)}/expand`);
        const data = await res.json();
        if (data.error) {
            alert("Failed to expand: " + data.error);
            return;
        }
        updateGraphData(data);
    } catch (e) {
        console.error("Expand request failed", e);
    }
}

// Tooltip
function showTooltip(event, d) {
    const tooltip = document.getElementById('graph-tooltip');
    tooltip.style.opacity = 1;
    tooltip.innerHTML = `
        <div class="font-bold text-[#e3e3e3] mb-1">${d.properties.name || d.id}</div>
        <div class="text-[#a8c7fa] text-xs">${d.labels.join(', ')}</div>
    `;
    const x = event.pageX + 10;
    const y = event.pageY + 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

function hideTooltip() {
    document.getElementById('graph-tooltip').style.opacity = 0;
}



async function fetchSampleGraph() {
    try {
        // Use updated sample endpoint logic without manual limit if possible, or limit 50
        const res = await fetch(`/api/graph/sample?limit=50`);
        const data = await res.json();
        updateGraphData(data);
    } catch (e) {
        console.error("Failed to load sample", e);
    }
}
=======
const AGENT_API = "http://127.0.0.1:8010";
const MGMT_API = "";

// D3 Global State
let simulation = null;
let svg = null;
let container = null;
let graphData = { nodes: [], links: [] };
let nodeElements = null;
let linkElements = null;
let linkLabelElements = null;
let textElements = null;
let transformState = d3.zoomIdentity;

// Selection State
let selectedNodes = new Set(); // Stores IDs

// --- Colors ---
const COLORS = {
    "TOPIC": "#FF7043",
    "SUBTOPIC": "#FFA726",
    "ENTITY_CONCEPT": "#AB47BC",
    "ONTOLOGY_CLASS": "#7E57C2",
    "PLACE": "#42A5F5",
    "CONTEXT": "#26C6DA",
    "CHUNK": "#78909C",
    "SEGMENT": "#66BB6A",
    "EPISODE": "#9CCC65",
    "CONVERSATION": "#EC407A",
    "DAY": "#8D6E63",
    "DEFAULT": "#8e918f"
};

function getColor(d) {
    const label = d.labels ? d.labels[0] : "Unknown";
    return COLORS[label] || COLORS.DEFAULT;
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    initD3Graph();
    // fetchSampleGraph(); // Disabled preloading
    fetchNodeTypes();
    fetchPgStats();
});

// --- Explorer & Controls ---

function toggleExplorer() {
    const panel = document.getElementById('explorer-panel');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        panel.classList.add('flex');
    } else {
        panel.classList.add('hidden');
        panel.classList.remove('flex');
    }
}

async function fetchPgStats() {
    try {
        const res = await fetch(`/api/graph/stats/pgvector`);
        const data = await res.json();

        if (data.error) {
            console.error("Stats error:", data.error);
            return;
        }

        // FalkorDB
        if (data.falkordb) {
            const n = document.getElementById('db-nodes');
            const e = document.getElementById('db-edges');
            const s = document.getElementById('falkor-size');
            if (n) n.innerText = data.falkordb.nodes || 0;
            if (e) e.innerText = data.falkordb.edges || 0;
            if (s) s.innerText = data.falkordb.bytes_fmt || 'N/A';
        }

        // Postgres
        if (data.pgvector) {
            const c = document.getElementById('pg-count');
            const z = document.getElementById('pg-size');
            if (c) c.innerText = data.pgvector.rows || 0;
            if (z) z.innerText = data.pgvector.size || '0 B';
        }

    } catch (e) {
        console.error("Failed to fetch DB stats", e);
    }
}

async function fetchNodeTypes() {
    try {
        const container = document.getElementById('type-filters');
        if (!container) return;

        const res = await fetch(`/api/graph/labels`);
        const data = await res.json();

        if (data.labels && data.labels.length > 0) {
            container.innerHTML = data.labels.map(label => `
                <label class="flex items-center gap-2 cursor-pointer group">
                    <input type="checkbox" value="${label}" class="accent-[#a8c7fa] rounded-sm bg-[#333537] border-[#444746]" checked>
                    <span class="text-xs text-[#c4c7c5] group-hover:text-[#e3e3e3]">${label}</span>
                </label>
            `).join('');
        } else {
            container.innerHTML = '<div class="text-xs text-[#8e918f]">No types found</div>';
        }
    } catch (e) {
        console.error("Failed to fetch types", e);
        const container = document.getElementById('type-filters');
        if (container) container.innerHTML = '<div class="text-xs text-red-400">Error loading types</div>';
    }
}

async function fetchSampleWithFilters() {
    const checkboxes = document.querySelectorAll('#type-filters input[type="checkbox"]:checked');
    const types = Array.from(checkboxes).map(cb => cb.value).join(',');

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const url = `${MGMT_API}/api/graph/sample?limit=50${types ? `&types=${encodeURIComponent(types)}` : ''}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (data.edges) data.links = data.edges;

        updateGraphData(data);
    } catch (e) {
        console.error("Fetch sample failed", e);
        alert("Failed to fetch sample: " + e.message);
    }
}



async function runCypherQuery(query) {
    if (!query) return;

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const res = await fetch(`/api/graph/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        // Process results to match graph format
        // Result is list of dicts. We need to extract nodes and rels.

        const nodes = new Map();
        const links = [];

        data.result.forEach(row => {
            Object.values(row).forEach(item => {
                if (!item) return;

                // If item looks like a node (has labels, properties, id)
                if (item.labels && item.id !== undefined) {
                    // Normalize
                    const nid = item.properties && item.properties.id ? item.properties.id : item.id.toString();
                    nodes.set(nid, {
                        id: nid,
                        labels: item.labels,
                        properties: item.properties || {},
                        element_id: item.id.toString()
                    });
                }

                // If item looks like a relationship (has relation, src_node, dest_node)
                if (item.relation && item.src_node !== undefined) {
                    links.push(item);
                }
            });
        });

        // Second pass for links to resolve IDs
        const finalLinks = [];
        links.forEach(l => {
            // Find source/target strings from nodes map based on internal IDs
            let source = null;
            let target = null;

            for (const [nid, node] of nodes.entries()) {
                if (node.element_id === l.src_node.toString()) source = nid;
                if (node.element_id === l.dest_node.toString()) target = nid;
            }

            if (source && target) {
                finalLinks.push({
                    id: l.id.toString(),
                    source: source,
                    target: target,
                    type: l.relation,
                    properties: l.properties || {}
                });
            }
        });

        const graphData = {
            nodes: Array.from(nodes.values()),
            edges: finalLinks
        };

        updateGraphData(graphData);
        alert(`Loaded ${graphData.nodes.length} nodes and ${graphData.edges.length} edges.`);

    } catch (e) {
        console.error("Query failed", e);
        alert("Query failed: " + e.message);
    }
}

// function runLimitedSample() { ... } replaced below
async function runLimitedSample() {
    const limit = document.getElementById('query-limit').value || 50;

    // Clear graph for fresh view
    updateGraphData({ nodes: [], links: [] }, true);

    try {
        const res = await fetch(`/api/graph/sample?limit=${limit}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (data.edges) data.links = data.edges;

        updateGraphData(data);

        // Optional: show stats in UI instead of alert
        // document.getElementById('stat-nodes').innerText = data.nodes.length; // updateGraphData does this

    } catch (e) {
        console.error("Failed to load sample", e);
        alert("Failed to load sample: " + e.message);
    }
}

// "Find Path" removed to simplify UI as requested, 
// keeping backend support for future use.

// --- Chat Logic ---
let chatHistory = [];

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const status = document.getElementById('chat-status');
    const text = input.value.trim();
    if (!text) return;

    addMessage('user', text);
    input.value = '';

    status.classList.remove('hidden');
    status.classList.add('flex');
    const useAgent = document.getElementById('use-agent').checked;

    try {
        const payload = {
            query: text,
            messages: chatHistory,
            use_agent: useAgent
        };

        const response = await fetch(`${AGENT_API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("API Request Failed");
        const data = await response.json();

        addMessage('assistant', data.answer, data);

        if (data.graph_data && data.graph_data.nodes.length > 0) {
            // Clear previous graph and show ONLY the context
            updateGraphData(data.graph_data, true);
        }

        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });

    } catch (e) {
        addMessage('assistant', `Error: ${e.message}`);
    } finally {
        status.classList.add('hidden');
        status.classList.remove('flex');
    }
}

function addMessage(role, content, data = null) {
    const container = document.getElementById('chat-messages');

    if (container.children.length === 1 && container.children[0].innerText.includes("explore")) {
        container.innerHTML = '';
    }

    const wrapper = document.createElement('div');
    wrapper.className = `flex w-full mb-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;

    const isUser = role === 'user';
    const bubbleClass = isUser
        ? 'bg-[#0842a0] text-[#d6e3ff] rounded-2xl rounded-tr-sm'
        : 'bg-[#333537] text-[#e3e3e3] rounded-2xl rounded-tl-sm border border-[#444746]';

    let html = `
        <div class="max-w-[90%] px-4 py-3 shadow-sm ${bubbleClass} text-sm leading-relaxed">
            <div>${content}</div>
    `;

    if (data) {
        html += `<div class="mt-2 pt-2 border-t border-[#444746]/50 space-y-2">`;

        // Execution Time & Detailed Timings
        if (data.step_timings) {
            const totalTime = data.step_timings.synthesize_answer
                ? (data.execution_time || Object.values(data.step_timings).reduce((a, b) => a + b, 0))
                : data.execution_time;

            html += `
                <div class="text-[10px] text-[#c4c7c5] flex flex-col gap-2 mt-2 bg-[#28292a] p-2 rounded border border-[#444746]/50">
                    <div class="flex items-center justify-between pb-1 border-b border-[#444746]/50 mb-1">
                        <span class="font-bold text-[#e3e3e3] flex items-center gap-1"><span class="material-icons-round text-[12px]">timer</span> Performance Metrics</span>
                        <span class="font-mono text-[#a8c7fa]">${totalTime ? totalTime.toFixed(2) : '0.00'}s</span>
                    </div>
                    <div class="space-y-1.5">
             `;

            // Sort by time desc
            const sortedTimings = Object.entries(data.step_timings).sort((a, b) => b[1] - a[1]);
            const maxTime = sortedTimings.length > 0 ? sortedTimings[0][1] : 1;

            sortedTimings.forEach(([step, time]) => {
                const width = (time / totalTime) * 100;
                // Color mapping for known steps
                let barColor = '#444746';
                if (step.includes('extract')) barColor = '#FF7043';
                if (step.includes('seed')) barColor = '#AB47BC';
                if (step.includes('expand')) barColor = '#42A5F5';
                if (step.includes('synthesize')) barColor = '#9CCC65';

                html += `
                <div class="flex flex-col gap-0.5">
                    <div class="flex justify-between items-center text-[9px] text-[#8e918f]">
                        <span class="uppercase tracking-wider">${step.replace(/_/g, ' ')}</span>
                        <span class="font-mono text-[#c4c7c5]">${time.toFixed(3)}s</span>
                    </div>
                    <div class="h-1.5 w-full bg-[#1e1f20] rounded-full overflow-hidden">
                        <div class="h-full rounded-full" style="width: ${Math.min(width, 100)}%; background-color: ${barColor}"></div>
                    </div>
                </div>`;
            });
            html += `</div></div>`;
        } else if (data.execution_time) {
            html += `<div class="text-[10px] text-[#c4c7c5] flex items-center gap-1"><span class="material-icons-round text-[10px]">timer</span> ${data.execution_time.toFixed(2)}s</div>`;
        }

        // Seeds
        if ((data.seed_entities && data.seed_entities.length > 0) || (data.seed_topics && data.seed_topics.length > 0)) {
            html += `<div class="flex flex-wrap gap-1 mt-1">`;
            if (data.seed_topics && data.seed_topics.length > 0) {
                html += data.seed_topics.map(s => `<span class="bg-[#FF7043]/20 text-[#FF7043] px-1.5 py-0.5 rounded text-[9px] font-bold border border-[#FF7043]/30">TOPIC: ${s}</span>`).join('');
            }
            if (data.seed_entities && data.seed_entities.length > 0) {
                html += data.seed_entities.map(s => `<span class="bg-[#AB47BC]/20 text-[#AB47BC] px-1.5 py-0.5 rounded text-[9px] font-bold border border-[#AB47BC]/30">ENTITY: ${s}</span>`).join('');
            }
            html += `</div>`;
        }

        if (data.reasoning_chain && data.reasoning_chain.length > 0) {
            html += `
                <div class="bg-[#1e1f20]/50 rounded p-2 text-[10px] text-[#c4c7c5] border border-[#444746]/30">
                    <div class="font-bold text-[#a8c7fa] mb-1 text-[9px]">REASONING</div>
                    ${data.reasoning_chain.map(s => `<div class="truncate">• ${s}</div>`).join('')}
                </div>
            `;
        }

        // Full Context Display
        if (data.context) {
            const contextId = `ctx-${Date.now()}`;
            html += `
                <div class="mt-2">
                    <button onclick="document.getElementById('${contextId}').classList.toggle('hidden')" class="text-[10px] text-[#a8c7fa] hover:text-[#d6e3ff] flex items-center gap-1 cursor-pointer w-full text-left">
                        <span class="material-icons-round text-[12px]">description</span> View Retrieved Context
                    </button>
                    <div id="${contextId}" class="hidden mt-2 p-2 bg-[#1e1f20] rounded border border-[#444746] text-[10px] font-mono text-[#c4c7c5] whitespace-pre-wrap overflow-x-auto max-h-60 scrollbar-thin">
                        ${data.context.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                    </div>
                </div>
             `;
        }

        if (data.full_prompt) {
            const promptId = `prompt-${Date.now()}`;
            html += `
                <div class="mt-1">
                    <button onclick="document.getElementById('${promptId}').classList.toggle('hidden')" class="text-[10px] text-[#8e918f] hover:text-[#c4c7c5] flex items-center gap-1 cursor-pointer w-full text-left">
                        <span class="material-icons-round text-[12px]">code</span> View Full Prompt
                    </button>
                    <div id="${promptId}" class="hidden mt-2 p-2 bg-[#1e1f20] rounded border border-[#444746] text-[10px] font-mono text-[#c4c7c5] whitespace-pre-wrap overflow-x-auto max-h-60 scrollbar-thin">
                        ${data.full_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                    </div>
                </div>
             `;
        }

        html += `</div>`;
    }

    html += `</div>`;
    wrapper.innerHTML = html;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}


// --- D3.js Implementation ---

function initD3Graph() {
    const containerEl = document.getElementById('graph-container');
    const width = containerEl.clientWidth;
    const height = containerEl.clientHeight;

    const zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
            transformState = event.transform;
            container.attr("transform", event.transform);
            ticked();
        });

    svg = d3.select("#graph-container").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("background-color", "#121212")
        .call(zoom)
        .on("dblclick.zoom", null)
        .on("click", (e) => {
            if (e.target.tagName === 'svg' && !e.shiftKey) {
                clearSelection();
            }
        });

    container = svg.append("g");

    simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(100)) // Increased distance for labels
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(30).iterations(2));

    container.append("g").attr("class", "links");
    container.append("g").attr("class", "link-labels"); // New group for edge labels
    container.append("g").attr("class", "nodes");
    container.append("g").attr("class", "labels");

    window.addEventListener('resize', () => {
        const w = containerEl.clientWidth;
        const h = containerEl.clientHeight;
        svg.attr("width", w).attr("height", h).attr("viewBox", [0, 0, w, h]);
        simulation.force("center", d3.forceCenter(w / 2, h / 2));
        simulation.alpha(0.3).restart();
    });
}

function getLinkId(l) {
    const sid = typeof l.source === 'object' ? l.source.id : l.source;
    const tid = typeof l.target === 'object' ? l.target.id : l.target;
    return `${sid}-${tid}-${l.type || 'rel'}`; // Include type in ID for uniqueness
}

function updateGraphData(data, reset = false) {
    if (reset) {
        graphData = { nodes: [], links: [] };
        simulation.nodes([]);
        simulation.force("link").links([]);
        // Clear all visuals
        container.select(".nodes").selectAll("*").remove();
        container.select(".links").selectAll("*").remove();
        container.select(".labels").selectAll("*").remove();
        container.select(".link-labels").selectAll("*").remove();
    }

    if (data.edges) data.links = data.edges;

    const nodeMap = new Map(graphData.nodes.map(n => [n.id, n]));

    let newNodesCount = 0;
    data.nodes.forEach(n => {
        if (!n || !n.id) return;
        if (!nodeMap.has(n.id)) {
            // Init pos: Random scatter near center to help simulation
            n.x = (Math.random() - 0.5) * 50;
            n.y = (Math.random() - 0.5) * 50;
            nodeMap.set(n.id, n);
            newNodesCount++;
        }
    });

    const linkMap = new Map();
    graphData.links.forEach(l => linkMap.set(getLinkId(l), l));

    data.links.forEach(l => {
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;

        if (!sid || !tid) return;

        if (nodeMap.has(sid) && nodeMap.has(tid)) {
            const id = `${sid}-${tid}-${l.type || 'rel'}`;
            if (!linkMap.has(id)) {
                l.source = nodeMap.get(sid);
                l.target = nodeMap.get(tid);
                linkMap.set(id, l);
            }
        }
    });

    graphData.nodes = Array.from(nodeMap.values());
    graphData.links = Array.from(linkMap.values());

    document.getElementById('stat-nodes').innerText = graphData.nodes.length;
    document.getElementById('stat-edges').innerText = graphData.links.length;

    renderGraph();

    if (newNodesCount > 0 || reset) {
        simulation.alpha(1).restart();
    }
}

function renderGraph() {
    // Links
    const linksGroup = container.select(".links");
    linkElements = linksGroup.selectAll("line")
        .data(graphData.links, d => getLinkId(d));

    linkElements.exit().remove();

    const linkEnter = linkElements.enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", 1)
        .attr("stroke", "#444746")
        .on("click", (event, d) => handleEdgeClick(event, d));

    linkElements = linkEnter.merge(linkElements);

    // Link Labels
    const linkLabelsGroup = container.select(".link-labels");
    linkLabelElements = linkLabelsGroup.selectAll("text")
        .data(graphData.links, d => getLinkId(d));

    linkLabelElements.exit().remove();

    const linkLabelEnter = linkLabelElements.enter().append("text")
        .text(d => d.type)
        .attr("text-anchor", "middle")
        .attr("font-size", "8px")
        .attr("fill", "#8e918f")
        .attr("dy", -3); // Offset above line

    linkLabelElements = linkLabelEnter.merge(linkLabelElements);

    // Nodes
    const nodesGroup = container.select(".nodes");
    nodeElements = nodesGroup.selectAll("circle")
        .data(graphData.nodes, d => d.id);

    nodeElements.exit().transition().duration(500).attr("r", 0).remove();

    const nodeEnter = nodeElements.enter().append("circle")
        .attr("class", "node")
        .attr("r", 0)
        .attr("fill", d => getColor(d))
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("click", (event, d) => handleNodeClick(event, d))
        .on("mouseover", (event, d) => showTooltip(event, d))
        .on("mouseout", hideTooltip);

    nodeEnter.transition().duration(500).attr("r", 6);

    nodeElements = nodeEnter.merge(nodeElements);

    updateSelectionVisuals();

    // Labels (Nodes)
    const labelsGroup = container.select(".labels");
    textElements = labelsGroup.selectAll("text")
        .data(graphData.nodes, d => d.id);

    textElements.exit().remove();

    const textEnter = textElements.enter().append("text")
        .attr("dy", 15)
        .attr("text-anchor", "middle")
        .attr("fill", "#e3e3e3")
        .attr("font-size", "10px")
        .attr("opacity", 0)
        .text(d => d.properties.name || d.id);

    textElements = textEnter.merge(textElements);

    // Restart
    simulation.nodes(graphData.nodes).on("tick", ticked);
    simulation.force("link").links(graphData.links);
}

function ticked() {
    linkElements
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    // Update Edge Labels (Center of line)
    linkLabelElements
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2)
        .attr("transform", d => {
            // Optional: Rotate text to align with line
            // For now, keep horizontal for readability
            return "";
        })
        .attr("opacity", transformState.k > 1.2 ? 1 : 0); // Hide when zoomed out

    nodeElements
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

    const k = transformState.k;
    textElements
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("opacity", k > 1.2 ? 1 : 0);
}

// --- Drag ---
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    hideTooltip();
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// --- Interactions ---

function handleNodeClick(event, d) {
    if (event.shiftKey) {
        if (selectedNodes.has(d.id)) {
            selectedNodes.delete(d.id);
        } else {
            selectedNodes.add(d.id);
        }
        updateSelectionVisuals();
        return;
    }

    selectedNodes.clear();
    selectedNodes.add(d.id);
    updateSelectionVisuals();

    showNodeDetails(d);
}

function handleEdgeClick(event, d) {
    selectedNodes.clear();
    updateSelectionVisuals();
    showEdgeDetails(d);
}

function clearSelection() {
    selectedNodes.clear();
    updateSelectionVisuals();
    closeDetails();
}

function updateSelectionVisuals() {
    if (!nodeElements) return;
    nodeElements
        .attr("stroke", d => selectedNodes.has(d.id) ? "#fff" : "#121212")
        .attr("stroke-width", d => selectedNodes.has(d.id) ? 2.5 : 1.5)
        .attr("r", d => selectedNodes.has(d.id) ? 8 : 6);
}

function showNodeDetails(d) {
    const panel = document.getElementById('node-details');
    panel.classList.remove('translate-x-[110%]');
    document.getElementById('expand-button-container').classList.remove('hidden');

    document.getElementById('detail-title').innerText = d.properties.name || d.id;
    const content = document.getElementById('detail-content');
    const type = d.properties.ontology_class || d.labels.join(', ');

    let html = `<div class="mb-4"><span class="px-2 py-1 rounded bg-[#333537] text-[#a8c7fa] text-xs font-bold uppercase tracking-wider">${type}</span></div>`;

    Object.entries(d.properties).forEach(([k, v]) => {
        if (!['embedding', 'embeddings'].includes(k.toLowerCase())) {
            html += `
                <div class="mb-3 border-b border-[#444746]/50 pb-2 last:border-0">
                    <div class="text-[#8e918f] text-[10px] uppercase font-bold mb-1">${k}</div>
                    <div class="text-[#e3e3e3] text-sm break-words">${typeof v === 'object' ? JSON.stringify(v) : v}</div>
                </div>`;
        }
    });
    content.innerHTML = html;
    window._currentNodeId = d.id;
}

function showEdgeDetails(d) {
    const panel = document.getElementById('node-details');
    panel.classList.remove('translate-x-[110%]');
    document.getElementById('expand-button-container').classList.add('hidden');

    document.getElementById('detail-title').innerText = d.type || 'Relationship';
    const content = document.getElementById('detail-content');

    let html = `
        <div class="mb-4 flex flex-col gap-2 p-2 bg-[#28292a] rounded border border-[#444746]">
            <div class="flex flex-col">
                <span class="text-[9px] uppercase font-bold text-[#8e918f]">Source</span>
                <span class="text-xs text-[#a8c7fa] truncate">${d.source.properties?.name || d.source.id}</span>
            </div>
            <div class="flex items-center justify-center -my-1">
                <span class="material-icons-round text-xs text-[#444746]">arrow_downward</span>
            </div>
            <div class="flex flex-col">
                <span class="text-[9px] uppercase font-bold text-[#8e918f]">Target</span>
                <span class="text-xs text-[#a8c7fa] truncate">${d.target.properties?.name || d.target.id}</span>
            </div>
        </div>
    `;

    if (d.properties) {
        Object.entries(d.properties).forEach(([k, v]) => {
            if (!['embedding', 'embeddings'].includes(k.toLowerCase())) {
                html += `
                    <div class="mb-3 border-b border-[#444746]/50 pb-2 last:border-0">
                        <div class="text-[#8e918f] text-[10px] uppercase font-bold mb-1">${k}</div>
                        <div class="text-[#e3e3e3] text-sm break-words">${typeof v === 'object' ? JSON.stringify(v) : v}</div>
                    </div>`;
            }
        });
    }
    content.innerHTML = html;
}

function closeDetails() {
    document.getElementById('node-details').classList.add('translate-x-[110%]');
}

async function expandSelectedNode() {
    const nodeId = window._currentNodeId;
    if (!nodeId) return;
    try {
        const res = await fetch(`/api/graph/node/${encodeURIComponent(nodeId)}/expand`);
        const data = await res.json();
        if (data.error) {
            alert("Failed to expand: " + data.error);
            return;
        }
        updateGraphData(data);
    } catch (e) {
        console.error("Expand request failed", e);
    }
}

// Tooltip
function showTooltip(event, d) {
    const tooltip = document.getElementById('graph-tooltip');
    tooltip.style.opacity = 1;
    tooltip.innerHTML = `
        <div class="font-bold text-[#e3e3e3] mb-1">${d.properties.name || d.id}</div>
        <div class="text-[#a8c7fa] text-xs">${d.labels.join(', ')}</div>
    `;
    const x = event.pageX + 10;
    const y = event.pageY + 10;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

function hideTooltip() {
    document.getElementById('graph-tooltip').style.opacity = 0;
}



async function fetchSampleGraph() {
    try {
        // Use updated sample endpoint logic without manual limit if possible, or limit 50
        const res = await fetch(`/api/graph/sample?limit=50`);
        const data = await res.json();
        updateGraphData(data);
    } catch (e) {
        console.error("Failed to load sample", e);
    }
}
>>>>>>> 33d176646d75b1ad20790a86705c13c6a898a3f4:dashboard/static/script.js
