const AGENT_API = "http://localhost:8000";
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
    fetchSampleGraph(); 
    fetchNodeTypes(); 
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

async function fetchNodeTypes() {
    try {
        const res = await fetch(`${MGMT_API}/api/graph/labels`);
        const data = await res.json();
        
        const container = document.getElementById('type-filters');
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
        document.getElementById('type-filters').innerHTML = '<div class="text-xs text-red-400">Error loading types</div>';
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
        
        // Execution Time
        if (data.execution_time) {
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
        
        if (data.full_prompt) {
             const promptId = `prompt-${Date.now()}`;
             html += `
                <div class="mt-2">
                    <button onclick="document.getElementById('${promptId}').classList.toggle('hidden')" class="text-[10px] text-[#a8c7fa] hover:text-[#d6e3ff] flex items-center gap-1 cursor-pointer">
                        <span class="material-icons-round text-[12px]">code</span> View Prompt Context
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
        .attr("stroke", "#444746");
    
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
    
    document.getElementById('detail-title').innerText = d.properties.name || d.id;
    const content = document.getElementById('detail-content');
    const type = d.properties.ontology_class || d.labels.join(', ');
    
    let html = `<div class="mb-4"><span class="px-2 py-1 rounded bg-[#333537] text-[#a8c7fa] text-xs font-bold uppercase tracking-wider">${type}</span></div>`;
    
    Object.entries(d.properties).forEach(([k, v]) => {
        if (k !== 'name' && k !== 'id' && k !== 'ontology_class') {
            html += `
                <div class="mb-3 border-b border-[#444746]/50 pb-2 last:border-0">
                    <div class="text-[#8e918f] text-[10px] uppercase font-bold mb-1">${k}</div>
                    <div class="text-[#e3e3e3] text-sm break-words">${v}</div>
                </div>`;
        }
    });
    content.innerHTML = html;
    window._currentNodeId = d.id;
}

function closeDetails() {
    document.getElementById('node-details').classList.add('translate-x-[110%]');
}

async function expandSelectedNode() {
    const nodeId = window._currentNodeId; 
    if (!nodeId) return;
    try {
        const res = await fetch(`${MGMT_API}/api/graph/node/${encodeURIComponent(nodeId)}/expand`);
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

// Search
async function searchNodes() {
    const q = document.getElementById('graph-search').value;
    if (!q) return;
    
    const res = await fetch(`${MGMT_API}/api/graph/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '';
    resultsDiv.classList.remove('hidden');
    resultsDiv.classList.add('flex');
    
    data.nodes.forEach(node => {
        const btn = document.createElement('button');
        btn.className = "text-left px-4 py-3 text-sm text-[#e3e3e3] hover:bg-[#333537] border-b border-[#444746] last:border-0 w-full flex items-center gap-3 transition-colors";
        const label = node.properties.name || node.id;
        const color = getColor(node);
        
        btn.innerHTML = `<span class="w-3 h-3 rounded-full shrink-0" style="background-color: ${color}"></span><div class="flex-1 truncate"><span class="font-medium">${label}</span></div>`;
        
        btn.onclick = () => {
            const existing = graphData.nodes.find(n => n.id === node.id);
            if (existing) {
                selectedNodes.clear();
                selectedNodes.add(existing.id);
                updateSelectionVisuals();
                showNodeDetails(existing);
            } else {
                window._currentNodeId = node.id;
                expandSelectedNode();
            }
            resultsDiv.classList.add('hidden');
            document.getElementById('graph-search').value = '';
        };
        resultsDiv.appendChild(btn);
    });
}

async function fetchSampleGraph() {
    try {
        // Use updated sample endpoint logic without manual limit if possible, or limit 50
        const res = await fetch(`${MGMT_API}/api/graph/sample?limit=50`);
        const data = await res.json();
        updateGraphData(data);
    } catch (e) {
        console.error("Failed to load sample", e);
    }
}
