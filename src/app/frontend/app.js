document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const graphContainer = document.getElementById('graph-container');
    const themeToggle = document.getElementById('theme-toggle');

    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'graph-tooltip';
    document.body.appendChild(tooltip);

    // Tooltip interaction state
    let tooltipTimeout = null;
    let isTooltipHovered = false;

    tooltip.addEventListener('mouseenter', () => {
        isTooltipHovered = true;
        if (tooltipTimeout) {
            clearTimeout(tooltipTimeout);
            tooltipTimeout = null;
        }
    });

    tooltip.addEventListener('mouseleave', () => {
        isTooltipHovered = false;
        tooltip.style.opacity = 0;
        // Move off-screen or hide effectively to prevent blocking clicks
        // (opacity transition handles fade, but pointer-events: auto means it blocks)
        // We'll rely on opacity=0 and maybe pointer-events manipulation if needed, 
        // but typically just opacity 0 with z-index is fine unless it traps clicks.
        // Actually, better to set display none or move it after transition?
        // Simple opacity is usually okay for small tooltips.
    });

    // Delegate click for read-more links
    tooltip.addEventListener('click', (e) => {
        if (e.target.classList.contains('read-more-link')) {
            e.preventDefault();
            const parent = e.target.parentElement;
            const short = parent.querySelector('.short-text');
            const full = parent.querySelector('.full-text');

            if (e.target.dataset.expanded === 'true') {
                short.style.display = 'inline';
                full.style.display = 'none';
                e.target.textContent = 'Read more';
                e.target.dataset.expanded = 'false';
            } else {
                short.style.display = 'none';
                full.style.display = 'inline';
                e.target.textContent = 'Read less';
                e.target.dataset.expanded = 'true';
            }
        }
    });

    // Theme Toggle
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        themeToggle.textContent = isDark ? '☀️' : '🌙';
    });

    // Load Schema
    let schema = {};
    async function loadSchema() {
        try {
            const response = await fetch('/schema');
            if (response.ok) {
                schema = await response.json();
                console.log("Schema loaded:", schema);
            }
        } catch (e) {
            console.error("Failed to load schema", e);
        }
    }
    loadSchema();

    // State for interactive legend
    let currentGraphData = null;
    let accumulatedGraphData = null;  // Accumulated graph data across the conversation
    const activeLayers = {};

    // Color map for node types
    const colorMap = {
        'Speaker': '#FFC107',
        'Speech': '#03A9F4',
        'Chunk': '#E0E0E0',
        'Community': '#8BC34A',
        'SubCommunity': '#CDDC39',
        'Topic': '#FF5722',
        'Subtopic': '#FF9800',
        'Entity': '#9C27B0',
        'Session': '#607D8B',
        'Document': '#795548',
        'Day': '#00BCD4',
        'Segment': '#3F51B5',
        'Unknown': '#9E9E9E'
    };

    // CSS class map for legend dots
    const dotClassMap = {
        'Speaker': 'speaker',
        'Speech': 'speech',
        'Chunk': 'chunk',
        'Community': 'community',
        'SubCommunity': 'subcommunity',
        'Subtopic': 'subtopic',
        'Topic': 'topic',
        'Entity': 'entity',
        'Session': 'session',
        'Document': 'document',
        'Day': 'day',
        'Segment': 'segment',
        'Unknown': 'unknown'
    };

    // Function to update legend dynamically based on graph data
    function updateLegend(graphData) {
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            return;
        }

        // Get unique node types from the graph
        const nodeTypes = new Set();
        graphData.nodes.forEach(node => {
            const type = getNodeType(node);
            if (type && type !== 'Unknown') {
                nodeTypes.add(type);
            }
        });

        // Initialize activeLayers for all found node types
        nodeTypes.forEach(type => {
            if (!(type in activeLayers)) {
                activeLayers[type] = true;
            }
        });

        // Generate legend HTML
        const legendDiv = document.getElementById('graph-legend');
        legendDiv.innerHTML = '';

        // Sort node types for consistent display
        const sortedTypes = Array.from(nodeTypes).sort();

        sortedTypes.forEach(type => {
            const dotClass = dotClassMap[type] || 'unknown';
            const legendItem = document.createElement('span');
            legendItem.className = 'legend-item';
            // Add 'active' class if the layer is active
            if (activeLayers[type] !== false) {
                legendItem.classList.add('active');
            }
            legendItem.innerHTML = `<span class="dot ${dotClass}"></span> ${type}`;
            legendDiv.appendChild(legendItem);
        });

        // Re-attach event listeners to legend items
        attachLegendListeners();
    }

    // Helper to determine node type (needs to be accessible outside renderGraph)
    function getNodeType(node) {
        const labels = node.labels || [];
        const props = node.properties || {};
        // Convert labels to lowercase for case-insensitive matching
        const labelsLower = labels.map(l => l.toLowerCase());

        // Check for Topic nodes - they might be labeled as Entity but have title/summary/community_id
        const isTopic = labelsLower.some(l => l === 'topic' || l === 'topics') ||
            props.node_type === 'TOPIC' ||
            (props.title && props.summary && props.community_id && !props.name);

        // Check for Subtopic nodes
        const isSubtopic = labelsLower.some(l => l === 'subtopic' || l === 'subtopics') ||
            (props.title && props.summary && (props.subtopic_local_id || props.parent_topic) && !props.name);

        if (labelsLower.some(l => l === 'speaker' || l === 'speakers')) return 'Speaker';
        if (labelsLower.some(l => l === 'community' || l === 'communities')) return 'Community';
        if (labelsLower.some(l => l === 'subcommunity' || l === 'subcommunities')) return 'SubCommunity';
        if (isSubtopic) return 'Subtopic';
        if (labelsLower.some(l => l === 'speech' || l === 'speeches')) return 'Speech';
        if (labelsLower.some(l => l === 'chunk' || l === 'chunks')) return 'Chunk';
        if (labelsLower.some(l => l === 'session' || l === 'sessions')) return 'Session';
        if (labelsLower.some(l => l === 'document' || l === 'documents')) return 'Document';
        if (labelsLower.some(l => l === 'day')) return 'Day';
        if (labelsLower.some(l => l === 'segment' || l === 'segments')) return 'Segment';
        if (isTopic) return 'Topic';
        if (labelsLower.some(l => l === 'entity' || l === 'entity_concept' || l === 'entities')) return 'Entity';
        return 'Unknown';
    }

    // Function to attach event listeners to legend items
    function attachLegendListeners() {
        document.querySelectorAll('.legend-item').forEach(item => {
            // Remove any existing listeners by cloning
            const newItem = item.cloneNode(true);
            item.parentNode.replaceChild(newItem, item);

            newItem.addEventListener('click', () => {
                // Extract the node type from the text (after the dot)
                const text = newItem.textContent.trim();
                // The format is "NodeType" - extract just the type name
                const type = text.replace(/^\S+\s+/, '').trim();

                if (activeLayers.hasOwnProperty(type)) {
                    activeLayers[type] = !activeLayers[type];

                    // Toggle visual classes
                    if (activeLayers[type]) {
                        newItem.classList.remove('inactive');
                        newItem.classList.add('active');
                    } else {
                        newItem.classList.remove('active');
                        newItem.classList.add('inactive');
                    }

                    // Re-render graph with current data
                    if (currentGraphData) {
                        renderGraph(currentGraphData, true);
                    }
                }
            });
        });
    }

    // Initialize D3 selection
    let svg = null;
    let simulation = null;

    function renderGraph(graphData, isUpdate = false) {
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            return;
        }

        // Update legend dynamically based on graph data
        updateLegend(graphData);

        if (!isUpdate) {
            currentGraphData = graphData;
        }

        // Clear previous graph
        graphContainer.innerHTML = '';

        // Dimensions
        const width = graphContainer.clientWidth;
        const height = graphContainer.clientHeight;

        // colorMap is now defined at the top level

        // Create SVG
        svg = d3.select(graphContainer).append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            }));

        const g = svg.append("g");

        // getNodeType is now defined at the top level

        // Process Nodes & Links for D3
        // Filter nodes based on active layers and map to new objects
        const nodes = graphData.nodes
            .filter(node => {
                const type = getNodeType(node);
                return activeLayers[type] !== false;
            })
            .map(d => ({ ...d }));

        // Ensure nodes have element_id, fallback to id if missing (though edges rely on element_id)
        nodes.forEach(n => {
            if (!n.element_id) {
                // Fallback attempt
                n.element_id = n.id;
            }
        });

        // Create a set of available node IDs for fast lookup
        const nodeIds = new Set(nodes.map(d => d.element_id));

        // Filter links to ensure both source and target exist in nodes
        // Include all edge types (entity_relation, HAS_ENTITY, HAS_CHUNK, HAS_SPEECH, IN_COMMUNITY, IN_SUBCOMMUNITY)
        const links = graphData.edges
            .map(d => ({ ...d, source: d.start, target: d.end }))
            .filter(d => {
                const sourceExists = nodeIds.has(d.source);
                const targetExists = nodeIds.has(d.target);
                return sourceExists && targetExists;
            });

        // Force Simulation
        simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.element_id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(35));

        // Arrow Marker
        svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25) // Adjust based on node radius
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");

        // Links with different colors based on edge type
        const link = g.append("g")
            .attr("stroke-opacity", 0.6)
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke-width", d => {
                // Thicker for entity_relation, thinner for structural
                const edgeType = d.type || '';
                const graphType = d.properties?.graph_type || '';
                if (graphType === 'entity_relation') return 2;
                if (edgeType === 'IN_COMMUNITY' || edgeType === 'IN_SUBCOMMUNITY') return 2;
                return 1.5;
            })
            .attr("stroke", d => {
                // Color code by edge type
                const edgeType = d.type || '';
                const graphType = d.properties?.graph_type || '';
                const label = d.properties?.label || '';

                // Get target node to check if it's a topic
                let targetNode = null;
                if (typeof d.target === 'object' && d.target.element_id) {
                    targetNode = d.target;
                } else {
                    const targetId = typeof d.target === 'object' ? d.target.element_id || d.target.id : d.target;
                    targetNode = nodes.find(n => n.element_id === targetId);
                }
                const isTargetTopic = targetNode && getNodeType(targetNode) === 'Topic';

                // Check if this is a Discussing_topic edge (IN_COMMUNITY connecting to Topic nodes)
                const isDiscussingTopic = (edgeType === 'IN_COMMUNITY' || label === 'IN_COMMUNITY') && isTargetTopic;
                const isSubtopicOf = edgeType === 'Subtopic_of' || edgeType === 'PARENT_COMMUNITY' ||
                    label === 'Subtopic_of' || label === 'PARENT_COMMUNITY';

                if (graphType === 'entity_relation') return '#9C27B0'; // Purple for entity relations
                if (isDiscussingTopic) return '#FF5722'; // Deep orange for discussing topic
                if (isSubtopicOf) return '#795548'; // Brown for subtopic relationships
                if (edgeType === 'HAS_ENTITY') return '#4CAF50'; // Green for chunk-entity
                if (edgeType === 'HAS_CHUNK') return '#2196F3'; // Blue for speech-chunk
                if (edgeType === 'HAS_SPEECH') return '#FF9800'; // Orange for speaker-speech
                if (edgeType === 'IN_COMMUNITY') return '#8BC34A'; // Light green for community (non-topic)
                if (edgeType === 'IN_SUBCOMMUNITY') return '#CDDC39'; // Lime for subcommunity
                return '#999'; // Default gray
            })
            .attr("marker-end", "url(#arrow)");

        // Hover interaction for links
        link.on("mouseover", (event, d) => {
            // Get the display label (same logic as edge labels)
            const edgeType = d.type || '';
            const label = d.properties?.label || '';
            let targetNode = null;
            if (typeof d.target === 'object' && d.target.element_id) {
                targetNode = d.target;
            } else {
                const targetId = typeof d.target === 'object' ? d.target.element_id || d.target.id : d.target;
                targetNode = nodes.find(n => n.element_id === targetId);
            }
            const isTargetTopic = targetNode && getNodeType(targetNode) === 'Topic';
            let displayLabel = label || d.properties?.relation_type || edgeType || '';
            if (isTargetTopic && (edgeType === 'IN_COMMUNITY' || label === 'IN_COMMUNITY')) {
                displayLabel = 'Discussing_topic';
            } else if (edgeType === 'PARENT_COMMUNITY' || label === 'PARENT_COMMUNITY') {
                displayLabel = 'Subtopic_of';
            }

            tooltip.style.opacity = 1;
            tooltip.innerHTML = `<strong>Relationship:</strong> ${displayLabel}`;
        })
            .on("mousemove", (event) => {
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY + 10) + 'px';
            })
            .on("mouseout", () => {
                tooltip.style.opacity = 0;
            });

        // Edge Labels (Static - show for entity_relation and community edges)
        const linkLabel = g.append("g")
            .attr("class", "link-labels")
            .selectAll("text")
            .data(links)
            .join("text")
            .attr("text-anchor", "middle")
            .attr("dy", -5)
            .attr("font-size", "8px")
            .attr("fill", "#666")
            .style("display", d => {
                // Show labels for entity_relation, topic, and community edges
                const edgeType = d.type || '';
                const graphType = d.properties?.graph_type || '';
                const label = d.properties?.label || '';

                const isDiscussingTopic = edgeType === 'IN_COMMUNITY' || edgeType === 'Discussing_topic' || label === 'Discussing_topic';
                const isSubtopicOf = edgeType === 'Subtopic_of' || edgeType === 'PARENT_COMMUNITY' || label === 'Subtopic_of';

                if (graphType === 'entity_relation') return 'block';
                if (isDiscussingTopic || isSubtopicOf) return 'block';
                if (edgeType === 'IN_SUBCOMMUNITY') return 'block';
                return 'none'; // Hide labels for HAS_ENTITY, HAS_CHUNK, HAS_SPEECH to reduce clutter
            })
            .text(d => {
                // Use relation type from properties if available, otherwise use edge type
                // Rename IN_COMMUNITY to Discussing_topic for topic nodes
                const edgeType = d.type || '';
                const label = d.properties?.label || '';
                const graphType = d.properties?.graph_type || '';

                // Get target node - handle both object and ID references
                let targetNode = null;
                if (typeof d.target === 'object' && d.target.element_id) {
                    targetNode = d.target;
                } else {
                    const targetId = typeof d.target === 'object' ? d.target.element_id || d.target.id : d.target;
                    targetNode = nodes.find(n => n.element_id === targetId);
                }

                const isTargetTopic = targetNode && getNodeType(targetNode) === 'Topic';

                // Rename IN_COMMUNITY to Discussing_topic when connecting to Topic nodes
                if (isTargetTopic && (edgeType === 'IN_COMMUNITY' || label === 'IN_COMMUNITY')) {
                    return 'Discussing_topic';
                }

                // Check for subtopic relationship (PARENT_COMMUNITY represents Subtopic_of)
                if (edgeType === 'PARENT_COMMUNITY' || label === 'PARENT_COMMUNITY') {
                    return 'Subtopic_of';
                }

                return label || d.properties?.relation_type || edgeType || '';
            });

        // Nodes
        const node = g.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .selectAll("g")
            .data(nodes)
            .join("g")
            .call(drag(simulation));

        // Node Circles
        node.append("circle")
            .attr("r", d => {
                const labels = d.labels || [];
                if (labels.includes('Community')) return 15;
                if (labels.includes('Chunk')) return 8;
                return 10;
            })
            .attr("fill", d => {
                // Use getNodeType helper for consistent case-insensitive matching
                const group = getNodeType(d);
                return colorMap[group] || colorMap['Unknown'];
            });

        // Node Labels (Static)
        const label = g.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("dy", -15)
            .attr("text-anchor", "middle")
            .text(d => {
                const props = d.properties || {};
                let name = props.name || props.title || props.speaker_name || d.id;
                if (d.labels && d.labels.includes('Chunk')) return "";
                return name && name.length > 20 ? name.substring(0, 17) + "..." : name;
            });

        // Click to Expand Node Connections
        node.on("click", async (event, d) => {
            event.stopPropagation();
            try {
                const response = await fetch(`/node-connections/${encodeURIComponent(d.element_id)}?database=falkordb`);
                if (!response.ok) return;
                
                const data = await response.json();
                if (!data.nodes || data.nodes.length === 0) return;
                
                // Merge new data with current graph
                const existingNodeIds = new Set(currentGraphData.nodes.map(n => n.element_id));
                const existingEdgeKeys = new Set(currentGraphData.edges.map(e => `${e.start}-${e.type}-${e.end}`));
                
                data.nodes.forEach(node => {
                    if (!existingNodeIds.has(node.element_id)) {
                        currentGraphData.nodes.push(node);
                        existingNodeIds.add(node.element_id);
                    }
                });
                
                data.edges.forEach(edge => {
                    const key = `${edge.start}-${edge.type}-${edge.end}`;
                    if (!existingEdgeKeys.has(key)) {
                        currentGraphData.edges.push(edge);
                        existingEdgeKeys.add(key);
                    }
                });
                
                // Update accumulated graph data
                accumulatedGraphData = currentGraphData;
                
                // Re-render graph
                renderGraph(currentGraphData, true);
            } catch (e) {
                console.error("Failed to fetch node connections:", e);
            }
        });

        // Tooltip Interaction for Nodes
        node.on("mouseover", (event, d) => {
            // Cancel any pending hide
            if (tooltipTimeout) {
                clearTimeout(tooltipTimeout);
                tooltipTimeout = null;
            }

            const props = d.properties || {};
            const labels = d.labels || [];
            // Use getNodeType for consistent case-insensitive matching
            const nodeType = getNodeType(d);
            let group = nodeType;

            // Map to schema type (uppercase for schema lookup)
            let schemaType = null;
            const labelsLower = labels.map(l => l.toLowerCase());
            if (labelsLower.some(l => l === 'speaker' || l === 'speakers')) schemaType = 'SPEAKER';
            else if (labelsLower.some(l => l === 'speech' || l === 'speeches')) schemaType = 'SPEECH';
            else if (labelsLower.some(l => l === 'chunk' || l === 'chunks')) schemaType = 'CHUNK';
            else if (labelsLower.some(l => l === 'community' || l === 'communities')) schemaType = 'COMMUNITY';
            else if (labelsLower.some(l => l === 'subcommunity' || l === 'subcommunities')) schemaType = 'SUBCOMMUNITY';
            else if (labelsLower.some(l => l === 'topic' || l === 'topics') || props.node_type === 'TOPIC' || props.topic_name !== undefined) schemaType = 'TOPIC';
            else if (labelsLower.some(l => l === 'entity' || l === 'entity_concept' || l === 'entities')) schemaType = 'ENTITY_CONCEPT';
            else if (labelsLower.some(l => l === 'session' || l === 'sessions')) schemaType = 'SESSION';
            else if (labelsLower.some(l => l === 'day')) schemaType = 'DAY';
            else if (labelsLower.some(l => l === 'segment' || l === 'segments')) schemaType = 'SEGMENT';

            let title = props.name || props.title || props.speaker_name || d.id;
            if (nodeType === 'Chunk' && props.text) title = props.text.substring(0, 50) + '...';

            let html = `<h4>${group}: ${title}</h4>`;

            // Helper to format value for display
            const formatValue = (v) => {
                if (v === null || v === undefined) return 'N/A';
                if (typeof v === 'number') {
                    // Format numbers with appropriate precision
                    if (Number.isInteger(v)) return v.toString();
                    if (Math.abs(v) < 0.01 || Math.abs(v) > 1000) {
                        return v.toExponential(3);
                    }
                    return v.toFixed(4);
                }
                if (typeof v === 'boolean') return v ? 'Yes' : 'No';
                if (typeof v === 'object') {
                    try {
                        return JSON.stringify(v, null, 2);
                    } catch {
                        return String(v);
                    }
                }
                return String(v);
            };

            // Helper to add row
            const addRow = (k, v) => {
                if (v === null || v === undefined) return;

                let content = formatValue(v);

                // Handle long strings with expand/collapse
                if (typeof v === 'string' && v.length > 100) {
                    const shortText = v.substring(0, 100) + '...';
                    content = `
                        <span class="short-text">${shortText}</span>
                        <span class="full-text" style="display:none">${v}</span>
                        <span class="read-more-link" data-expanded="false">Read more</span>
                    `;
                }

                // Format key name (convert snake_case to Title Case)
                const formattedKey = k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                html += `<div><strong>${formattedKey}:</strong> ${content}</div>`;
            };

            // Properties are already filtered in the backend, so we just display them
            // Properties to prioritize/show prominently (for better ordering)
            const priorityKeys = {
                'SPEAKER': ['speaker_name', 'first_name', 'last_name', 'name', 'party', 'Speaker_party_name',
                    'Speaker_role', 'Speaker_gender', 'Speaker_birth', 'Topic'],
                'SPEECH': ['content', 'date', 'document_date', 'sentiment', 'topic', 'Topic', 'content_length',
                    'local_speech_order', 'global_speech_order', 'line_number', 'name'],
                'CHUNK': ['text', 'length', 'name', 'llama_metadata'],
                'COMMUNITY': ['title', 'summary', 'community_id', 'name'],
                'SUBCOMMUNITY': ['title', 'summary', 'community_id', 'subtopic_local_id', 'parent_topic', 'name'],
                'TOPIC': ['title', 'summary', 'community_id', 'name'],
                'SUBTOPIC': ['title', 'summary', 'community_id', 'subtopic_local_id', 'parent_topic', 'name'],
                'ENTITY_CONCEPT': ['name', 'entity_type', 'centrality_summary', 'centrality_high_measures',
                    'centrality_low_measures', 'centrality_average_measures',
                    'pagerank_centrality', 'betweenness_centrality', 'degree_centrality',
                    'closeness_centrality', 'eigenvector_centrality', 'harmonic_centrality',
                    'load_centrality',
                    'pagerank_centrality_description', 'betweenness_centrality_description',
                    'degree_centrality_description', 'closeness_centrality_description',
                    'eigenvector_centrality_description', 'harmonic_centrality_description',
                    'load_centrality_description'],
                'SESSION': ['filename', 'date', 'speech_count', 'Term', 'name'],
                'DAY': ['date', 'name', 'segment_count'],
                'SEGMENT': ['content', 'content_length', 'line_number', 'document_date', 'date', 
                    'local_segment_order', 'global_segment_order', 'name']
            };

            // Get priority keys for this node type
            const nodePriorityKeys = priorityKeys[schemaType] || [];

            // First, show priority properties in order
            if (nodePriorityKeys.length > 0) {
                nodePriorityKeys.forEach(key => {
                    if (props[key] !== undefined && props[key] !== null) {
                        addRow(key, props[key]);
                    }
                });
            }

            // Then show other filtered properties (excluding already shown)
            const shownKeys = new Set(nodePriorityKeys);
            Object.entries(props).forEach(([k, v]) => {
                if (v !== undefined && v !== null && !shownKeys.has(k)) {
                    addRow(k, v);
                }
            });

            tooltip.innerHTML = html;
            tooltip.style.opacity = 1;

            // Set position only on mouseover (entry), allowing user to move to tooltip
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
        })
            .on("mouseout", () => {
                // Delay hiding to allow moving to tooltip
                tooltipTimeout = setTimeout(() => {
                    if (!isTooltipHovered) {
                        tooltip.style.opacity = 0;
                    }
                }, 300);
            });

        // Remove native title if present
        node.select("title").remove();

        // Simulation Tick
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);

            node
                .attr("transform", d => `translate(${d.x},${d.y})`);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });
    }

    // Drag behavior
    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }

    function createThinkingElement(reasoningChain, contextText) {
        if ((!reasoningChain || reasoningChain.length === 0) && !contextText) return null;

        const container = document.createElement('div');
        container.className = 'thinking-process';

        const header = document.createElement('div');
        header.className = 'thinking-header';
        const stepsCount = reasoningChain ? reasoningChain.length : 0;
        header.innerHTML = `<span>🤔 Debug Info (${stepsCount} steps)</span> <span>▼</span>`;

        const body = document.createElement('div');
        body.className = 'thinking-body';

        if (reasoningChain) {
            reasoningChain.forEach(step => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'thinking-step';
                stepDiv.textContent = step;
                body.appendChild(stepDiv);
            });
        }

        if (contextText) {
            const contextDiv = document.createElement('div');
            contextDiv.className = 'context-block';
            contextDiv.textContent = "=== LLM Context ===\n" + contextText;
            body.appendChild(contextDiv);
        }

        header.addEventListener('click', () => {
            container.classList.toggle('open');
            header.querySelector('span:last-child').textContent = container.classList.contains('open') ? '▲' : '▼';
        });

        container.appendChild(header);
        container.appendChild(body);
        return container;
    }

    function addMessage(content, isUser, reasoningChain = null, context = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

        if (isUser) {
            msgDiv.textContent = content;
        } else {
            // Add thinking process if available
            if (reasoningChain || context) {
                const thinkingEl = createThinkingElement(reasoningChain, context);
                if (thinkingEl) msgDiv.appendChild(thinkingEl);
            }

            const contentDiv = document.createElement('div');
            contentDiv.className = 'bot-content';
            contentDiv.textContent = content;
            msgDiv.appendChild(contentDiv);
        }

        messagesDiv.appendChild(msgDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    async function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;

        addMessage(query, true);
        queryInput.value = '';
        queryInput.disabled = true;
        sendBtn.disabled = true;

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.textContent = 'Thinking...';
        messagesDiv.appendChild(loadingDiv);

        try {
            // Prepare request body with accumulated graph data
            const requestBody = {
                query: query,
                accumulated_graph_data: accumulatedGraphData
            };

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (loadingDiv.parentNode) messagesDiv.removeChild(loadingDiv);

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // Update accumulated graph data with the response
            if (data.graph_data) {
                accumulatedGraphData = data.graph_data;
                renderGraph(data.graph_data);
            }

            // Add message with reasoning and context
            addMessage(data.answer, false, data.reasoning_chain, data.context);

        } catch (error) {
            if (loadingDiv.parentNode) messagesDiv.removeChild(loadingDiv);
            addMessage(`Error: ${error.message}`, false);
        } finally {
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});




