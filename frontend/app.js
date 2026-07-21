const canvas = document.querySelector("#graphCanvas");
const ctx = canvas.getContext("2d");
const emptyState = document.querySelector("#emptyState");
const namespaceSelect = document.querySelector("#namespaceSelect");
const namespaceCards = document.querySelector("#namespaceCards");
const nsRefreshButton = document.querySelector("#nsRefreshButton");
const sessionInput = document.querySelector("#sessionInput");
const sessionsList = document.querySelector("#sessions");
const limitInput = document.querySelector("#limitInput");
const refreshButton = document.querySelector("#refreshButton");
const flushButton = document.querySelector("#flushButton");
const searchForm = document.querySelector("#searchForm");
const queryInput = document.querySelector("#queryInput");
const resultsEl = document.querySelector("#results");
const healthEl = document.querySelector("#health");
const nodeCountEl = document.querySelector("#nodeCount");
const edgeCountEl = document.querySelector("#edgeCount");
const stmChunksEl = document.querySelector("#stmChunks");
const ltmTopicsEl = document.querySelector("#ltmTopics");
const selectionEl = document.querySelector("#selection");
const dropZone = document.querySelector("#dropZone");
const ingestStatus = document.querySelector("#ingestStatus");
const flushStatsSection = document.querySelector("#flushStatsSection");
const flushStatsGrid = document.querySelector("#flushStatsGrid");
const flushErrors = document.querySelector("#flushErrors");
const opLogEl = document.querySelector("#opLog");

const palette = ["#56c5b6", "#8ea7ff", "#f2c86d", "#f18f9b", "#7bd88f", "#c993ff", "#79c7ff", "#f5a96f"];
const state = {
  nodes: [],
  links: [],
  nodeById: new Map(),
  highlighted: new Set(),
  selected: null,
  width: 0,
  height: 0,
  dpr: 1,
  panX: 0,
  panY: 0,
  zoom: 1,
  draggingNode: null,
  draggingCanvas: false,
  pointer: null,
  animating: false,
};

// â”€â”€ Op log (last 5 entries) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const opLog = [];

function pushOpLog(type, summary, ok = true) {
  const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  opLog.unshift({ ts, type, summary, ok });
  if (opLog.length > 5) opLog.pop();
  renderOpLog();
}

function renderOpLog() {
  opLogEl.innerHTML = "";
  for (const entry of opLog) {
    const li = document.createElement("li");
    li.className = `op-entry ${entry.ok ? "op-ok" : "op-err"}`;
    li.innerHTML = `<span class="op-ts">${escapeHtml(entry.ts)}</span><span class="op-type">${escapeHtml(entry.type)}</span><span class="op-summary">${escapeHtml(entry.summary)}</span>`;
    opLogEl.append(li);
  }
}

// â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function currentScope() {
  return document.querySelector("input[name='scope']:checked").value;
}

function currentNamespace() {
  return namespaceSelect.value || "";
}

function nsLabel(namespace) {
  return namespace || "default";
}

function setText(el, value) {
  el.textContent = value ?? "0";
}

function setHealth(ok, label) {
  healthEl.textContent = label;
  healthEl.classList.toggle("ok", ok);
  healthEl.classList.toggle("bad", !ok);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

async function fetchJson(url, init) {
  const response = await fetch(url, { headers: { Accept: "application/json" }, ...init });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function updateFlushButton() {
  const stm = currentScope() === "stm";
  const hasSession = sessionInput.value.trim().length > 0;
  flushButton.disabled = !(stm && hasSession);
}

// â”€â”€ Graph simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  state.dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.width = Math.max(1, rect.width);
  state.height = Math.max(1, rect.height);
  canvas.width = Math.floor(state.width * state.dpr);
  canvas.height = Math.floor(state.height * state.dpr);
  canvas.style.width = `${state.width}px`;
  canvas.style.height = `${state.height}px`;
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  draw();
}

function colorFor(node) {
  const key = String(node.topic || node.type || node.scope || "default");
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) >>> 0;
  }
  return palette[hash % palette.length];
}

function radiusFor(node) {
  const rank = Math.max(0, Number(node.pagerank || node.score || 1));
  return Math.max(6, Math.min(18, 7 + Math.sqrt(rank) * 3));
}

function prepareGraph(data) {
  const width = Math.max(state.width, 320);
  const height = Math.max(state.height, 320);
  const centerX = width / 2;
  const centerY = height / 2;
  const spread = Math.min(width, height) * 0.32;

  state.nodes = (data.nodes || []).map((node, index, all) => {
    const angle = (Math.PI * 2 * index) / Math.max(all.length, 1);
    return {
      ...node,
      x: centerX + Math.cos(angle) * spread * Math.random(),
      y: centerY + Math.sin(angle) * spread * Math.random(),
      vx: 0,
      vy: 0,
    };
  });
  state.nodeById = new Map(state.nodes.map((node) => [String(node.id), node]));
  state.links = (data.links || [])
    .map((link) => ({
      ...link,
      sourceNode: state.nodeById.get(String(link.source)),
      targetNode: state.nodeById.get(String(link.target)),
    }))
    .filter((link) => link.sourceNode && link.targetNode);
  state.selected = null;
  state.highlighted.clear();
  setText(nodeCountEl, data.node_count ?? state.nodes.length);
  setText(edgeCountEl, data.edge_count ?? state.links.length);
  emptyState.hidden = state.nodes.length > 0;
  fitGraph();
  startSimulation();
}

function fitGraph() {
  state.panX = 0;
  state.panY = 0;
  state.zoom = 1;
}

function tick() {
  const nodes = state.nodes;
  const links = state.links;
  const centerX = state.width / 2;
  const centerY = state.height / 2;

  for (let i = 0; i < nodes.length; i += 1) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j += 1) {
      const b = nodes[j];
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let dist2 = dx * dx + dy * dy;
      if (dist2 < 1) {
        dx = Math.random() - 0.5;
        dy = Math.random() - 0.5;
        dist2 = dx * dx + dy * dy;
      }
      const force = Math.min(900 / dist2, 0.8);
      a.vx -= dx * force;
      a.vy -= dy * force;
      b.vx += dx * force;
      b.vy += dy * force;
    }
  }

  for (const link of links) {
    const a = link.sourceNode;
    const b = link.targetNode;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distance = Math.max(1, Math.hypot(dx, dy));
    const target = 90 + Math.min(90, Number(link.weight || 1) * 4);
    const force = (distance - target) * 0.006;
    const fx = (dx / distance) * force;
    const fy = (dy / distance) * force;
    a.vx += fx;
    a.vy += fy;
    b.vx -= fx;
    b.vy -= fy;
  }

  for (const node of nodes) {
    if (state.draggingNode === node) {
      node.vx = 0;
      node.vy = 0;
      continue;
    }
    node.vx += (centerX - node.x) * 0.003;
    node.vy += (centerY - node.y) * 0.003;
    node.vx *= 0.86;
    node.vy *= 0.86;
    node.x += node.vx;
    node.y += node.vy;
  }
}

function startSimulation() {
  if (state.animating) return;
  state.animating = true;
  let frames = 0;
  function frame() {
    frames += 1;
    if (frames < 420 || state.draggingNode) {
      tick();
      draw();
      requestAnimationFrame(frame);
      return;
    }
    state.animating = false;
    draw();
  }
  requestAnimationFrame(frame);
}

function worldToScreen(x, y) {
  return { x: x * state.zoom + state.panX, y: y * state.zoom + state.panY };
}

function screenToWorld(x, y) {
  return { x: (x - state.panX) / state.zoom, y: (y - state.panY) / state.zoom };
}

function draw() {
  ctx.clearRect(0, 0, state.width, state.height);
  ctx.fillStyle = "#0d1015";
  ctx.fillRect(0, 0, state.width, state.height);

  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.zoom, state.zoom);

  for (const link of state.links) {
    const a = link.sourceNode;
    const b = link.targetNode;
    const highlighted = state.highlighted.has(String(a.id)) && state.highlighted.has(String(b.id));
    ctx.strokeStyle = highlighted ? "rgba(242, 200, 109, 0.9)" : "rgba(166, 175, 189, 0.22)";
    ctx.lineWidth = highlighted ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const node of state.nodes) {
    const radius = radiusFor(node);
    const isHighlighted = state.highlighted.has(String(node.id));
    const isSelected = state.selected && String(state.selected.id) === String(node.id);
    ctx.beginPath();
    ctx.fillStyle = colorFor(node);
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (isHighlighted || isSelected) {
      ctx.lineWidth = isSelected ? 4 : 3;
      ctx.strokeStyle = isSelected ? "#ffffff" : "#f2c86d";
      ctx.stroke();
    }
  }

  ctx.fillStyle = "rgba(243, 245, 248, 0.9)";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const node of state.nodes) {
    if (state.zoom < 0.55 && !state.highlighted.has(String(node.id))) continue;
    const label = String(node.label || node.id).slice(0, 28);
    ctx.fillText(label, node.x, node.y + radiusFor(node) + 5);
  }

  ctx.restore();
}

function pickNode(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const point = screenToWorld(clientX - rect.left, clientY - rect.top);
  for (let index = state.nodes.length - 1; index >= 0; index -= 1) {
    const node = state.nodes[index];
    if (Math.hypot(point.x - node.x, point.y - node.y) <= radiusFor(node) + 6) {
      return node;
    }
  }
  return null;
}

function showSelection(node) {
  state.selected = node;
  if (!node) {
    selectionEl.textContent = "Select a node";
    draw();
    return;
  }
  selectionEl.innerHTML = `<strong>${escapeHtml(node.label || node.id)}</strong><br>${escapeHtml(node.type || "Entity")} ${node.topic ? `â€” topic ${escapeHtml(node.topic)}` : ""}${node.pagerank ? `<br><small>PageRank: ${Number(node.pagerank).toFixed(4)}</small>` : ""}`;
  draw();
}

// â”€â”€ API calls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function loadHealth() {
  try {
    const data = await fetchJson("/api/health");
    const ok = Object.entries(data).filter(([key]) => key.endsWith("connected")).every(([, value]) => Boolean(value));
    setHealth(ok, ok ? "Online" : "Partial");
  } catch {
    setHealth(false, "Offline");
  }
}

async function loadStats() {
  try {
    const params = new URLSearchParams();
    if (sessionInput.value) params.set("session_id", sessionInput.value);
    if (currentNamespace()) params.set("namespace", currentNamespace());
    const data = await fetchJson(`/api/stats?${params}`);
    setText(stmChunksEl, data.stm?.chunks ?? data.stm?.chunk_count ?? 0);
    setText(ltmTopicsEl, data.ltm?.topics ?? data.ltm?.topic_count ?? 0);
  } catch {
    setText(stmChunksEl, "-");
    setText(ltmTopicsEl, "-");
  }
}

function metaCell(label, value) {
  const shown = value === null || value === undefined ? "—" : value;
  return `<span class="ns-metric"><b>${escapeHtml(String(shown))}</b> ${escapeHtml(label)}</span>`;
}

function renderNamespaceCards(entries, active) {
  namespaceCards.innerHTML = "";
  if (!entries.length) {
    namespaceCards.innerHTML = `<div class="ns-empty">No namespaces found</div>`;
    return;
  }
  for (const ns of entries) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `ns-card${ns.namespace === active ? " active" : ""}`;
    const stm = ns.stm || {};
    const ltm = ns.ltm || {};
    card.innerHTML = `
      <span class="ns-name">${escapeHtml(nsLabel(ns.namespace))}</span>
      <span class="ns-dbs">${escapeHtml(ns.stm_db || "")} · ${escapeHtml(ns.ltm_db || "")}</span>
      <span class="ns-row"><i>STM</i>${metaCell("sess", stm.sessions)}${metaCell("chunk", stm.chunks)}${metaCell("turn", stm.turns)}${metaCell("ent", stm.entities)}</span>
      <span class="ns-row"><i>LTM</i>${metaCell("sess", ltm.sessions)}${metaCell("ent", ltm.entities)}${metaCell("topic", ltm.topics)}${metaCell("chunk", ltm.chunks)}${metaCell("turn", ltm.turns)}</span>
    `;
    card.addEventListener("click", () => {
      if (namespaceSelect.value === ns.namespace) return;
      namespaceSelect.value = ns.namespace;
      onNamespaceChange();
    });
    namespaceCards.append(card);
  }
}

async function loadNamespaces(preferred = null) {
  const previous = preferred ?? namespaceSelect.value;
  try {
    const data = await fetchJson("/api/namespaces");
    const entries = data.namespaces || [];
    const values = entries.map((ns) => ns.namespace);

    namespaceSelect.innerHTML = "";
    for (const ns of entries) {
      const option = document.createElement("option");
      option.value = ns.namespace;
      option.textContent = nsLabel(ns.namespace);
      namespaceSelect.append(option);
    }

    // Preserve current selection; else fall back to the server default, else default.
    let selected = previous;
    if (!values.includes(selected)) {
      selected = values.includes(data.default_namespace) ? data.default_namespace : (values[0] ?? "");
    }
    namespaceSelect.value = selected;

    renderNamespaceCards(entries, namespaceSelect.value);
  } catch (error) {
    namespaceCards.innerHTML = `<div class="ns-empty ns-err">${escapeHtml(error.message)}</div>`;
  }
}

function onNamespaceChange() {
  updateFlushButton();
  // Selecting a namespace clears session-scoped state — a session id belongs to
  // one namespace's databases and rarely carries over.
  loadSessions();
  loadGraph();
  // Reflect the new active card without a full metadata reload.
  for (const card of namespaceCards.querySelectorAll(".ns-card")) {
    card.classList.toggle("active", card.querySelector(".ns-name")?.textContent === nsLabel(currentNamespace()));
  }
}

async function loadSessions() {
  try {
    const params = new URLSearchParams();
    if (currentNamespace()) params.set("namespace", currentNamespace());
    const data = await fetchJson(`/api/sessions?${params}`);
    sessionsList.innerHTML = "";
    for (const sessionId of data.sessions || []) {
      const option = document.createElement("option");
      option.value = sessionId;
      sessionsList.append(option);
    }
  } catch {
    sessionsList.innerHTML = "";
  }
}

async function loadGraph() {
  const scope = currentScope();
  const session = sessionInput.value.trim();

  // Client-side guard: STM graph requires a session ID
  if (scope === "stm" && !session) {
    prepareGraph({ nodes: [], links: [] });
    selectionEl.textContent = "Enter a session ID to view the STM graph.";
    loadStats();
    return;
  }

  const params = new URLSearchParams({ scope, limit: limitInput.value || "200" });
  if (session) params.set("session_id", session);
  if (currentNamespace()) params.set("namespace", currentNamespace());
  resultsEl.innerHTML = "";
  selectionEl.textContent = "Loading graphâ€¦";
  try {
    const data = await fetchJson(`/api/graph?${params}`);
    prepareGraph(data);
    selectionEl.textContent = state.nodes.length ? "Select a node" : "No nodes found";
  } catch (error) {
    prepareGraph({ nodes: [], links: [] });
    selectionEl.textContent = error.message;
  }
  loadStats();
}

function renderResults(hits) {
  resultsEl.innerHTML = "";
  if (!hits.length) {
    resultsEl.textContent = "No recall hits";
    return;
  }
  for (const hit of hits) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result-item";
    button.innerHTML = `<small>${escapeHtml(hit.type || "hit")} Â· score ${escapeHtml(Number(hit.score ?? 0).toFixed(4))}</small><span class="result-text">${escapeHtml(hit.text || "")}</span>`;
    button.addEventListener("click", () => {
      state.highlighted = new Set((hit.entities || []).map(String));
      const first = [...state.highlighted].map((id) => state.nodeById.get(id)).find(Boolean);
      if (first) showSelection(first);
      draw();
    });
    resultsEl.append(button);
  }
}

async function search(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  const scope = currentScope();

  // Client-side guard: STM recall requires a session
  if (scope === "stm" && !sessionInput.value.trim()) {
    resultsEl.textContent = "Enter a session ID to recall from STM.";
    return;
  }

  const effectiveScope = scope === "stm" ? "stm" : sessionInput.value.trim() ? "both" : "ltm";
  const params = new URLSearchParams({ q: query, scope: effectiveScope, top_k: "8" });
  if (sessionInput.value.trim()) params.set("session_id", sessionInput.value.trim());
  if (currentNamespace()) params.set("namespace", currentNamespace());
  resultsEl.textContent = "Searchingâ€¦";
  try {
    const data = await fetchJson(`/api/recall?${params}`);
    renderResults(data.hits || []);
    pushOpLog("recall", `${(data.hits || []).length} hits for "${query.slice(0, 40)}"`);
  } catch (error) {
    resultsEl.textContent = error.message;
    pushOpLog("recall", error.message, false);
  }
}

// â”€â”€ Ingest drop-zone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function setDropState(cls, text) {
  dropZone.className = `drop-zone ${cls}`;
  const label = dropZone.querySelector(".drop-label");
  if (label) label.textContent = text;
}

function resetDropZone() {
  setDropState("", "Drop .md / .txt file to ingest into STM");
}

async function ingestFile(file) {
  const sessionId = sessionInput.value.trim();
  if (!sessionId) {
    setDropState("err", "Set a session ID first");
    ingestStatus.textContent = "Enter a session ID in the Session field above, then drop a file.";
    ingestStatus.className = "ingest-status err";
    setTimeout(resetDropZone, 5000);
    return;
  }

  setDropState("loading", "Ingestingâ€¦");
  ingestStatus.textContent = "";

  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("session_id", sessionId);
  formData.append("namespace", currentNamespace());
  formData.append("title", file.name.replace(/\.[^.]+$/, ""));

  try {
    const data = await fetchJson("/api/ingest", { method: "POST", body: formData });
    const summary = `${data.chunk_count} chunks Â· ${data.entity_count} entities`;
    setDropState("ok", `âś“ ${summary}`);
    ingestStatus.textContent = `Ingested in ${data.elapsed_s}s`;
    ingestStatus.className = "ingest-status ok";
    pushOpLog("ingest", `${escapeHtml(file.name)}: ${summary}`);
    // Refresh graph if we're on STM scope
    if (currentScope() === "stm") loadGraph();
    else loadStats();
    // Reset drop zone after a moment
    setTimeout(resetDropZone, 4000);
  } catch (error) {
    setDropState("err", "âś— Ingest failed");
    ingestStatus.textContent = error.message;
    ingestStatus.className = "ingest-status err";
    pushOpLog("ingest", error.message, false);
    setTimeout(resetDropZone, 4000);
  }
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (dropZone.className.includes("loading")) return;
  setDropState("hover", "Drop to ingestâ€¦");
});

dropZone.addEventListener("dragleave", () => {
  if (dropZone.className.includes("loading")) return;
  resetDropZone();
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (dropZone.className.includes("loading")) return;
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  ingestFile(file);
});

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".md,.txt,text/markdown,text/plain";
    input.addEventListener("change", () => { if (input.files?.[0]) ingestFile(input.files[0]); });
    input.click();
  }
});

// â”€â”€ Flush â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderFlushStats(data) {
  flushStatsSection.hidden = false;

  const rows = [
    ["Entities dropped", data.entities_dropped],
    ["Entities merged", data.merged_entities],
    ["PageRank updated", data.pagerank_updated],
    ["Graph embeddings", data.graph_embeddings_computed],
    ["Communities", data.communities_computed],
    ["Modularity", typeof data.modularity === "number" ? data.modularity.toFixed(4) : "â€”"],
    ["Chunks consolidated", data.chunks_consolidated],
    ["Time", `${data.elapsed_s}s`],
  ];

  flushStatsGrid.innerHTML = rows
    .map(([label, value]) => `<div><dt>${escapeHtml(String(label))}</dt><dd>${escapeHtml(String(value ?? "â€”"))}</dd></div>`)
    .join("");

  if (data.errors && data.errors.length > 0) {
    flushErrors.hidden = false;
    flushErrors.innerHTML = `<strong>Warnings</strong><ul>${data.errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>`;
  } else {
    flushErrors.hidden = true;
  }
}

async function flushSession() {
  const sessionId = sessionInput.value.trim();
  if (!sessionId) return;

  flushButton.disabled = true;
  flushButton.textContent = "Flushingâ€¦";
  flushStatsSection.hidden = true;

  try {
    const data = await fetchJson("/api/flush", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        namespace: currentNamespace(),
        options: { clear_stm: true, run_gds: true, merge_entities: true },
      }),
    });

    renderFlushStats(data);
    pushOpLog("flush", `${data.communities_computed} communities Â· ${data.merged_entities} merged Â· ${data.chunks_consolidated} consolidated`);

    // Switch to LTM and reload to show result
    const ltmRadio = document.querySelector("input[name='scope'][value='ltm']");
    if (ltmRadio) { ltmRadio.checked = true; updateFlushButton(); }
    loadGraph();
    loadSessions();
    loadNamespaces();
  } catch (error) {
    flushStatsSection.hidden = false;
    flushStatsGrid.innerHTML = "";
    flushErrors.hidden = false;
    flushErrors.innerHTML = `<strong>Flush failed</strong><p>${escapeHtml(error.message)}</p>`;
    pushOpLog("flush", error.message, false);
  } finally {
    flushButton.textContent = "Flush â†’ LTM";
    updateFlushButton();
  }
}

flushButton.addEventListener("click", flushSession);

// â”€â”€ Canvas interactions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  const node = pickNode(event.clientX, event.clientY);
  const rect = canvas.getBoundingClientRect();
  state.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  if (node) {
    state.draggingNode = node;
    showSelection(node);
  } else {
    state.draggingCanvas = true;
  }
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.pointer) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (state.draggingNode) {
    const point = screenToWorld(x, y);
    state.draggingNode.x = point.x;
    state.draggingNode.y = point.y;
    startSimulation();
    draw();
  } else if (state.draggingCanvas) {
    state.panX += x - state.pointer.x;
    state.panY += y - state.pointer.y;
    draw();
  }
  state.pointer = { x, y };
});

canvas.addEventListener("pointerup", () => {
  state.draggingNode = null;
  state.draggingCanvas = false;
  state.pointer = null;
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouse = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const before = screenToWorld(mouse.x, mouse.y);
  const factor = event.deltaY > 0 ? 0.9 : 1.1;
  state.zoom = Math.max(0.25, Math.min(3, state.zoom * factor));
  const after = worldToScreen(before.x, before.y);
  state.panX += mouse.x - after.x;
  state.panY += mouse.y - after.y;
  draw();
}, { passive: false });

// â”€â”€ Event wiring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
refreshButton.addEventListener("click", loadGraph);
searchForm.addEventListener("submit", search);
namespaceSelect.addEventListener("change", onNamespaceChange);
nsRefreshButton.addEventListener("click", loadNamespaces);

sessionInput.addEventListener("input", () => { updateFlushButton(); });
sessionInput.addEventListener("change", loadGraph);
limitInput.addEventListener("change", loadGraph);

for (const input of document.querySelectorAll("input[name='scope']")) {
  input.addEventListener("change", () => { updateFlushButton(); loadGraph(); });
}

// â”€â”€ Bootstrap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("session_id")) sessionInput.value = urlParams.get("session_id");
if (urlParams.get("scope")) {
  const radio = document.querySelector(`input[name='scope'][value='${urlParams.get("scope")}']`);
  if (radio) radio.checked = true;
}

window.addEventListener("resize", resize);
resize();
updateFlushButton();
loadHealth();

// Resolve the namespace list before loading namespace-scoped graph/sessions so
// the first requests target the right database pair.
(async () => {
  await loadNamespaces(urlParams.get("namespace"));
  updateFlushButton();
  loadSessions();
  loadGraph();
})();
