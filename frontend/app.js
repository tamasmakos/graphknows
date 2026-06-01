const canvas = document.querySelector("#graphCanvas");
const ctx = canvas.getContext("2d");
const emptyState = document.querySelector("#emptyState");
const sessionInput = document.querySelector("#sessionInput");
const sessionsList = document.querySelector("#sessions");
const limitInput = document.querySelector("#limitInput");
const refreshButton = document.querySelector("#refreshButton");
const searchForm = document.querySelector("#searchForm");
const queryInput = document.querySelector("#queryInput");
const resultsEl = document.querySelector("#results");
const healthEl = document.querySelector("#health");
const nodeCountEl = document.querySelector("#nodeCount");
const edgeCountEl = document.querySelector("#edgeCount");
const stmChunksEl = document.querySelector("#stmChunks");
const ltmTopicsEl = document.querySelector("#ltmTopics");
const selectionEl = document.querySelector("#selection");

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

function currentScope() {
  return document.querySelector("input[name='scope']:checked").value;
}

function setText(el, value) {
  el.textContent = value ?? "0";
}

function setHealth(ok, label) {
  healthEl.textContent = label;
  healthEl.classList.toggle("ok", ok);
  healthEl.classList.toggle("bad", !ok);
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

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
  selectionEl.innerHTML = `<strong>${escapeHtml(node.label || node.id)}</strong><br>${escapeHtml(node.type || "Entity")} ${node.topic ? `- topic ${escapeHtml(node.topic)}` : ""}`;
  draw();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

async function loadHealth() {
  try {
    const data = await fetchJson("/api/health");
    const ok = Object.entries(data).filter(([key]) => key.endsWith("connected")).every(([, value]) => Boolean(value));
    setHealth(ok, ok ? "Online" : "Partial");
  } catch (error) {
    setHealth(false, "Offline");
  }
}

async function loadStats() {
  try {
    const params = new URLSearchParams();
    if (sessionInput.value) params.set("session_id", sessionInput.value);
    const data = await fetchJson(`/api/stats?${params}`);
    setText(stmChunksEl, data.stm?.chunks ?? data.stm?.chunk_count ?? 0);
    setText(ltmTopicsEl, data.ltm?.topics ?? data.ltm?.topic_count ?? 0);
  } catch (error) {
    setText(stmChunksEl, "-");
    setText(ltmTopicsEl, "-");
  }
}

async function loadSessions() {
  try {
    const data = await fetchJson("/api/sessions");
    sessionsList.innerHTML = "";
    for (const sessionId of data.sessions || []) {
      const option = document.createElement("option");
      option.value = sessionId;
      sessionsList.append(option);
    }
  } catch (error) {
    sessionsList.innerHTML = "";
  }
}

async function loadGraph() {
  const params = new URLSearchParams({ scope: currentScope(), limit: limitInput.value || "200" });
  if (sessionInput.value.trim()) params.set("session_id", sessionInput.value.trim());
  resultsEl.innerHTML = "";
  selectionEl.textContent = "Loading graph";
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
    button.innerHTML = `<small>${escapeHtml(hit.type || "hit")} - score ${escapeHtml(hit.score ?? "")}</small><span class="result-text">${escapeHtml(hit.text || "")}</span>`;
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
  const scope = currentScope() === "stm" ? "stm" : sessionInput.value.trim() ? "both" : "ltm";
  const params = new URLSearchParams({ q: query, scope, top_k: "8" });
  if (sessionInput.value.trim()) params.set("session_id", sessionInput.value.trim());
  resultsEl.textContent = "Searching";
  try {
    const data = await fetchJson(`/api/recall?${params}`);
    renderResults(data.hits || []);
  } catch (error) {
    resultsEl.textContent = error.message;
  }
}

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

refreshButton.addEventListener("click", loadGraph);
searchForm.addEventListener("submit", search);
sessionInput.addEventListener("change", loadGraph);
limitInput.addEventListener("change", loadGraph);
for (const input of document.querySelectorAll("input[name='scope']")) {
  input.addEventListener("change", loadGraph);
}

const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("session_id")) sessionInput.value = urlParams.get("session_id");
if (urlParams.get("scope")) {
  const radio = document.querySelector(`input[name='scope'][value='${urlParams.get("scope")}']`);
  if (radio) radio.checked = true;
}

window.addEventListener("resize", resize);
resize();
loadHealth();
loadSessions();
loadGraph();
