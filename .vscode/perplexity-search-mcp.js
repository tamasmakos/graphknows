#!/usr/bin/env node
/**
 * Perplexity Search-only MCP server.
 *
 * Exposes exactly ONE tool: perplexity_search (Perplexity Search API).
 * Does NOT expose perplexity_ask, perplexity_research, or perplexity_reason,
 * which all call Sonar LLM endpoints.
 *
 * Zero external dependencies — uses only Node.js built-ins (Node ≥ 18 required
 * for native fetch).
 */
"use strict";

const API_KEY = process.env.PERPLEXITY_API_KEY;
const BASE_URL = process.env.PERPLEXITY_BASE_URL || "https://api.perplexity.ai";
const TIMEOUT_MS = parseInt(process.env.PERPLEXITY_TIMEOUT_MS || "300000", 10);

if (!API_KEY) {
  process.stderr.write("PERPLEXITY_API_KEY environment variable is required\n");
  process.exit(1);
}

// ── MCP stdio transport ────────────────────────────────────────────────────────
let buffer = "";
process.stdin.setEncoding("utf8");

process.stdin.on("data", (chunk) => {
  buffer += chunk;
  let nl;
  while ((nl = buffer.indexOf("\n")) !== -1) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (line) {
      try {
        handleMessage(JSON.parse(line));
      } catch (e) {
        process.stderr.write(`JSON parse error: ${e.message}\n`);
      }
    }
  }
});

process.stdin.on("end", () => process.exit(0));

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function reply(id, result) {
  send({ jsonrpc: "2.0", id, result });
}

function replyError(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

// ── Tool definition ────────────────────────────────────────────────────────────
const TOOL_DEF = {
  name: "perplexity_search",
  description:
    "Search the web using Perplexity's Search API. Returns a ranked list of " +
    "results with titles, URLs, snippets, and dates. " +
    "Does NOT invoke any Perplexity Sonar LLM model — raw search results only.",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search query string",
      },
      max_results: {
        type: "integer",
        description: "Maximum number of results to return (1–10, default 10)",
        minimum: 1,
        maximum: 10,
      },
      max_tokens_per_page: {
        type: "integer",
        description: "Maximum tokens per page snippet (default 1024)",
      },
      country: {
        type: "string",
        description:
          "ISO 3166-1 alpha-2 country code for regional results (e.g. 'US', 'GB')",
      },
    },
    required: ["query"],
  },
};

// ── Search implementation ──────────────────────────────────────────────────────
async function doSearch(query, maxResults = 10, maxTokensPerPage = 1024, country) {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let resp;
  try {
    resp = await fetch(`${BASE_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
        "User-Agent": "perplexity-search-mcp/1.0.0",
      },
      body: JSON.stringify({
        query,
        max_results: maxResults,
        max_tokens_per_page: maxTokensPerPage,
        ...(country ? { country } : {}),
      }),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(tid);
  }

  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Perplexity Search API error: ${resp.status} ${resp.statusText}\n${txt}`);
  }

  const data = await resp.json();

  if (!data.results || data.results.length === 0) {
    return "No search results found.";
  }

  let out = `Found ${data.results.length} search results:\n\n`;
  data.results.forEach((r, i) => {
    out += `${i + 1}. ${r.title || "No title"}\n`;
    out += `   URL: ${r.url || "No URL"}\n`;
    const snippet = r.snippet || r.description || "";
    if (snippet) out += `   ${snippet}\n`;
    if (r.date) out += `   Date: ${r.date}\n`;
    out += "\n";
  });

  return out;
}

// ── MCP message handler ────────────────────────────────────────────────────────
async function handleMessage(msg) {
  // Notifications have no id — silently ignore
  if (msg.id === undefined && msg.method) return;

  const { id, method, params } = msg;

  if (method === "initialize") {
    reply(id, {
      protocolVersion: "2024-11-05",
      serverInfo: { name: "perplexity-search", version: "1.0.0" },
      capabilities: { tools: {} },
      instructions:
        "Perplexity Search API — returns ranked web results (title, URL, snippet, date). " +
        "This server intentionally exposes only perplexity_search. " +
        "Sonar LLM tools (perplexity_ask, perplexity_research, perplexity_reason) are not available.",
    });
  } else if (method === "tools/list") {
    reply(id, { tools: [TOOL_DEF] });
  } else if (method === "tools/call") {
    const { name, arguments: args = {} } = params || {};
    if (name !== "perplexity_search") {
      replyError(id, -32601, `Unknown tool: ${name}`);
      return;
    }
    try {
      const text = await doSearch(
        args.query,
        args.max_results ?? 10,
        args.max_tokens_per_page ?? 1024,
        args.country
      );
      reply(id, { content: [{ type: "text", text }] });
    } catch (err) {
      reply(id, {
        content: [{ type: "text", text: `Error: ${err.message}` }],
        isError: true,
      });
    }
  } else if (method === "ping") {
    reply(id, {});
  } else if (id !== undefined) {
    replyError(id, -32601, `Method not found: ${method}`);
  }
}
