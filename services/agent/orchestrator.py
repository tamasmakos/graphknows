"""IterativeOrchestrator — ReAct loop for the agent service."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx
from groq import AsyncGroq

from agent.tools.base import BaseTool
from graphknows.shared.models.session import StreamEvent, ReasoningStep

log = logging.getLogger("agent.orchestrator")

_SYSTEM_PROMPT_TEMPLATE = """You are an EU public records investigator with real-time web access.

You combine two complementary search capabilities:
1. **EU Official Archives** — European Parliament debates, EUR-Lex legislation, EC press releases, ECB publications, ECA audit reports, Council conclusions (primary sources, citable)
2. **Tavily Web Search** — live web for general context, news, academic analysis, and background that official archives don't carry

Available tools:
{tool_list}

STRATEGY — always run two parallel search angles:
1. First `search` call: target EU official sources (ep_debates, eurlex, ec_press) for primary evidence
2. Second `search` call: use "tavily" or omit sources to get web context, news, and analysis
3. Optionally `retrieve` for KG context from past sessions
4. Synthesise with `answer` — cite BOTH official and web sources separately

For `search`: {{"query": "...", "sources": ["ep_debates", "eurlex", "tavily"], "max_docs": 8, "session_id": "...", "time_range": "week|month|year"}}
  - EU sources: ep_debates, ep_resolutions, eurlex, council_conclusions, ec_press, ecb_publications, eca_reports, eprs_briefings
  - Web source: tavily (always available — searches live web + EU authoritative domains)
  - Omit sources to auto-route
For `retrieve`: {{"query": "...", "k": 5}}
For `answer`: {{"text": "..."}}

REPORT FORMAT — the `answer` text must be a full investigative report:
## [Investigation Title]

**Executive Summary**: 2-3 sentence overview with the key finding.

### Key Findings
1. [Finding] — [Doc-1: Title (Institution, Year)](url)
2. [Finding] — [Doc-2: Title (Source, Year)](url)

### Official Record
- EU legislation, debates, press releases with CELEX/ELI numbers where available

### Context & Analysis
- News coverage, academic commentary, broader context from web sources

### Key Entities
- **People**: [MEPs, Commissioners, etc.]
- **Organizations**: [EU institutions, NGOs, companies]
- **Legislation/Policy**: [Regulations, Directives, Decisions]

### Sources
- [Doc-N]: [Full title] — [Institution/Domain] — [Date] — [URL]

Distinguish: 🏛️ for official EU records, 🌐 for web sources.
Cite inline with [Doc-N] matching the Sources section.

Respond ONLY with valid JSON:
{{"thought": "...", "tool": "<tool_name>", "params": {{...}}}}
OR when done:
{{"thought": "...", "tool": "answer", "params": {{"text": "..."}}}}
"""


class IterativeOrchestrator:
    """ReAct orchestrator that drives the agent tool loop.

    Attributes:
        tools: Mapping of tool name to BaseTool instance.
        llm_model: Groq model name for orchestration.
        max_steps: Maximum tool invocation steps before forcing synthesis.
    """

    def __init__(
        self,
        tools: dict[str, BaseTool],
        groq_api_key: str,
        llm_model: str = "llama-3.3-70b-versatile",
        max_steps: int = 8,
    ) -> None:
        self._tools = tools
        self._groq = AsyncGroq(api_key=groq_api_key) if groq_api_key else None
        self._model = llm_model
        self._max_steps = max_steps

    async def run(self, query: str, session_id: str) -> AsyncIterator[StreamEvent]:
        """Run the ReAct loop for a query, yielding StreamEvents.

        Args:
            query: The user query to answer.
            session_id: Session identifier for context tracking.

        Yields:
            StreamEvent frames: step_start, step_result, answer, or error.
        """
        tool_list = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self._tools.items()
        )
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tool_list=tool_list)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        step_id = 0
        observations = []

        for _ in range(self._max_steps):
            # Think
            if self._groq is None:
                yield StreamEvent(event="answer", data={"text": "LLM not configured.", "steps": 0})
                return

            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON object from response (LLM sometimes adds preamble/markdown)
                import re
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        parsed = None
                else:
                    parsed = None

                if parsed is None:
                    # If the LLM produced plain text with no JSON, treat it as the final answer
                    log.warning("LLM returned non-JSON, treating as final answer: %s", raw[:200])
                    yield StreamEvent(event="answer", data={"text": raw, "steps": step_id})
                    return

            tool_name = parsed.get("tool", "")
            params = parsed.get("params", {})
            thought = parsed.get("thought", "")

            if tool_name == "search" and isinstance(params, dict) and not params.get("session_id"):
                params["session_id"] = session_id

            # Done
            if tool_name == "answer":
                # LLM may use "text", "answer", or "content" as the key.
                text = params.get("text") or params.get("answer") or params.get("content") or ""
                # Guard: if text is itself a JSON string (double-wrapped), unwrap it
                if isinstance(text, str) and text.strip().startswith("{"):
                    try:
                        inner = json.loads(text)
                        inner_text = (
                            inner.get("params", {}).get("text")
                            or inner.get("text")
                            or inner.get("answer")
                            or inner.get("content")
                        )
                        if inner_text:
                            text = inner_text
                    except (json.JSONDecodeError, AttributeError):
                        pass
                yield StreamEvent(event="answer", data={"text": text, "steps": step_id})
                return

            # Emit step_start
            step_id += 1
            yield StreamEvent(event="step_start", data={"step_id": step_id, "tool": tool_name, "thought": thought, "params": params})

            # Call tool
            tool = self._tools.get(tool_name)
            if tool is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = await tool.run(params)
                except NotImplementedError as exc:
                    result = {"error": str(exc)}
                except Exception as exc:
                    result = {"error": str(exc)}

            # Emit step_result
            yield StreamEvent(event="step_result", data={"step_id": step_id, "tool": tool_name, "result": result})

            # Add to messages for next LLM call
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)[:2000]}"})

        # Max steps reached — force synthesis
        yield StreamEvent(event="answer", data={"text": "Max reasoning steps reached. Partial results gathered.", "steps": step_id})
