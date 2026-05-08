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
  IMPORTANT: The `text` value must be a PLAIN MARKDOWN STRING — NOT JSON, NOT escaped, NOT nested objects.
  Write the report directly as markdown text starting with ## heading. Example:
  {{"tool": "answer", "params": {{"text": "## My Report\n\n**Executive Summary**: ..."}}}}
  Do NOT put another JSON object inside the text field.

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
                parsed = None
                brace_idx = raw.find('{')
                if brace_idx >= 0:
                    # Count braces to find the matching closing brace
                    depth = 0
                    in_str = False
                    escape = False
                    for end_idx, ch in enumerate(raw[brace_idx:], start=brace_idx):
                        if escape:
                            escape = False
                            continue
                        if ch == '\\' and in_str:
                            escape = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if in_str:
                            continue
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(raw[brace_idx:end_idx + 1])
                                except json.JSONDecodeError:
                                    pass
                                break

                if parsed is None:
                    # LLM produced text with no extractable JSON — treat as final answer
                    log.warning("LLM returned non-JSON, treating as final answer: %s", raw[:200])
                    import re as _re2
                    text_fallback = raw.strip()
                    text_fallback = _re2.sub(r'\{"?tool"?.*$', '', text_fallback, flags=_re2.DOTALL).strip()
                    report_fb = _re2.search(r'^##\s+(?!Step\s+\d)', text_fallback, flags=_re2.MULTILINE)
                    if report_fb and report_fb.start() > 0:
                        text_fallback = text_fallback[report_fb.start():]
                    else:
                        lines_fb = text_fallback.split('\n')
                        clean_fb: list[str] = []
                        in_step_fb = False
                        for line in lines_fb:
                            if _re2.match(r'^##\s+Step\s+\d', line):
                                in_step_fb = True
                                continue
                            if _re2.match(r'^##\s+', line):
                                in_step_fb = False
                            if not in_step_fb:
                                clean_fb.append(line)
                        text_fallback = '\n'.join(clean_fb).strip()
                    yield StreamEvent(event="answer", data={"text": text_fallback, "steps": step_id})
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
                # Recursively unwrap: Groq sometimes wraps the full JSON response inside params.text
                for _ in range(3):
                    if not (isinstance(text, str) and text.strip().startswith("{")):
                        break
                    try:
                        inner = json.loads(text)
                        inner_text = (
                            inner.get("params", {}).get("text")
                            or inner.get("text")
                            or inner.get("answer")
                            or inner.get("content")
                        )
                        if inner_text and inner_text != text:
                            text = inner_text
                        else:
                            break
                    except (json.JSONDecodeError, AttributeError):
                        break
                # Strip any trailing JSON tool calls the model accidentally included in the text
                import re as _re
                text = _re.sub(r'\{["\']tool["\'].*$', '', text, flags=_re.DOTALL).strip()
                # Strip planning preamble: "## Step N: ..." sections that leak LLM reasoning
                # Strategy 1: find first non-Step ## heading and start there
                report_start = _re.search(r'^##\s+(?!Step\s+\d)', text, flags=_re.MULTILINE)
                if report_start and report_start.start() > 0:
                    text = text[report_start.start():]
                else:
                    # Strategy 2: drop all lines that are ## Step ... headings and their content
                    # Keep only lines that look like real report content
                    lines = text.split('\n')
                    clean: list[str] = []
                    in_step = False
                    for line in lines:
                        if _re.match(r'^##\s+Step\s+\d', line):
                            in_step = True
                            continue
                        if _re.match(r'^##\s+', line):
                            in_step = False  # new non-step section starts
                        if not in_step:
                            clean.append(line)
                    text = '\n'.join(clean).strip()
                    # If still all Step-preamble, just take the last paragraph as answer
                    if not text or _re.match(r'^##\s+Step', text):
                        last_para = [p.strip() for p in _re.split(r'\n\n+', text_fallback if 'text_fallback' in dir() else text) if p.strip()]
                        text = last_para[-1] if last_para else text
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
