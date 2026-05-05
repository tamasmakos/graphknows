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

_SYSTEM_PROMPT_TEMPLATE = """You are a political intelligence assistant with access to a knowledge graph
of European Parliament debates, legislation, and regulatory documents.

You have the following tools available. Use ONLY these tool names exactly as listed:

{tool_list}

CRITICAL — use EXACTLY these parameter field names (copy-paste from the Params spec above):
- retrieve: {{"query": "...", "k": 10}}
- get_chunks: {{"entity_id": "ENTITY_NAME", "top_k": 5}}   <- entity name in UPPER_SNAKE_CASE from retrieve results
- get_community: {{"entity_id": "ENTITY_NAME"}}              <- same entity name format
- get_path: {{"entity_a": "ENTITY_A", "entity_b": "ENTITY_B"}}  <- not entity1/entity2
- build: {{"input_files": []}}
- search: {{"query": "..."}}

Think step by step. Use tools to gather evidence before synthesising an answer.
Always cite your sources.

Respond ONLY with valid JSON in this format:
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
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                yield StreamEvent(event="error", data={"detail": f"LLM returned non-JSON: {raw[:200]}"})
                return

            tool_name = parsed.get("tool", "")
            params = parsed.get("params", {})
            thought = parsed.get("thought", "")

            # Done
            if tool_name == "answer":
                yield StreamEvent(event="answer", data={"text": params.get("text", ""), "steps": step_id})
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
