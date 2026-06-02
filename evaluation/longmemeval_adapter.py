"""Optional LongMemEval-V2 adapter backed by graphknows MCP tools."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from evaluation.graphknows_mcp_client import GraphknowsMCPClient


@dataclass
class LongMemEvalGraphknowsMemory:
    """LongMemEval-style memory backend using graphknows MCP."""

    client: GraphknowsMCPClient
    session_id: str = "eval-longmemeval"
    top_k: int = 5
    max_context_chars: int = 8000

    async def insert(self, trajectory: dict[str, Any]) -> None:
        """Insert a LongMemEval trajectory into graphknows STM."""
        documents = []
        steps = trajectory.get("steps", [])
        for idx, step in enumerate(steps):
            observation = step.get("observation") or step.get("text") or ""
            notes = step.get("notes") or step.get("trajectory_notes") or ""
            text = "\n".join(str(part) for part in (observation, notes) if part)
            if text.strip():
                documents.append(
                    {
                        "id": f"{trajectory.get('id', 'trajectory')}-{idx}",
                        "title": f"LongMemEval trajectory {trajectory.get('id', '')} step {idx}",
                        "source": "longmemeval-v2",
                        "text": text,
                    }
                )
        if documents:
            await self.client.call_tool(
                "corpus_ingest",
                {"documents": documents, "session_id": self.session_id},
            )

    async def query(self, query: str, query_image: str | None = None) -> list[dict[str, str]]:
        """Return compact text context items for a LongMemEval query."""
        del query_image
        result = await self.client.call_tool(
            "memory_recall",
            {
                "query": query,
                "session_id": self.session_id,
                "top_k": self.top_k,
                "scope": "both",
            },
        )
        budget = self.max_context_chars
        items: list[dict[str, str]] = []
        for hit in result.get("hits", []):
            text = str(hit.get("text", "")).strip()
            if not text or budget <= 0:
                continue
            clipped = text[:budget]
            budget -= len(clipped)
            items.append({"type": "text", "value": clipped})
        return items


async def amain(argv: list[str] | None = None) -> int:
    """Dry-run the adapter against one synthetic trajectory."""
    parser = argparse.ArgumentParser(description="Dry-run the LongMemEval graphknows adapter.")
    parser.add_argument("--command", default="graphknows-mcp")
    parser.add_argument("--session-id", default="eval-longmemeval-dry-run")
    args = parser.parse_args(argv)
    async with GraphknowsMCPClient(args.command) as client:
        memory = LongMemEvalGraphknowsMemory(client=client, session_id=args.session_id)
        await memory.insert(
            {
                "id": "dry-run",
                "steps": [
                    {
                        "observation": (
                            "LongMemEval dry run: Project Helios stores memory context "
                            "in graphknows."
                        )
                    }
                ],
            }
        )
        context = await memory.query("Where does Project Helios store memory context?")
    print(context)
    return 0


def main() -> None:
    """Run the adapter dry-run CLI."""
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()

