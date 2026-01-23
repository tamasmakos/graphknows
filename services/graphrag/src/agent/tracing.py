
"""
Tracing module for the Knowledge Graph Agent.

This module provides event handlers to capture the agent's reasoning process,
including thoughts, tool calls, and observations, to be displayed in the frontend.
"""

from typing import Any, List, Optional, Dict
import json

from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMCompletionEndEvent, LLMChatEndEvent
from llama_index.core.instrumentation.events.agent import (
    AgentToolCallEvent,
)
# Note: Import logic might need adjustment based on installed llama-index-core version
# detailed inspection of events in test_event_handler.py suggested LLMCompletionEndEvent is key.

class GraphAgentEventHandler(BaseEventHandler):
    """
    Custom event handler to capture agent reasoning steps.
    """
    
    events: List[BaseEvent] = []
    reasoning_steps: List[str] = []
        
    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        self.events.append(event)
        
        # Process events into human-readable steps
        event_type = type(event).__name__
        
        # Capture Agent Thoughts (LLM Completions)
        if isinstance(event, (LLMCompletionEndEvent, LLMChatEndEvent)):
            response_text = ""
            if hasattr(event.response, "text"):
                response_text = event.response.text
            elif hasattr(event.response, "message"):
                response_text = event.response.message.content
                
            if response_text:
                # Filter out pure JSON if it's just a tool call request (heuristic)
                # But sometimes thoughts are mixed with tool calls.
                # LlamaIndex usually separates them or puts them in specific formats.
                # simpler approach: just log everything that looks like text.
                self.reasoning_steps.append(f"🤔 **Thought:** {response_text}")

        # Capture Tool Calls
        if isinstance(event, AgentToolCallEvent):
            tool_name = event.tool.name
            tool_args = str(event.tool.tool_kwargs) if hasattr(event.tool, "tool_kwargs") else str(event.tool.fn_kwargs) if hasattr(event.tool, "fn_kwargs") else ""
            # Some versions use arguments differently
            if not tool_args and hasattr(event, "tool_args"):
                 tool_args = str(event.tool_args)
            
            self.reasoning_steps.append(f"🔧 **Tool Call:** {tool_name}({tool_args})")
    
    def get_reasoning_chain(self) -> List[str]:
        """
        Return the captured reasoning chain as a list of strings.
        """
        return self.reasoning_steps
