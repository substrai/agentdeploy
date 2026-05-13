"""Base adapter interface for agent frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agentdeploy.session.manager import Session


class BaseAdapter(ABC):
    """Base class for agent framework adapters.

    Implement this to support any agent framework (LangChain, CrewAI, etc.)
    """

    name: str = "base"

    @abstractmethod
    def invoke(self, message: str, session: Session, **kwargs) -> str:
        """Invoke the agent with a message.

        Args:
            message: User message
            session: Current session with history

        Returns:
            Agent response string
        """
        pass

    def get_tool_calls(self) -> list:
        """Get tool calls made during last invocation."""
        return []

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage from last invocation."""
        return {"input_tokens": 0, "output_tokens": 0}


class CustomAdapter(BaseAdapter):
    """Adapter for custom agent functions (default)."""

    name = "custom"

    def __init__(self, agent_fn):
        self.agent_fn = agent_fn

    def invoke(self, message: str, session: Session, **kwargs) -> str:
        result = self.agent_fn(message, session)
        return str(result) if result else ""


class EchoAdapter(BaseAdapter):
    """Simple echo adapter for testing."""

    name = "echo"

    def invoke(self, message: str, session: Session, **kwargs) -> str:
        return f"Echo: {message}"
