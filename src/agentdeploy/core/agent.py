"""Agent decorator and configuration.

The @agent decorator turns a Python function into a deployable agent
with session management, tool access, and cost tracking.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for a deployed agent."""

    name: str
    model: str = "bedrock/claude-3-sonnet"
    system_prompt: str = "You are a helpful assistant."
    tools: List[Any] = field(default_factory=list)
    max_iterations: int = 10
    timeout_seconds: int = 60
    memory_strategy: str = "sliding_window"
    memory_window_size: int = 20
    memory_ttl_hours: int = 24
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt[:50] + "..." if len(self.system_prompt) > 50 else self.system_prompt,
            "tools": [getattr(t, "name", str(t)) for t in self.tools],
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "memory_strategy": self.memory_strategy,
        }


class AgentFunction:
    """Wrapper around a decorated agent function."""

    def __init__(self, fn: Callable, config: AgentConfig):
        self.fn = fn
        self.config = config
        self.name = config.name
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def invoke(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Invoke the agent with a message.

        Args:
            message: User message
            session_id: Optional session ID for conversation continuity

        Returns:
            Dict with response and metadata
        """
        from agentdeploy.core.runtime import AgentRuntime
        runtime = AgentRuntime(self.config)
        return runtime.invoke(message, session_id=session_id, agent_fn=self.fn)

    def __repr__(self) -> str:
        return f"AgentFunction(name='{self.name}', model='{self.config.model}')"


def agent(
    name: str,
    model: str = "bedrock/claude-3-sonnet",
    system_prompt: str = "You are a helpful assistant.",
    tools: Optional[List[Any]] = None,
    max_iterations: int = 10,
    timeout_seconds: int = 60,
    memory_strategy: str = "sliding_window",
    **kwargs,
) -> Callable:
    """Decorator to turn a function into a deployable agent.

    Usage:
        @agent(name="my-agent", model="bedrock/claude-3-sonnet", tools=[search_kb])
        def my_agent(message: str, session: Session) -> str:
            pass

    Args:
        name: Agent name (used for deployment and routing)
        model: LLM model identifier
        system_prompt: System prompt for the agent
        tools: List of Tool-decorated functions
        max_iterations: Max reasoning loops
        timeout_seconds: Max execution time
        memory_strategy: Session memory strategy
    """
    def decorator(fn: Callable) -> AgentFunction:
        config = AgentConfig(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            memory_strategy=memory_strategy,
            metadata=kwargs,
        )
        return AgentFunction(fn, config)
    return decorator
