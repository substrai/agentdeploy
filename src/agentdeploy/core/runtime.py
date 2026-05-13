"""Agent runtime - handles the request lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentdeploy.core.agent import AgentConfig
from agentdeploy.session.manager import Session, SessionManager
from agentdeploy.tools.registry import ToolRegistry, ToolCallResult
from agentdeploy.cost.enforcer import CostEnforcer, CostBudget


@dataclass
class InvocationResult:
    """Result of an agent invocation."""

    response: str
    session_id: str
    success: bool = True
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "tool_calls": self.tool_calls,
            "turn_count": self.turn_count,
            "error": self.error,
        }


class AgentRuntime:
    """Handles the full agent request lifecycle.

    Request flow:
    1. Load/create session
    2. Check cost budget
    3. Execute agent function
    4. Track tool calls
    5. Record cost
    6. Save session
    7. Return result with metadata

    Usage:
        runtime = AgentRuntime(config)
        result = runtime.invoke("Hello!", session_id="sess-123")
    """

    def __init__(
        self,
        config: AgentConfig,
        session_manager: Optional[SessionManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        cost_enforcer: Optional[CostEnforcer] = None,
    ):
        self.config = config
        self.session_manager = session_manager or SessionManager(ttl_hours=config.memory_ttl_hours)
        self.tool_registry = tool_registry or ToolRegistry()
        self.cost_enforcer = cost_enforcer or CostEnforcer()
        self._invocation_log: List[InvocationResult] = []

        # Register tools
        for tool in config.tools:
            if hasattr(tool, "definition"):
                self.tool_registry.register(tool)

    def invoke(
        self,
        message: str,
        session_id: Optional[str] = None,
        tenant_id: str = "default",
        agent_fn: Optional[Callable] = None,
    ) -> InvocationResult:
        """Invoke the agent with full lifecycle management.

        Args:
            message: User message
            session_id: Optional session ID
            tenant_id: Tenant making the request
            agent_fn: Agent function to call

        Returns:
            InvocationResult with response and metadata
        """
        start_time = time.time()

        # 1. Load/create session
        session = self.session_manager.get_or_create(
            session_id, self.config.name, tenant_id
        )

        # 2. Check cost budget
        cost_check = self.cost_enforcer.check_request(
            session_id=session.session_id, estimated_cost=0.01
        )
        if not cost_check.allowed:
            return InvocationResult(
                response="",
                session_id=session.session_id,
                success=False,
                error=cost_check.message,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # 3. Add user message to session
        session.add_message("user", message)

        # 4. Execute agent
        try:
            if agent_fn:
                response = agent_fn(message, session)
                if response is None:
                    # Default behavior: echo with context
                    response = f"[{self.config.name}] Received: {message}"
            else:
                response = f"[{self.config.name}] Received: {message}"
        except Exception as e:
            return InvocationResult(
                response="",
                session_id=session.session_id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

        # 5. Add assistant response to session
        session.add_message("assistant", response)

        # 6. Calculate cost (estimate)
        input_tokens = len(message) // 4
        output_tokens = len(response) // 4
        cost = self._estimate_cost(input_tokens, output_tokens)

        # 7. Record cost
        session.total_tokens += input_tokens + output_tokens
        session.total_cost += cost
        self.cost_enforcer.record_cost(session.session_id, cost, input_tokens + output_tokens)

        # 8. Save session
        self.session_manager.save(session)

        latency_ms = (time.time() - start_time) * 1000

        result = InvocationResult(
            response=response,
            session_id=session.session_id,
            success=True,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(cost, 8),
            tool_calls=[r.__dict__ for r in self.tool_registry.call_log[-5:]],
            turn_count=session.turn_count,
        )

        self._invocation_log.append(result)
        return result

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.session_manager.get(session_id)

    @property
    def invocation_history(self) -> List[InvocationResult]:
        return self._invocation_log

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model."""
        pricing = {
            "bedrock/claude-3-haiku": (0.00025, 0.00125),
            "bedrock/claude-3-sonnet": (0.003, 0.015),
            "bedrock/claude-3-opus": (0.015, 0.075),
            "openai/gpt-4o-mini": (0.00015, 0.0006),
            "openai/gpt-4o": (0.005, 0.015),
        }
        input_price, output_price = pricing.get(self.config.model, (0.001, 0.002))
        return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
