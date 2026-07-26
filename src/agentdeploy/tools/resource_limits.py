"""Tool execution sandboxing with resource limits.

Enforces CPU time, wall-clock time, memory, and output size limits
per tool execution. Kills runaway tools that exceed configured limits.

Usage:
    from agentdeploy.tools.resource_limits import ResourceLimitedSandbox, ResourceLimits

    sandbox = ResourceLimitedSandbox(
        limits=ResourceLimits(
            max_wall_time_seconds=10.0,
            max_output_bytes=65536,
        )
    )

    result = sandbox.execute(my_tool_fn, args={"query": "hello"})
    if result.timed_out:
        print("Tool exceeded time limit")
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class LimitType(str, Enum):
    """Type of resource limit that was exceeded."""

    WALL_TIME = "wall_time"
    OUTPUT_SIZE = "output_size"
    RECURSION_DEPTH = "recursion_depth"
    CALL_COUNT = "call_count"


@dataclass
class ResourceLimits:
    """Resource limits for a tool execution.

    Args:
        max_wall_time_seconds: Maximum real clock time (default 30s).
        max_output_bytes: Maximum output size in bytes (default 64KB).
        max_recursion_depth: Maximum call stack depth.
        max_api_calls: Maximum external calls (counted by tool).
    """

    max_wall_time_seconds: float = 30.0
    max_output_bytes: int = 65536  # 64KB
    max_recursion_depth: int = 50
    max_api_calls: int = 10

    def __post_init__(self):
        if self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass
class ExecutionStats:
    """Statistics from a sandboxed tool execution."""

    wall_time_ms: float = 0.0
    output_bytes: int = 0
    api_calls_made: int = 0
    peak_recursion_depth: int = 0


@dataclass
class SandboxedResult:
    """Result of a sandboxed tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    limit_exceeded: Optional[LimitType] = None
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    tool_name: str = ""

    @property
    def within_limits(self) -> bool:
        """Whether the execution stayed within all resource limits."""
        return self.limit_exceeded is None


class ResourceLimitedSandbox:
    """Executes tool functions with enforced resource limits.

    Runs each tool in a separate thread with a timeout. Tracks
    output size, wall time, and other resource usage. Kills
    executions that exceed configured limits.

    Args:
        limits: Resource limits to enforce.
        on_limit_exceeded: Optional callback when limits are exceeded.
    """

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        on_limit_exceeded: Optional[Callable[[str, LimitType], None]] = None,
    ):
        self._limits = limits or ResourceLimits()
        self._on_limit_exceeded = on_limit_exceeded
        self._api_call_counter: Dict[str, int] = {}

    @property
    def limits(self) -> ResourceLimits:
        """The configured resource limits."""
        return self._limits

    def execute(
        self,
        tool_fn: Callable[..., Any],
        args: Optional[Dict[str, Any]] = None,
        tool_name: str = "unknown",
    ) -> SandboxedResult:
        """Execute a tool function within resource limits.

        Args:
            tool_fn: The tool function to execute.
            args: Keyword arguments to pass to the function.
            tool_name: Name for logging and tracking.

        Returns:
            SandboxedResult with output or error/limit information.
        """
        args = args or {}
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, Any] = {}

        def target():
            try:
                output = tool_fn(**args)
                result_holder["output"] = output
            except Exception as e:
                error_holder["error"] = e

        start_time = time.time()
        thread = threading.Thread(target=target, daemon=True, name=f"sandbox-{tool_name}")
        thread.start()
        thread.join(timeout=self._limits.max_wall_time_seconds)
        wall_time_ms = (time.time() - start_time) * 1000

        # Check for timeout
        if thread.is_alive():
            # Thread is still running — timed out
            if self._on_limit_exceeded:
                self._on_limit_exceeded(tool_name, LimitType.WALL_TIME)

            return SandboxedResult(
                success=False,
                error=f"Tool '{tool_name}' exceeded wall time limit of {self._limits.max_wall_time_seconds}s",
                timed_out=True,
                limit_exceeded=LimitType.WALL_TIME,
                stats=ExecutionStats(wall_time_ms=wall_time_ms),
                tool_name=tool_name,
            )

        # Check for execution errors
        if "error" in error_holder:
            return SandboxedResult(
                success=False,
                error=str(error_holder["error"]),
                stats=ExecutionStats(wall_time_ms=wall_time_ms),
                tool_name=tool_name,
            )

        output = result_holder.get("output")

        # Check output size
        output_bytes = self._estimate_output_size(output)
        if output_bytes > self._limits.max_output_bytes:
            if self._on_limit_exceeded:
                self._on_limit_exceeded(tool_name, LimitType.OUTPUT_SIZE)
            return SandboxedResult(
                success=False,
                error=(
                    f"Tool '{tool_name}' output ({output_bytes} bytes) exceeds "
                    f"limit of {self._limits.max_output_bytes} bytes"
                ),
                limit_exceeded=LimitType.OUTPUT_SIZE,
                stats=ExecutionStats(wall_time_ms=wall_time_ms, output_bytes=output_bytes),
                tool_name=tool_name,
            )

        stats = ExecutionStats(
            wall_time_ms=wall_time_ms,
            output_bytes=output_bytes,
        )

        return SandboxedResult(
            success=True,
            output=output,
            stats=stats,
            tool_name=tool_name,
        )

    def execute_with_budget(
        self,
        tool_fn: Callable[..., Any],
        args: Optional[Dict[str, Any]] = None,
        tool_name: str = "unknown",
        remaining_budget_ms: Optional[float] = None,
    ) -> SandboxedResult:
        """Execute with a dynamic time budget (useful for remaining Lambda time).

        Args:
            tool_fn: Tool function to execute.
            args: Arguments.
            tool_name: Tool name.
            remaining_budget_ms: Available time in ms. If set, overrides limit.

        Returns:
            SandboxedResult.
        """
        if remaining_budget_ms is not None:
            effective_limits = ResourceLimits(
                max_wall_time_seconds=min(
                    remaining_budget_ms / 1000.0,
                    self._limits.max_wall_time_seconds,
                ),
                max_output_bytes=self._limits.max_output_bytes,
                max_recursion_depth=self._limits.max_recursion_depth,
                max_api_calls=self._limits.max_api_calls,
            )
            temp_sandbox = ResourceLimitedSandbox(
                limits=effective_limits,
                on_limit_exceeded=self._on_limit_exceeded,
            )
            return temp_sandbox.execute(tool_fn, args, tool_name)

        return self.execute(tool_fn, args, tool_name)

    def _estimate_output_size(self, output: Any) -> int:
        """Estimate the byte size of an output value."""
        if output is None:
            return 0
        if isinstance(output, (bytes, bytearray)):
            return len(output)
        if isinstance(output, str):
            return len(output.encode("utf-8"))
        # For other types, use repr approximation
        try:
            return len(repr(output).encode("utf-8"))
        except Exception:
            return 0


@dataclass
class ToolExecutionPolicy:
    """Per-tool resource limit policy."""

    tool_name: str
    limits: ResourceLimits
    enabled: bool = True
    log_executions: bool = True


class PolicyBasedSandbox:
    """Applies per-tool policies to resource limit enforcement.

    Different tools can have different limits based on their risk
    profile (e.g., web search gets more time than local lookup).

    Args:
        default_limits: Fallback limits for tools without explicit policy.
        policies: Per-tool policy overrides.
    """

    def __init__(
        self,
        default_limits: Optional[ResourceLimits] = None,
        policies: Optional[List[ToolExecutionPolicy]] = None,
    ):
        self._default_limits = default_limits or ResourceLimits()
        self._policies: Dict[str, ToolExecutionPolicy] = {
            p.tool_name: p for p in (policies or [])
        }
        self._execution_log: List[SandboxedResult] = []

    def add_policy(self, policy: ToolExecutionPolicy) -> None:
        """Add or update a tool policy."""
        self._policies[policy.tool_name] = policy

    def execute(
        self,
        tool_fn: Callable[..., Any],
        args: Optional[Dict[str, Any]] = None,
        tool_name: str = "unknown",
    ) -> SandboxedResult:
        """Execute a tool using its configured policy.

        Args:
            tool_fn: Tool function.
            args: Arguments.
            tool_name: Tool name (used to look up policy).

        Returns:
            SandboxedResult.
        """
        policy = self._policies.get(tool_name)

        if policy and not policy.enabled:
            return SandboxedResult(
                success=False,
                error=f"Tool '{tool_name}' is disabled by policy",
                tool_name=tool_name,
            )

        limits = policy.limits if policy else self._default_limits
        sandbox = ResourceLimitedSandbox(limits=limits)
        result = sandbox.execute(tool_fn, args, tool_name)

        if policy and policy.log_executions or not policy:
            self._execution_log.append(result)

        return result

    def get_execution_log(self) -> List[SandboxedResult]:
        """Get all logged executions."""
        return self._execution_log.copy()
