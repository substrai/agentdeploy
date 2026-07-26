"""Tests for tool execution sandboxing with resource limits."""

from __future__ import annotations

import time
import pytest

from agentdeploy.tools.resource_limits import (
    ExecutionStats,
    LimitType,
    PolicyBasedSandbox,
    ResourceLimitedSandbox,
    ResourceLimits,
    SandboxedResult,
    ToolExecutionPolicy,
)


def _fast_tool(**kwargs) -> str:
    return "result"


def _slow_tool(**kwargs) -> str:
    time.sleep(5.0)
    return "slow result"


def _large_output_tool(**kwargs) -> str:
    return "x" * 100_000  # 100KB


def _raising_tool(**kwargs) -> str:
    raise RuntimeError("Tool exploded")


class TestResourceLimitsInit:
    def test_default_limits(self):
        limits = ResourceLimits()
        assert limits.max_wall_time_seconds == 30.0
        assert limits.max_output_bytes == 65536

    def test_custom_limits(self):
        limits = ResourceLimits(max_wall_time_seconds=5.0, max_output_bytes=1024)
        assert limits.max_wall_time_seconds == 5.0
        assert limits.max_output_bytes == 1024

    def test_invalid_wall_time(self):
        with pytest.raises(ValueError):
            ResourceLimits(max_wall_time_seconds=0)

    def test_invalid_output_bytes(self):
        with pytest.raises(ValueError):
            ResourceLimits(max_output_bytes=0)


class TestSandboxedExecution:
    def test_successful_execution(self):
        sandbox = ResourceLimitedSandbox()
        result = sandbox.execute(_fast_tool, tool_name="fast")
        assert result.success is True
        assert result.output == "result"
        assert result.within_limits is True

    def test_timeout_detected(self):
        limits = ResourceLimits(max_wall_time_seconds=0.1)
        sandbox = ResourceLimitedSandbox(limits=limits)
        result = sandbox.execute(_slow_tool, tool_name="slow")
        assert result.success is False
        assert result.timed_out is True
        assert result.limit_exceeded == LimitType.WALL_TIME

    def test_output_size_limit(self):
        limits = ResourceLimits(max_output_bytes=1000)
        sandbox = ResourceLimitedSandbox(limits=limits)
        result = sandbox.execute(_large_output_tool, tool_name="large")
        assert result.success is False
        assert result.limit_exceeded == LimitType.OUTPUT_SIZE

    def test_tool_error_captured(self):
        sandbox = ResourceLimitedSandbox()
        result = sandbox.execute(_raising_tool, tool_name="raising")
        assert result.success is False
        assert result.timed_out is False
        assert "exploded" in result.error

    def test_stats_tracked(self):
        sandbox = ResourceLimitedSandbox()
        result = sandbox.execute(_fast_tool, tool_name="fast")
        assert result.stats.wall_time_ms >= 0
        assert result.stats.output_bytes > 0

    def test_args_passed_to_tool(self):
        def echo_tool(**kwargs) -> str:
            return kwargs.get("message", "")

        sandbox = ResourceLimitedSandbox()
        result = sandbox.execute(echo_tool, args={"message": "hello"}, tool_name="echo")
        assert result.output == "hello"

    def test_within_limits_false_on_timeout(self):
        limits = ResourceLimits(max_wall_time_seconds=0.1)
        sandbox = ResourceLimitedSandbox(limits=limits)
        result = sandbox.execute(_slow_tool, tool_name="slow")
        assert result.within_limits is False


class TestLimitExceededCallback:
    def test_callback_called_on_timeout(self):
        events = []

        def on_limit(tool_name, limit_type):
            events.append((tool_name, limit_type))

        limits = ResourceLimits(max_wall_time_seconds=0.1)
        sandbox = ResourceLimitedSandbox(limits=limits, on_limit_exceeded=on_limit)
        sandbox.execute(_slow_tool, tool_name="slow")

        assert len(events) == 1
        assert events[0] == ("slow", LimitType.WALL_TIME)

    def test_callback_called_on_output_limit(self):
        events = []

        def on_limit(tool_name, limit_type):
            events.append((tool_name, limit_type))

        limits = ResourceLimits(max_output_bytes=100)
        sandbox = ResourceLimitedSandbox(limits=limits, on_limit_exceeded=on_limit)
        sandbox.execute(_large_output_tool, tool_name="large")

        assert len(events) == 1
        assert events[0][1] == LimitType.OUTPUT_SIZE


class TestExecuteWithBudget:
    def test_budget_overrides_limit(self):
        limits = ResourceLimits(max_wall_time_seconds=30.0)
        sandbox = ResourceLimitedSandbox(limits=limits)

        result = sandbox.execute_with_budget(
            _slow_tool,
            tool_name="slow",
            remaining_budget_ms=100,  # 0.1 second budget
        )
        assert result.timed_out is True

    def test_no_budget_uses_default_limit(self):
        limits = ResourceLimits(max_wall_time_seconds=5.0)
        sandbox = ResourceLimitedSandbox(limits=limits)

        result = sandbox.execute_with_budget(
            _fast_tool,
            tool_name="fast",
            remaining_budget_ms=None,
        )
        assert result.success is True


class TestPolicyBasedSandbox:
    def test_default_limits_applied(self):
        sandbox = PolicyBasedSandbox(default_limits=ResourceLimits(max_wall_time_seconds=5.0))
        result = sandbox.execute(_fast_tool, tool_name="fast")
        assert result.success is True

    def test_per_tool_policy_applied(self):
        default = ResourceLimits(max_wall_time_seconds=30.0)
        tight_policy = ToolExecutionPolicy(
            tool_name="tight",
            limits=ResourceLimits(max_wall_time_seconds=0.1),
        )
        sandbox = PolicyBasedSandbox(default_limits=default, policies=[tight_policy])

        result = sandbox.execute(_slow_tool, tool_name="tight")
        assert result.timed_out is True

    def test_disabled_tool_returns_error(self):
        policy = ToolExecutionPolicy(
            tool_name="disabled-tool",
            limits=ResourceLimits(),
            enabled=False,
        )
        sandbox = PolicyBasedSandbox(policies=[policy])
        result = sandbox.execute(_fast_tool, tool_name="disabled-tool")
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_add_policy_at_runtime(self):
        sandbox = PolicyBasedSandbox()
        sandbox.add_policy(ToolExecutionPolicy(
            tool_name="new-tool",
            limits=ResourceLimits(max_wall_time_seconds=0.1),
        ))
        result = sandbox.execute(_slow_tool, tool_name="new-tool")
        assert result.timed_out is True

    def test_execution_log_populated(self):
        sandbox = PolicyBasedSandbox()
        sandbox.execute(_fast_tool, tool_name="t1")
        sandbox.execute(_fast_tool, tool_name="t2")
        log = sandbox.get_execution_log()
        assert len(log) == 2

    def test_execution_log_is_copy(self):
        sandbox = PolicyBasedSandbox()
        sandbox.execute(_fast_tool, tool_name="t1")
        log1 = sandbox.get_execution_log()
        log1.clear()
        log2 = sandbox.get_execution_log()
        assert len(log2) == 1  # Original not affected


class TestSandboxedResultProperties:
    def test_within_limits_true(self):
        result = SandboxedResult(success=True, output="ok", tool_name="t")
        assert result.within_limits is True

    def test_within_limits_false(self):
        result = SandboxedResult(
            success=False, limit_exceeded=LimitType.WALL_TIME, tool_name="t"
        )
        assert result.within_limits is False
