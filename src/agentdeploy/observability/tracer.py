"""Agent tracing - full request lifecycle visibility."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TraceEventType(Enum):
    REQUEST_START = "request_start"
    AUTH_CHECK = "auth_check"
    RATE_LIMIT_CHECK = "rate_limit_check"
    SESSION_LOAD = "session_load"
    BUDGET_CHECK = "budget_check"
    AGENT_START = "agent_start"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_END = "agent_end"
    SESSION_SAVE = "session_save"
    REQUEST_END = "request_end"
    ERROR = "error"


@dataclass
class TraceEntry:
    """A single event in a request trace."""

    event_type: TraceEventType
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class RequestTrace:
    """Complete trace for a single agent request."""

    trace_id: str
    agent_name: str
    tenant_id: str
    session_id: str
    entries: List[TraceEntry] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    tool_calls: int = 0
    success: bool = True

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None

    def add_event(self, event_type: TraceEventType, duration_ms: float = 0.0, **kwargs) -> None:
        self.entries.append(TraceEntry(
            event_type=event_type, duration_ms=duration_ms,
            metadata=kwargs.get("metadata", {}),
            success=kwargs.get("success", True),
            error=kwargs.get("error"),
        ))
        if event_type == TraceEventType.TOOL_CALL:
            self.tool_calls += 1

    def complete(self, success: bool = True) -> None:
        self.end_time = time.time()
        self.success = success
        self.add_event(TraceEventType.REQUEST_END, metadata={"total_ms": self.duration_ms})

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        lines = [
            f"{status} Trace: {self.trace_id}",
            f"  Agent: {self.agent_name} | Tenant: {self.tenant_id}",
            f"  Duration: {self.duration_ms:.0f}ms | Tools: {self.tool_calls} | Cost: ${self.total_cost:.6f}",
            f"  Events ({len(self.entries)}):",
        ]
        for entry in self.entries:
            icon = "✓" if entry.success else "✗"
            lines.append(f"    {icon} {entry.event_type.value} ({entry.duration_ms:.1f}ms)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "tool_calls": self.tool_calls,
            "success": self.success,
            "events": [e.to_dict() for e in self.entries],
        }


class AgentTracer:
    """Traces agent request lifecycles.

    Usage:
        tracer = AgentTracer()
        trace = tracer.start_trace("my-agent", "tenant-1", "sess-1")
        trace.add_event(TraceEventType.LLM_CALL, duration_ms=500)
        trace.complete()
        print(trace.summary())
    """

    def __init__(self, max_traces: int = 10000):
        self._traces: Dict[str, RequestTrace] = {}
        self._max_traces = max_traces

    def start_trace(self, agent_name: str, tenant_id: str, session_id: str) -> RequestTrace:
        """Start a new request trace."""
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        trace = RequestTrace(
            trace_id=trace_id, agent_name=agent_name,
            tenant_id=tenant_id, session_id=session_id,
        )
        trace.add_event(TraceEventType.REQUEST_START)
        self._traces[trace_id] = trace
        self._cleanup()
        return trace

    def get_trace(self, trace_id: str) -> Optional[RequestTrace]:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def get_recent(self, limit: int = 20, agent_name: Optional[str] = None) -> List[RequestTrace]:
        """Get recent traces."""
        traces = list(self._traces.values())
        if agent_name:
            traces = [t for t in traces if t.agent_name == agent_name]
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    def get_by_tenant(self, tenant_id: str, limit: int = 50) -> List[RequestTrace]:
        """Get traces for a specific tenant."""
        traces = [t for t in self._traces.values() if t.tenant_id == tenant_id]
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    def get_failed(self, limit: int = 50) -> List[RequestTrace]:
        """Get failed traces."""
        failed = [t for t in self._traces.values() if not t.success and t.is_complete]
        failed.sort(key=lambda t: t.start_time, reverse=True)
        return failed[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics from traces."""
        completed = [t for t in self._traces.values() if t.is_complete]
        if not completed:
            return {"total": 0}

        durations = [t.duration_ms for t in completed]
        costs = [t.total_cost for t in completed]
        success_count = sum(1 for t in completed if t.success)

        return {
            "total": len(completed),
            "success_rate": success_count / len(completed),
            "avg_duration_ms": sum(durations) / len(durations),
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "total_cost": sum(costs),
            "avg_cost": sum(costs) / len(costs) if costs else 0,
            "total_tool_calls": sum(t.tool_calls for t in completed),
        }

    def _cleanup(self) -> None:
        if len(self._traces) > self._max_traces:
            sorted_traces = sorted(self._traces.items(), key=lambda x: x[1].start_time)
            for trace_id, _ in sorted_traces[:len(self._traces) - self._max_traces]:
                del self._traces[trace_id]
