"""Tests for observability/tracing."""

from agentdeploy.observability.tracer import AgentTracer, TraceEventType, RequestTrace


class TestAgentTracer:
    def setup_method(self):
        self.tracer = AgentTracer()

    def test_start_trace(self):
        trace = self.tracer.start_trace("my-agent", "tenant-1", "sess-1")
        assert trace.agent_name == "my-agent"
        assert trace.tenant_id == "tenant-1"
        assert len(trace.entries) == 1  # REQUEST_START

    def test_add_events(self):
        trace = self.tracer.start_trace("agent", "t1", "s1")
        trace.add_event(TraceEventType.SESSION_LOAD, duration_ms=2.0)
        trace.add_event(TraceEventType.LLM_CALL, duration_ms=500.0)
        trace.add_event(TraceEventType.TOOL_CALL, duration_ms=50.0)
        assert len(trace.entries) == 4
        assert trace.tool_calls == 1

    def test_complete_trace(self):
        trace = self.tracer.start_trace("agent", "t1", "s1")
        trace.add_event(TraceEventType.LLM_CALL, duration_ms=100)
        trace.complete(success=True)
        assert trace.is_complete
        assert trace.success
        assert trace.duration_ms > 0

    def test_get_trace(self):
        trace = self.tracer.start_trace("agent", "t1", "s1")
        retrieved = self.tracer.get_trace(trace.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id

    def test_get_recent(self):
        for i in range(5):
            t = self.tracer.start_trace(f"agent-{i}", "t1", "s1")
            t.complete()
        recent = self.tracer.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_by_tenant(self):
        self.tracer.start_trace("agent", "tenant-a", "s1").complete()
        self.tracer.start_trace("agent", "tenant-b", "s2").complete()
        self.tracer.start_trace("agent", "tenant-a", "s3").complete()
        traces = self.tracer.get_by_tenant("tenant-a")
        assert len(traces) == 2

    def test_get_failed(self):
        t1 = self.tracer.start_trace("agent", "t1", "s1")
        t1.complete(success=True)
        t2 = self.tracer.start_trace("agent", "t1", "s2")
        t2.complete(success=False)
        failed = self.tracer.get_failed()
        assert len(failed) == 1
        assert not failed[0].success

    def test_metrics(self):
        for i in range(10):
            t = self.tracer.start_trace("agent", "t1", f"s{i}")
            t.total_cost = 0.001
            t.add_event(TraceEventType.TOOL_CALL, duration_ms=10)
            t.complete()
        metrics = self.tracer.get_metrics()
        assert metrics["total"] == 10
        assert metrics["success_rate"] == 1.0
        assert metrics["total_cost"] == 0.01
        assert metrics["total_tool_calls"] == 10

    def test_trace_summary(self):
        trace = self.tracer.start_trace("agent", "t1", "s1")
        trace.add_event(TraceEventType.LLM_CALL, duration_ms=500)
        trace.complete()
        summary = trace.summary()
        assert "agent" in summary
        assert "llm_call" in summary

    def test_trace_to_dict(self):
        trace = self.tracer.start_trace("agent", "t1", "s1")
        trace.complete()
        d = trace.to_dict()
        assert "trace_id" in d
        assert "events" in d
        assert d["agent_name"] == "agent"
