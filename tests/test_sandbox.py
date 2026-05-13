"""Tests for tool sandboxing."""

from agentdeploy.tools.registry import Tool, ToolPermission
from agentdeploy.tools.sandbox import ToolSandbox


@Tool(description="Search KB")
def search_kb(query: str) -> list:
    return [f"result for: {query}"]

@Tool(description="Delete data", permissions=[ToolPermission.ADMIN], requires_approval=True)
def delete_data(user_id: str) -> dict:
    return {"deleted": True}

@Tool(description="Failing tool")
def broken_tool() -> str:
    raise RuntimeError("Broken")


class TestToolSandbox:
    def setup_method(self):
        self.sandbox = ToolSandbox()
        self.sandbox.register_tool(search_kb)
        self.sandbox.register_tool(delete_data)
        self.sandbox.register_tool(broken_tool)

    def test_allowed_by_default(self):
        check = self.sandbox.check_permission("search_kb", "default")
        assert check.allowed

    def test_denied_tool(self):
        self.sandbox.set_permissions("restricted", denied=["search_kb"])
        check = self.sandbox.check_permission("search_kb", "restricted")
        assert not check.allowed
        assert "denied" in check.reason

    def test_not_in_allowed_list(self):
        self.sandbox.set_permissions("limited", allowed=["search_kb"])
        check = self.sandbox.check_permission("delete_data", "limited")
        assert not check.allowed

    def test_requires_approval(self):
        check = self.sandbox.check_permission("delete_data", "default")
        assert not check.allowed
        assert check.requires_approval

    def test_execute_allowed(self):
        result = self.sandbox.execute("search_kb", tenant_id="default", session_id="s1", query="hello")
        assert result.success
        assert "hello" in str(result.output)

    def test_execute_blocked(self):
        self.sandbox.set_permissions("blocked", denied=["search_kb"])
        result = self.sandbox.execute("search_kb", tenant_id="blocked", session_id="s1", query="hi")
        assert not result.success
        assert "Permission denied" in result.error

    def test_execute_tool_error(self):
        result = self.sandbox.execute("broken_tool", tenant_id="default", session_id="s1")
        assert not result.success
        assert "Broken" in result.error

    def test_audit_log(self):
        self.sandbox.execute("search_kb", tenant_id="t1", session_id="s1", query="test")
        log = self.sandbox.get_audit_log(tenant_id="t1")
        assert len(log) == 1
        assert log[0].tool_name == "search_kb"
        assert log[0].success

    def test_audit_log_blocked(self):
        self.sandbox.set_permissions("no-access", denied=["search_kb"])
        self.sandbox.execute("search_kb", tenant_id="no-access", session_id="s1", query="x")
        blocked = self.sandbox.get_blocked_calls()
        assert len(blocked) == 1
        assert blocked[0].blocked

    def test_stats(self):
        self.sandbox.execute("search_kb", tenant_id="t1", session_id="s1", query="a")
        self.sandbox.execute("search_kb", tenant_id="t1", session_id="s1", query="b")
        self.sandbox.set_permissions("blocked", denied=["search_kb"])
        self.sandbox.execute("search_kb", tenant_id="blocked", session_id="s1", query="c")
        stats = self.sandbox.get_stats()
        assert stats["total_calls"] == 3
        assert stats["blocked"] == 1
        assert stats["success"] == 2
