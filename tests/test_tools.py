"""Tests for tool registry."""

from agentdeploy.tools.registry import Tool, ToolRegistry, ToolPermission, ToolFunction


@Tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

@Tool(description="Divide numbers", permissions=[ToolPermission.WRITE])
def divide(a: int, b: int) -> float:
    return a / b

@Tool(description="Failing tool")
def fail_tool() -> str:
    raise RuntimeError("Tool crashed")


class TestToolDecorator:
    def test_creates_tool_function(self):
        assert isinstance(add, ToolFunction)
        assert add.name == "add"
        assert add.description == "Add two numbers"

    def test_direct_call(self):
        assert add(2, 3) == 5

    def test_execute_success(self):
        result = add.execute(a=2, b=3)
        assert result.success
        assert result.output == 5
        assert result.latency_ms >= 0

    def test_execute_failure(self):
        result = fail_tool.execute()
        assert not result.success
        assert "crashed" in result.error


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(add)
        assert registry.get("add") is not None
        assert registry.tool_count == 1

    def test_execute_by_name(self):
        registry = ToolRegistry()
        registry.register(add)
        result = registry.execute("add", a=5, b=3)
        assert result.success
        assert result.output == 8

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert not result.success
        assert "not found" in result.error

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(add)
        registry.register(divide)
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_is_allowed(self):
        registry = ToolRegistry()
        assert registry.is_allowed("add", allowed_tools=["add", "divide"])
        assert not registry.is_allowed("add", denied_tools=["add"])
        assert not registry.is_allowed("add", allowed_tools=["divide"])
        assert registry.is_allowed("add", allowed_tools=["all"])

    def test_call_log(self):
        registry = ToolRegistry()
        registry.register(add)
        registry.execute("add", a=1, b=2)
        registry.execute("add", a=3, b=4)
        assert len(registry.call_log) == 2
