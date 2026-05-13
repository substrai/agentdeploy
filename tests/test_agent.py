"""Tests for the @agent decorator and runtime."""

import pytest
from agentdeploy.core.agent import agent, AgentConfig, AgentFunction
from agentdeploy.core.runtime import AgentRuntime, InvocationResult
from agentdeploy.session.manager import Session
from agentdeploy.tools.registry import Tool
from agentdeploy.cost.enforcer import CostEnforcer, CostBudget


@Tool(description="Echo tool")
def echo_tool(text: str) -> str:
    return f"echo: {text}"


@agent(name="test-agent", model="bedrock/claude-3-haiku", tools=[echo_tool])
def my_test_agent(message: str, session: Session) -> str:
    return f"Response to: {message}"


class TestAgentDecorator:
    def test_creates_agent_function(self):
        assert isinstance(my_test_agent, AgentFunction)
        assert my_test_agent.name == "test-agent"
        assert my_test_agent.config.model == "bedrock/claude-3-haiku"

    def test_agent_config(self):
        config = my_test_agent.config
        assert config.name == "test-agent"
        assert len(config.tools) == 1
        assert config.max_iterations == 10

    def test_direct_call(self):
        session = Session(session_id="test", agent_name="test")
        result = my_test_agent("Hello", session)
        assert "Hello" in result

    def test_invoke(self):
        result = my_test_agent.invoke("Hello world")
        assert result.response == "Response to: Hello world"
        assert result.success

    def test_config_to_dict(self):
        d = my_test_agent.config.to_dict()
        assert d["name"] == "test-agent"
        assert "echo_tool" in d["tools"]


class TestAgentRuntime:
    def setup_method(self):
        config = AgentConfig(name="runtime-test", model="bedrock/claude-3-haiku", tools=[echo_tool])
        self.runtime = AgentRuntime(config)

    def test_invoke_basic(self):
        result = self.runtime.invoke("Hello")
        assert result.success
        assert "Hello" in result.response
        assert result.session_id

    def test_invoke_creates_session(self):
        result = self.runtime.invoke("Hi", session_id="sess-1")
        assert result.session_id == "sess-1"
        session = self.runtime.get_session("sess-1")
        assert session is not None
        assert session.turn_count == 2  # user msg + assistant msg both increment

    def test_invoke_continues_session(self):
        self.runtime.invoke("First message", session_id="sess-2")
        self.runtime.invoke("Second message", session_id="sess-2")
        session = self.runtime.get_session("sess-2")
        assert session.turn_count == 4  # 2 user + 2 assistant
        assert len(session.messages) == 4

    def test_invoke_tracks_cost(self):
        result = self.runtime.invoke("Hello world")
        assert result.cost > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_invoke_with_custom_fn(self):
        def custom_fn(msg, sess):
            return f"Custom: {msg}"
        result = self.runtime.invoke("Test", agent_fn=custom_fn)
        assert result.response == "Custom: Test"

    def test_invoke_handles_error(self):
        def failing_fn(msg, sess):
            raise ValueError("Agent crashed")
        result = self.runtime.invoke("Test", agent_fn=failing_fn)
        assert not result.success
        assert "crashed" in result.error

    def test_cost_enforcement_blocks(self):
        budget = CostBudget(max_cost_per_session=0.00001)  # very low
        config = AgentConfig(name="budget-test", model="bedrock/claude-3-opus")
        enforcer = CostEnforcer(budget)
        runtime = AgentRuntime(config, cost_enforcer=enforcer)
        
        # First call records cost
        r1 = runtime.invoke("Hello " * 50, session_id="budget-sess")
        # Record enough cost to exceed
        enforcer.record_cost("budget-sess", 0.001)
        # Next call should be blocked
        r2 = runtime.invoke("Another", session_id="budget-sess")
        assert not r2.success
        assert "budget" in r2.error.lower() or "exceeded" in r2.error.lower()

    def test_invocation_history(self):
        self.runtime.invoke("A")
        self.runtime.invoke("B")
        assert len(self.runtime.invocation_history) == 2

    def test_result_to_dict(self):
        result = self.runtime.invoke("Test")
        d = result.to_dict()
        assert "response" in d
        assert "session_id" in d
        assert "cost" in d
