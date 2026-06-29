"""
AgentDeploy - Zero-to-Production AI Agent Deployment Framework

Takes any agent definition (LangChain, CrewAI, custom) and deploys it
as a production-grade serverless API with auth, scaling, monitoring,
cost controls, multi-tenancy, and session management.

Usage:
    from agentdeploy import agent, Tool, Session

    @agent(name="my-agent", model="bedrock/claude-3-sonnet")
    def my_agent(message: str, session: Session) -> str:
        return "Hello!"
"""

__version__ = "1.4.0"

from agentdeploy.core.agent import agent, AgentConfig
from agentdeploy.core.runtime import AgentRuntime, InvocationResult
from agentdeploy.session.manager import Session, SessionManager
from agentdeploy.tools.registry import Tool, ToolRegistry, ToolPermission
from agentdeploy.adapters.base import BaseAdapter
from agentdeploy.cost.enforcer import CostEnforcer, CostBudget

__all__ = [
    "agent",
    "AgentConfig",
    "AgentRuntime",
    "InvocationResult",
    "Session",
    "SessionManager",
    "Tool",
    "ToolRegistry",
    "ToolPermission",
    "BaseAdapter",
    "CostEnforcer",
    "CostBudget",
]
