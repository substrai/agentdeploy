"""Tool registry and @Tool decorator."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ToolPermission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    EXECUTE = "execute"


@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""

    name: str
    description: str
    fn: Callable
    permissions: List[ToolPermission] = field(default_factory=lambda: [ToolPermission.READ])
    timeout_seconds: int = 30
    requires_approval: bool = False
    version: str = "1.0.0"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "timeout_seconds": self.timeout_seconds,
            "requires_approval": self.requires_approval,
            "version": self.version,
        }


@dataclass
class ToolCallResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ToolFunction:
    """Wrapper around a Tool-decorated function."""

    def __init__(self, fn: Callable, definition: ToolDefinition):
        self.fn = fn
        self.definition = definition
        self.name = definition.name
        self.description = definition.description
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)

    def execute(self, **kwargs) -> ToolCallResult:
        """Execute the tool with tracking."""
        start = time.time()
        try:
            output = self.fn(**kwargs)
            return ToolCallResult(
                tool_name=self.name,
                success=True,
                output=output,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"


def Tool(
    description: str = "",
    permissions: Optional[List[ToolPermission]] = None,
    timeout_seconds: int = 30,
    requires_approval: bool = False,
    version: str = "1.0.0",
) -> Callable:
    """Decorator to register a function as an agent tool.

    Usage:
        @Tool(description="Search the knowledge base")
        def search_kb(query: str) -> list:
            return results
    """
    def decorator(fn: Callable) -> ToolFunction:
        definition = ToolDefinition(
            name=fn.__name__,
            description=description or fn.__doc__ or "",
            fn=fn,
            permissions=permissions or [ToolPermission.READ],
            timeout_seconds=timeout_seconds,
            requires_approval=requires_approval,
            version=version,
        )
        return ToolFunction(fn, definition)
    return decorator


class ToolRegistry:
    """Central registry of all available tools.

    Usage:
        registry = ToolRegistry()
        registry.register(search_kb)
        result = registry.execute("search_kb", query="hello")
    """

    def __init__(self):
        self._tools: Dict[str, ToolFunction] = {}
        self._call_log: List[ToolCallResult] = []

    def register(self, tool: ToolFunction) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolFunction]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> ToolCallResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolCallResult(tool_name=name, success=False, error=f"Tool '{name}' not found")
        result = tool.execute(**kwargs)
        self._call_log.append(result)
        return result

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return [t.definition for t in self._tools.values()]

    def is_allowed(self, tool_name: str, allowed_tools: Optional[List[str]] = None, denied_tools: Optional[List[str]] = None) -> bool:
        """Check if a tool is allowed for a tenant."""
        if denied_tools and tool_name in denied_tools:
            return False
        if allowed_tools and allowed_tools != ["all"] and tool_name not in allowed_tools:
            return False
        return True

    @property
    def call_log(self) -> List[ToolCallResult]:
        return self._call_log

    @property
    def tool_count(self) -> int:
        return len(self._tools)
