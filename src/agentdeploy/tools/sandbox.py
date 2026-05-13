"""Tool sandboxing - per-tenant permissions and audit trail."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentdeploy.tools.registry import ToolFunction, ToolCallResult, ToolPermission


@dataclass
class ToolAuditEntry:
    """Audit log entry for a tool call."""

    tool_name: str
    tenant_id: str
    session_id: str
    inputs: Dict[str, Any]
    output: Any = None
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    blocked: bool = False
    block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "success": self.success,
            "blocked": self.blocked,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class SandboxCheckResult:
    """Result of a sandbox permission check."""

    allowed: bool
    tool_name: str
    tenant_id: str
    reason: str
    requires_approval: bool = False


class ToolSandbox:
    """Enforces per-tenant tool permissions with audit trail.

    Usage:
        sandbox = ToolSandbox()
        sandbox.set_permissions("free-tier", allowed=["search_kb"], denied=["delete_data"])
        
        check = sandbox.check_permission("search_kb", "free-tier")
        if check.allowed:
            result = sandbox.execute("search_kb", tenant_id="free-tier", query="hello")
    """

    def __init__(self):
        self._permissions: Dict[str, Dict[str, Any]] = {}  # tenant -> {allowed, denied}
        self._audit_log: List[ToolAuditEntry] = []
        self._tools: Dict[str, ToolFunction] = {}
        self._approval_queue: List[Dict[str, Any]] = []

    def register_tool(self, tool: ToolFunction) -> None:
        """Register a tool in the sandbox."""
        self._tools[tool.name] = tool

    def set_permissions(
        self,
        tenant_id: str,
        allowed: Optional[List[str]] = None,
        denied: Optional[List[str]] = None,
    ) -> None:
        """Set tool permissions for a tenant."""
        self._permissions[tenant_id] = {
            "allowed": allowed or ["all"],
            "denied": denied or [],
        }

    def check_permission(self, tool_name: str, tenant_id: str) -> SandboxCheckResult:
        """Check if a tenant can use a tool.

        Args:
            tool_name: Tool to check
            tenant_id: Tenant requesting access

        Returns:
            SandboxCheckResult with allow/deny
        """
        perms = self._permissions.get(tenant_id, {"allowed": ["all"], "denied": []})

        # Check denied list first
        if tool_name in perms["denied"]:
            return SandboxCheckResult(
                allowed=False, tool_name=tool_name, tenant_id=tenant_id,
                reason=f"Tool '{tool_name}' is denied for tenant '{tenant_id}'",
            )

        # Check allowed list
        if perms["allowed"] != ["all"] and tool_name not in perms["allowed"]:
            return SandboxCheckResult(
                allowed=False, tool_name=tool_name, tenant_id=tenant_id,
                reason=f"Tool '{tool_name}' not in allowed list for tenant '{tenant_id}'",
            )

        # Check if tool requires approval
        tool = self._tools.get(tool_name)
        if tool and tool.definition.requires_approval:
            return SandboxCheckResult(
                allowed=False, tool_name=tool_name, tenant_id=tenant_id,
                reason="Tool requires human approval",
                requires_approval=True,
            )

        return SandboxCheckResult(
            allowed=True, tool_name=tool_name, tenant_id=tenant_id, reason="Allowed",
        )

    def execute(
        self,
        tool_name: str,
        tenant_id: str,
        session_id: str = "",
        **kwargs,
    ) -> ToolCallResult:
        """Execute a tool with sandbox enforcement.

        Args:
            tool_name: Tool to execute
            tenant_id: Tenant making the call
            session_id: Current session
            **kwargs: Tool arguments

        Returns:
            ToolCallResult (blocked if not permitted)
        """
        # Check permission
        check = self.check_permission(tool_name, tenant_id)

        if not check.allowed:
            entry = ToolAuditEntry(
                tool_name=tool_name, tenant_id=tenant_id, session_id=session_id,
                inputs=kwargs, blocked=True, block_reason=check.reason,
            )
            self._audit_log.append(entry)
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"Permission denied: {check.reason}",
            )

        # Execute tool
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolCallResult(tool_name=tool_name, success=False, error=f"Tool '{tool_name}' not found")

        start = time.time()
        try:
            output = tool.fn(**kwargs)
            latency = (time.time() - start) * 1000
            entry = ToolAuditEntry(
                tool_name=tool_name, tenant_id=tenant_id, session_id=session_id,
                inputs=kwargs, output=str(output)[:200], success=True, latency_ms=latency,
            )
            self._audit_log.append(entry)
            return ToolCallResult(tool_name=tool_name, success=True, output=output, latency_ms=latency)
        except Exception as e:
            latency = (time.time() - start) * 1000
            entry = ToolAuditEntry(
                tool_name=tool_name, tenant_id=tenant_id, session_id=session_id,
                inputs=kwargs, success=False, error=str(e), latency_ms=latency,
            )
            self._audit_log.append(entry)
            return ToolCallResult(tool_name=tool_name, success=False, error=str(e), latency_ms=latency)

    def get_audit_log(
        self, tenant_id: Optional[str] = None, tool_name: Optional[str] = None, limit: int = 100
    ) -> List[ToolAuditEntry]:
        """Get audit log with optional filters."""
        entries = self._audit_log
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        return entries[-limit:]

    def get_blocked_calls(self, tenant_id: Optional[str] = None) -> List[ToolAuditEntry]:
        """Get all blocked tool calls."""
        entries = [e for e in self._audit_log if e.blocked]
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        total = len(self._audit_log)
        blocked = sum(1 for e in self._audit_log if e.blocked)
        failed = sum(1 for e in self._audit_log if not e.success and not e.blocked)
        return {
            "total_calls": total,
            "blocked": blocked,
            "failed": failed,
            "success": total - blocked - failed,
            "block_rate": blocked / total if total > 0 else 0,
        }
