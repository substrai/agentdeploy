"""Tool permission matrix per tenant.

Defines which tenants can use which tools with granular allow/deny rules,
rate limits, and audit logging of unauthorized attempts.

Usage:
    from agentdeploy.tools.permissions import ToolPermissionMatrix

    matrix = ToolPermissionMatrix()
    matrix.grant(tenant_id="acme", tool_name="search_kb", actions=["read"])
    matrix.deny(tenant_id="acme", tool_name="delete_records")

    result = matrix.check("acme", "search_kb", "read")
    if result.allowed:
        # Execute tool
        pass
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PermissionLevel(str, Enum):
    """Level of access granted to a tenant for a tool."""

    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"  # Allowed with constraints


@dataclass
class PermissionRule:
    """A single permission rule for (tenant, tool)."""

    tenant_id: str
    tool_name: str
    level: PermissionLevel
    actions: List[str] = field(default_factory=lambda: ["*"])
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_calls_per_hour: Optional[int] = None
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def allows_action(self, action: str) -> bool:
        if self.level == PermissionLevel.DENY:
            return False
        return "*" in self.actions or action in self.actions


@dataclass
class PermissionCheckResult:
    """Result of a permission check."""

    allowed: bool
    tenant_id: str
    tool_name: str
    action: str
    rule: Optional[PermissionRule]
    reason: str
    check_time: float = field(default_factory=time.time)


@dataclass
class UnauthorizedAttempt:
    """Record of an unauthorized tool access attempt."""

    tenant_id: str
    tool_name: str
    action: str
    reason: str
    timestamp: float = field(default_factory=time.time)


class ToolPermissionMatrix:
    """Manages per-tenant tool permissions with audit logging.

    Supports explicit allow/deny rules, wildcard tool names,
    action-level permissions, and expiring rules.

    Args:
        default_deny: If True (default), deny all tools unless explicitly granted.
        audit_unauthorized: Whether to log unauthorized attempts.
    """

    def __init__(
        self,
        default_deny: bool = True,
        audit_unauthorized: bool = True,
    ):
        self._default_deny = default_deny
        self._audit_unauthorized = audit_unauthorized
        self._rules: List[PermissionRule] = []
        self._audit_log: List[UnauthorizedAttempt] = []
        self._call_counts: Dict[str, int] = {}  # "tenant::tool" → count

    def grant(
        self,
        tenant_id: str,
        tool_name: str,
        actions: Optional[List[str]] = None,
        max_calls_per_hour: Optional[int] = None,
        expires_at: Optional[float] = None,
        notes: str = "",
    ) -> PermissionRule:
        """Grant a tenant access to a tool.

        Args:
            tenant_id: Tenant identifier.
            tool_name: Tool name (use '*' for all tools).
            actions: Allowed actions (default: all via ['*']).
            max_calls_per_hour: Optional rate limit.
            expires_at: Unix timestamp for expiry.
            notes: Audit notes.

        Returns:
            The created PermissionRule.
        """
        rule = PermissionRule(
            tenant_id=tenant_id,
            tool_name=tool_name,
            level=PermissionLevel.ALLOW,
            actions=actions or ["*"],
            max_calls_per_hour=max_calls_per_hour,
            expires_at=expires_at,
            notes=notes,
        )
        self._rules.append(rule)
        return rule

    def deny(
        self,
        tenant_id: str,
        tool_name: str,
        notes: str = "",
    ) -> PermissionRule:
        """Explicitly deny a tenant access to a tool.

        Deny rules take precedence over allow rules.

        Args:
            tenant_id: Tenant identifier.
            tool_name: Tool name.
            notes: Audit notes.

        Returns:
            The created PermissionRule.
        """
        rule = PermissionRule(
            tenant_id=tenant_id,
            tool_name=tool_name,
            level=PermissionLevel.DENY,
            notes=notes,
        )
        self._rules.append(rule)
        return rule

    def check(
        self,
        tenant_id: str,
        tool_name: str,
        action: str = "*",
    ) -> PermissionCheckResult:
        """Check if a tenant is permitted to use a tool.

        Evaluation order:
        1. Check for explicit DENY rules (highest priority)
        2. Check for explicit ALLOW rules
        3. Fall back to default (deny if default_deny=True)

        Args:
            tenant_id: Tenant identifier.
            tool_name: Tool being accessed.
            action: Action being performed.

        Returns:
            PermissionCheckResult with allow/deny decision.
        """
        # 1. Check explicit DENY rules first
        for rule in self._rules:
            if not self._rule_applies(rule, tenant_id, tool_name):
                continue
            if rule.level == PermissionLevel.DENY and not rule.is_expired:
                result = PermissionCheckResult(
                    allowed=False,
                    tenant_id=tenant_id,
                    tool_name=tool_name,
                    action=action,
                    rule=rule,
                    reason=f"Explicit DENY rule for {tenant_id}::{tool_name}",
                )
                self._record_unauthorized(result)
                return result

        # 2. Check ALLOW rules
        for rule in self._rules:
            if not self._rule_applies(rule, tenant_id, tool_name):
                continue
            if rule.level != PermissionLevel.ALLOW:
                continue
            if rule.is_expired:
                continue
            if not rule.allows_action(action):
                continue

            # Check rate limit
            if rule.max_calls_per_hour is not None:
                key = f"{tenant_id}::{tool_name}"
                count = self._call_counts.get(key, 0)
                if count >= rule.max_calls_per_hour:
                    result = PermissionCheckResult(
                        allowed=False,
                        tenant_id=tenant_id,
                        tool_name=tool_name,
                        action=action,
                        rule=rule,
                        reason=f"Rate limit exceeded ({count}/{rule.max_calls_per_hour} per hour)",
                    )
                    self._record_unauthorized(result)
                    return result
                self._call_counts[key] = count + 1

            return PermissionCheckResult(
                allowed=True,
                tenant_id=tenant_id,
                tool_name=tool_name,
                action=action,
                rule=rule,
                reason="Allowed by explicit rule",
            )

        # 3. Default
        allowed = not self._default_deny
        reason = "Default allow" if allowed else f"No permission rule for {tenant_id}::{tool_name}"

        result = PermissionCheckResult(
            allowed=allowed,
            tenant_id=tenant_id,
            tool_name=tool_name,
            action=action,
            rule=None,
            reason=reason,
        )

        if not allowed:
            self._record_unauthorized(result)

        return result

    def get_tenant_tools(self, tenant_id: str) -> List[str]:
        """Get all tools a tenant is explicitly allowed to use."""
        return [
            rule.tool_name
            for rule in self._rules
            if rule.tenant_id == tenant_id
            and rule.level == PermissionLevel.ALLOW
            and not rule.is_expired
            and rule.tool_name != "*"
        ]

    def get_tenant_permissions(self, tenant_id: str) -> List[PermissionRule]:
        """Get all permission rules for a tenant."""
        return [r for r in self._rules if r.tenant_id == tenant_id]

    def revoke(self, tenant_id: str, tool_name: str) -> int:
        """Remove all permission rules for a (tenant, tool) pair.

        Returns:
            Number of rules removed.
        """
        before = len(self._rules)
        self._rules = [
            r for r in self._rules
            if not (r.tenant_id == tenant_id and r.tool_name == tool_name)
        ]
        return before - len(self._rules)

    def get_audit_log(self) -> List[UnauthorizedAttempt]:
        """Get all unauthorized access attempts."""
        return self._audit_log.copy()

    def get_audit_log_for_tenant(self, tenant_id: str) -> List[UnauthorizedAttempt]:
        """Get unauthorized attempts for a specific tenant."""
        return [a for a in self._audit_log if a.tenant_id == tenant_id]

    def reset_call_counts(self) -> None:
        """Reset all rate limit counters (call at start of each hour)."""
        self._call_counts.clear()

    def _rule_applies(
        self,
        rule: PermissionRule,
        tenant_id: str,
        tool_name: str,
    ) -> bool:
        """Check if a rule applies to this (tenant, tool) combination."""
        tenant_match = rule.tenant_id == tenant_id or rule.tenant_id == "*"
        tool_match = rule.tool_name == tool_name or rule.tool_name == "*"
        return tenant_match and tool_match

    def _record_unauthorized(self, result: PermissionCheckResult) -> None:
        """Log an unauthorized attempt."""
        if self._audit_unauthorized:
            self._audit_log.append(UnauthorizedAttempt(
                tenant_id=result.tenant_id,
                tool_name=result.tool_name,
                action=result.action,
                reason=result.reason,
            ))
