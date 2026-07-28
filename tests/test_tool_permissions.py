"""Tests for tool permission matrix per tenant."""

from __future__ import annotations

import time
import pytest

from agentdeploy.tools.permissions import (
    PermissionLevel,
    PermissionRule,
    ToolPermissionMatrix,
    UnauthorizedAttempt,
)


class TestToolPermissionMatrixInit:
    def test_default_deny(self):
        m = ToolPermissionMatrix(default_deny=True)
        result = m.check("tenant-a", "search_kb")
        assert result.allowed is False

    def test_default_allow(self):
        m = ToolPermissionMatrix(default_deny=False)
        result = m.check("tenant-a", "any_tool")
        assert result.allowed is True


class TestGrantPermission:
    def test_grant_allows_tool(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb")
        result = m.check("acme", "search_kb")
        assert result.allowed is True

    def test_grant_with_specific_action(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb", actions=["read"])
        assert m.check("acme", "search_kb", "read").allowed is True
        assert m.check("acme", "search_kb", "write").allowed is False

    def test_wildcard_action_allows_all(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb", actions=["*"])
        assert m.check("acme", "search_kb", "read").allowed is True
        assert m.check("acme", "search_kb", "delete").allowed is True

    def test_wildcard_tool_grants_all(self):
        m = ToolPermissionMatrix()
        m.grant("admin-tenant", "*")
        assert m.check("admin-tenant", "any_tool").allowed is True
        assert m.check("admin-tenant", "other_tool").allowed is True

    def test_grant_returns_rule(self):
        m = ToolPermissionMatrix()
        rule = m.grant("acme", "search_kb", notes="Granted for Q4")
        assert rule.level == PermissionLevel.ALLOW
        assert rule.notes == "Granted for Q4"


class TestDenyPermission:
    def test_deny_blocks_tool(self):
        m = ToolPermissionMatrix(default_deny=False)
        m.deny("acme", "delete_records")
        result = m.check("acme", "delete_records")
        assert result.allowed is False

    def test_deny_overrides_allow(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "dangerous_tool")
        m.deny("acme", "dangerous_tool")
        result = m.check("acme", "dangerous_tool")
        assert result.allowed is False

    def test_deny_only_affects_specified_tenant(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "tool-x")
        m.grant("other", "tool-x")
        m.deny("acme", "tool-x")
        assert m.check("acme", "tool-x").allowed is False
        assert m.check("other", "tool-x").allowed is True


class TestRateLimit:
    def test_rate_limit_enforced(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "api_call", max_calls_per_hour=3)
        for _ in range(3):
            assert m.check("acme", "api_call").allowed is True
        assert m.check("acme", "api_call").allowed is False

    def test_reset_clears_counts(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "api_call", max_calls_per_hour=2)
        m.check("acme", "api_call")
        m.check("acme", "api_call")
        assert m.check("acme", "api_call").allowed is False
        m.reset_call_counts()
        assert m.check("acme", "api_call").allowed is True


class TestExpiredRules:
    def test_expired_rule_not_applied(self):
        m = ToolPermissionMatrix()
        past_time = time.time() - 3600  # 1 hour ago
        m.grant("acme", "old_tool", expires_at=past_time)
        result = m.check("acme", "old_tool")
        assert result.allowed is False

    def test_future_expiry_still_valid(self):
        m = ToolPermissionMatrix()
        future_time = time.time() + 3600  # 1 hour from now
        m.grant("acme", "temp_tool", expires_at=future_time)
        assert m.check("acme", "temp_tool").allowed is True


class TestGetTenantTools:
    def test_lists_allowed_tools(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb")
        m.grant("acme", "send_email")
        tools = m.get_tenant_tools("acme")
        assert "search_kb" in tools
        assert "send_email" in tools

    def test_excludes_denied_tools(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb")
        m.deny("acme", "delete_records")
        tools = m.get_tenant_tools("acme")
        assert "delete_records" not in tools

    def test_excludes_wildcard(self):
        m = ToolPermissionMatrix()
        m.grant("admin", "*")
        tools = m.get_tenant_tools("admin")
        assert "*" not in tools


class TestRevoke:
    def test_revoke_removes_rules(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "search_kb")
        assert m.check("acme", "search_kb").allowed is True
        removed = m.revoke("acme", "search_kb")
        assert removed == 1
        assert m.check("acme", "search_kb").allowed is False

    def test_revoke_nonexistent(self):
        m = ToolPermissionMatrix()
        removed = m.revoke("nobody", "nothing")
        assert removed == 0


class TestAuditLog:
    def test_unauthorized_logged(self):
        m = ToolPermissionMatrix(default_deny=True)
        m.check("acme", "forbidden_tool")
        log = m.get_audit_log()
        assert len(log) == 1
        assert log[0].tenant_id == "acme"
        assert log[0].tool_name == "forbidden_tool"

    def test_authorized_not_logged(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "allowed_tool")
        m.check("acme", "allowed_tool")
        log = m.get_audit_log()
        assert len(log) == 0

    def test_audit_log_for_tenant(self):
        m = ToolPermissionMatrix()
        m.check("tenant-a", "tool-x")
        m.check("tenant-b", "tool-y")
        log_a = m.get_audit_log_for_tenant("tenant-a")
        assert len(log_a) == 1
        assert log_a[0].tenant_id == "tenant-a"

    def test_audit_disabled(self):
        m = ToolPermissionMatrix(audit_unauthorized=False)
        m.check("acme", "tool")
        assert len(m.get_audit_log()) == 0

    def test_deny_rule_logged(self):
        m = ToolPermissionMatrix()
        m.grant("acme", "tool")
        m.deny("acme", "tool")
        m.check("acme", "tool")
        log = m.get_audit_log()
        assert len(log) == 1
        assert "DENY" in log[0].reason
