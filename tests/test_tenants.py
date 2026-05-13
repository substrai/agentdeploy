"""Tests for multi-tenancy and rate limiting."""

import time
import pytest
from agentdeploy.tenants.manager import TenantManager, Tenant, TenantConfig
from agentdeploy.tenants.rate_limiter import RateLimiter


class TestTenantManager:
    def setup_method(self):
        self.mgr = TenantManager()

    def test_default_tenant_exists(self):
        assert self.mgr.get("default") is not None

    def test_create_tenant(self):
        config = TenantConfig(tenant_id="enterprise", name="Enterprise Client", budget_daily=200.0)
        tenant = self.mgr.create("enterprise", config)
        assert tenant.tenant_id == "enterprise"
        assert tenant.config.budget_daily == 200.0

    def test_create_duplicate_raises(self):
        self.mgr.create("test-1")
        with pytest.raises(ValueError):
            self.mgr.create("test-1")

    def test_get_or_default(self):
        tenant = self.mgr.get_or_default("nonexistent")
        assert tenant.tenant_id == "default"

    def test_delete_tenant(self):
        self.mgr.create("to-delete")
        assert self.mgr.delete("to-delete")
        assert self.mgr.get("to-delete") is None

    def test_cannot_delete_default(self):
        assert not self.mgr.delete("default")

    def test_deactivate_tenant(self):
        self.mgr.create("to-deactivate")
        self.mgr.deactivate("to-deactivate")
        tenant = self.mgr.get("to-deactivate")
        assert not tenant.is_active

    def test_list_tenants(self):
        self.mgr.create("a")
        self.mgr.create("b")
        assert self.mgr.tenant_count == 3  # default + a + b

    def test_list_active_only(self):
        self.mgr.create("active-1")
        self.mgr.create("inactive-1")
        self.mgr.deactivate("inactive-1")
        active = self.mgr.list_tenants(active_only=True)
        assert all(t.is_active for t in active)

    def test_record_usage(self):
        self.mgr.create("usage-test")
        tenant = self.mgr.get("usage-test")
        tenant.record_usage(cost=0.05, tokens=500)
        tenant.record_usage(cost=0.03, tokens=300)
        assert tenant.total_requests == 2
        assert tenant.total_cost == 0.08
        assert tenant.total_tokens == 800

    def test_over_budget(self):
        config = TenantConfig(tenant_id="budget-test", budget_daily=0.10)
        self.mgr.create("budget-test", config)
        tenant = self.mgr.get("budget-test")
        tenant.record_usage(cost=0.11)
        assert tenant.is_over_budget()

    def test_tool_permissions(self):
        config = TenantConfig(
            tenant_id="restricted",
            tools_allowed=["search_kb", "check_status"],
            tools_denied=["delete_data"],
        )
        assert config.is_tool_allowed("search_kb")
        assert not config.is_tool_allowed("admin_panel")
        assert not config.is_tool_allowed("delete_data")

    def test_tools_all_allowed(self):
        config = TenantConfig(tenant_id="open", tools_allowed=["all"])
        assert config.is_tool_allowed("anything")
        assert config.is_tool_allowed("admin_panel")

    def test_usage_report(self):
        self.mgr.create("report-test")
        tenant = self.mgr.get("report-test")
        tenant.record_usage(cost=0.05)
        report = self.mgr.get_usage_report()
        assert "report-test" in report
        assert report["report-test"]["cost"] == 0.05

    def test_reset_daily(self):
        self.mgr.create("reset-test")
        tenant = self.mgr.get("reset-test")
        tenant.record_usage(cost=5.0, tokens=1000)
        self.mgr.reset_daily_usage()
        assert tenant.total_cost == 0.0


class TestRateLimiter:
    def test_within_limit(self):
        limiter = RateLimiter(default_limit=10)
        result = limiter.check("tenant-1")
        assert result.allowed
        assert result.remaining == 9

    def test_exceeds_limit(self):
        limiter = RateLimiter(default_limit=3, window_seconds=60)
        limiter.check("tenant-1")
        limiter.check("tenant-1")
        limiter.check("tenant-1")
        result = limiter.check("tenant-1")
        assert not result.allowed
        assert result.remaining == 0

    def test_custom_limit_per_tenant(self):
        limiter = RateLimiter(default_limit=100)
        limiter.set_limit("free-tier", 5)
        for _ in range(5):
            limiter.check("free-tier")
        result = limiter.check("free-tier")
        assert not result.allowed

    def test_different_tenants_independent(self):
        limiter = RateLimiter(default_limit=2)
        limiter.check("tenant-a")
        limiter.check("tenant-a")
        # tenant-a is at limit
        result_a = limiter.check("tenant-a")
        assert not result_a.allowed
        # tenant-b should still be fine
        result_b = limiter.check("tenant-b")
        assert result_b.allowed

    def test_check_only_no_record(self):
        limiter = RateLimiter(default_limit=10)
        result = limiter.check_only("tenant-1")
        assert result.allowed
        assert result.current_rate == 0  # didn't record

    def test_get_usage(self):
        limiter = RateLimiter(default_limit=100)
        limiter.check("tenant-1")
        limiter.check("tenant-1")
        usage = limiter.get_usage("tenant-1")
        assert usage["current"] == 2
        assert usage["remaining"] == 98

    def test_reset(self):
        limiter = RateLimiter(default_limit=5)
        for _ in range(5):
            limiter.check("tenant-1")
        limiter.reset("tenant-1")
        result = limiter.check("tenant-1")
        assert result.allowed

    def test_retry_after(self):
        limiter = RateLimiter(default_limit=1, window_seconds=60)
        limiter.check("tenant-1")
        result = limiter.check("tenant-1")
        assert not result.allowed
        assert result.retry_after_seconds > 0
