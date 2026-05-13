"""Tenant management - isolation, configuration, and provisioning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TenantConfig:
    """Configuration for a single tenant."""

    tenant_id: str
    name: str = ""
    rate_limit: str = "100/min"  # requests per minute
    budget_daily: float = 50.0
    budget_monthly: float = 1000.0
    on_budget_exceeded: str = "block"  # block | downgrade | alert
    model_override: Optional[str] = None
    tools_allowed: List[str] = field(default_factory=lambda: ["all"])
    tools_denied: List[str] = field(default_factory=list)
    max_turns_per_session: int = 100
    max_sessions: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rate_limit_per_minute(self) -> int:
        """Parse rate limit string to requests per minute."""
        parts = self.rate_limit.split("/")
        return int(parts[0])

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed for this tenant."""
        if tool_name in self.tools_denied:
            return False
        if self.tools_allowed == ["all"]:
            return True
        return tool_name in self.tools_allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "rate_limit": self.rate_limit,
            "budget_daily": self.budget_daily,
            "budget_monthly": self.budget_monthly,
            "on_budget_exceeded": self.on_budget_exceeded,
            "model_override": self.model_override,
            "tools_allowed": self.tools_allowed,
            "tools_denied": self.tools_denied,
            "max_turns_per_session": self.max_turns_per_session,
        }


@dataclass
class Tenant:
    """A tenant with usage tracking."""

    config: TenantConfig
    created_at: float = field(default_factory=time.time)
    total_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    active_sessions: int = 0
    is_active: bool = True

    @property
    def tenant_id(self) -> str:
        return self.config.tenant_id

    def record_usage(self, cost: float = 0.0, tokens: int = 0) -> None:
        """Record usage for this tenant."""
        self.total_requests += 1
        self.total_cost += cost
        self.total_tokens += tokens

    def get_daily_cost(self) -> float:
        """Get cost for current day (simplified - returns total for now)."""
        return self.total_cost

    def is_over_budget(self) -> bool:
        """Check if tenant has exceeded daily budget."""
        return self.total_cost >= self.config.budget_daily

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.config.to_dict(),
            "total_requests": self.total_requests,
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "active_sessions": self.active_sessions,
            "is_active": self.is_active,
        }


class TenantManager:
    """Manages tenants with isolation and configuration.

    Usage:
        manager = TenantManager()
        manager.create("enterprise-client", TenantConfig(...))
        tenant = manager.get("enterprise-client")
        if tenant.is_over_budget():
            handle_budget_exceeded(tenant)
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        # Create default tenant
        self._tenants["default"] = Tenant(
            config=TenantConfig(tenant_id="default", name="Default Tenant")
        )

    def create(self, tenant_id: str, config: Optional[TenantConfig] = None) -> Tenant:
        """Create a new tenant."""
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        cfg = config or TenantConfig(tenant_id=tenant_id, name=tenant_id)
        tenant = Tenant(config=cfg)
        self._tenants[tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_or_default(self, tenant_id: Optional[str]) -> Tenant:
        """Get tenant or return default."""
        if tenant_id and tenant_id in self._tenants:
            return self._tenants[tenant_id]
        return self._tenants["default"]

    def update(self, tenant_id: str, config: TenantConfig) -> bool:
        """Update tenant configuration."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        tenant.config = config
        return True

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant (cannot delete default)."""
        if tenant_id == "default":
            return False
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            return True
        return False

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivate a tenant without deleting."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.is_active = False
            return True
        return False

    def list_tenants(self, active_only: bool = False) -> List[Tenant]:
        """List all tenants."""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return tenants

    def get_usage_report(self) -> Dict[str, Dict[str, Any]]:
        """Get usage report for all tenants."""
        return {
            tid: {
                "requests": t.total_requests,
                "cost": round(t.total_cost, 6),
                "tokens": t.total_tokens,
                "budget_remaining": round(t.config.budget_daily - t.total_cost, 4),
                "over_budget": t.is_over_budget(),
            }
            for tid, t in self._tenants.items()
        }

    def reset_daily_usage(self) -> None:
        """Reset daily usage counters for all tenants."""
        for tenant in self._tenants.values():
            tenant.total_cost = 0.0
            tenant.total_requests = 0
            tenant.total_tokens = 0

    @property
    def tenant_count(self) -> int:
        return len(self._tenants)
