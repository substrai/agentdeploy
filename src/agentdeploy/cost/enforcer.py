"""Cost enforcement - circuit breakers and budget tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BudgetAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    DOWNGRADE = "downgrade"
    ALERT = "alert"
    KILL = "kill"


@dataclass
class CostBudget:
    """Budget configuration for cost enforcement."""

    max_cost_per_request: float = 0.50
    max_cost_per_session: float = 5.00
    daily_budget: float = 50.00
    monthly_budget: float = 1000.00
    on_exceed: str = "block"  # block | downgrade | alert | kill

    def get_action(self) -> BudgetAction:
        return BudgetAction(self.on_exceed)


@dataclass
class CostRecord:
    """Tracks cost for a specific scope."""

    scope: str  # "request", "session", "tenant", "global"
    scope_id: str
    total_cost: float = 0.0
    request_count: int = 0
    period_start: float = field(default_factory=time.time)


@dataclass
class CostCheckResult:
    """Result of a cost enforcement check."""

    allowed: bool
    action: BudgetAction
    current_cost: float
    budget_limit: float
    usage_percent: float
    message: str
    suggested_model: Optional[str] = None  # for downgrade action


class CostEnforcer:
    """Enforces cost budgets and circuit breakers.

    Usage:
        enforcer = CostEnforcer(budget)
        check = enforcer.check_request(session_id="sess-1", estimated_cost=0.05)
        if not check.allowed:
            return error_response(check.message)
        # ... execute agent ...
        enforcer.record_cost(session_id="sess-1", cost=0.03)
    """

    def __init__(self, budget: Optional[CostBudget] = None):
        self.budget = budget or CostBudget()
        self._session_costs: Dict[str, CostRecord] = {}
        self._daily_cost: CostRecord = CostRecord(scope="daily", scope_id="global")
        self._request_log: List[Dict[str, Any]] = []

    def check_request(
        self, session_id: str, estimated_cost: float = 0.0, tenant_id: str = "default"
    ) -> CostCheckResult:
        """Check if a request is within budget.

        Args:
            session_id: Current session ID
            estimated_cost: Estimated cost of this request
            tenant_id: Tenant making the request

        Returns:
            CostCheckResult with allow/deny decision
        """
        # Check session budget
        session_record = self._session_costs.get(session_id)
        if session_record:
            projected = session_record.total_cost + estimated_cost
            if projected > self.budget.max_cost_per_session:
                return CostCheckResult(
                    allowed=False,
                    action=self.budget.get_action(),
                    current_cost=session_record.total_cost,
                    budget_limit=self.budget.max_cost_per_session,
                    usage_percent=session_record.total_cost / self.budget.max_cost_per_session,
                    message=f"Session budget exceeded: ${session_record.total_cost:.4f} >= ${self.budget.max_cost_per_session}",
                    suggested_model="bedrock/claude-3-haiku" if self.budget.on_exceed == "downgrade" else None,
                )

        # Check per-request budget
        if estimated_cost > self.budget.max_cost_per_request:
            return CostCheckResult(
                allowed=False,
                action=BudgetAction.BLOCK,
                current_cost=estimated_cost,
                budget_limit=self.budget.max_cost_per_request,
                usage_percent=estimated_cost / self.budget.max_cost_per_request,
                message=f"Request cost ${estimated_cost:.4f} exceeds per-request limit ${self.budget.max_cost_per_request}",
            )

        # Check daily budget
        daily_projected = self._daily_cost.total_cost + estimated_cost
        if daily_projected > self.budget.daily_budget:
            return CostCheckResult(
                allowed=False,
                action=self.budget.get_action(),
                current_cost=self._daily_cost.total_cost,
                budget_limit=self.budget.daily_budget,
                usage_percent=self._daily_cost.total_cost / self.budget.daily_budget,
                message=f"Daily budget exceeded: ${self._daily_cost.total_cost:.2f} >= ${self.budget.daily_budget}",
                suggested_model="bedrock/claude-3-haiku" if self.budget.on_exceed == "downgrade" else None,
            )

        # Allowed
        usage_pct = self._daily_cost.total_cost / self.budget.daily_budget if self.budget.daily_budget > 0 else 0
        return CostCheckResult(
            allowed=True,
            action=BudgetAction.ALLOW,
            current_cost=self._daily_cost.total_cost,
            budget_limit=self.budget.daily_budget,
            usage_percent=usage_pct,
            message="Within budget",
        )

    def record_cost(self, session_id: str, cost: float, tokens: int = 0) -> None:
        """Record actual cost after execution."""
        # Session cost
        if session_id not in self._session_costs:
            self._session_costs[session_id] = CostRecord(scope="session", scope_id=session_id)
        self._session_costs[session_id].total_cost += cost
        self._session_costs[session_id].request_count += 1

        # Daily cost
        self._daily_cost.total_cost += cost
        self._daily_cost.request_count += 1

        # Log
        self._request_log.append({
            "session_id": session_id,
            "cost": cost,
            "tokens": tokens,
            "timestamp": time.time(),
        })

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        record = self._session_costs.get(session_id)
        return record.total_cost if record else 0.0

    def get_daily_cost(self) -> float:
        """Get total daily cost."""
        return self._daily_cost.total_cost

    def get_daily_remaining(self) -> float:
        """Get remaining daily budget."""
        return max(self.budget.daily_budget - self._daily_cost.total_cost, 0)

    def reset_daily(self) -> None:
        """Reset daily cost counter."""
        self._daily_cost = CostRecord(scope="daily", scope_id="global")
