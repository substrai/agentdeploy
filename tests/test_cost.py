"""Tests for cost enforcement."""

from agentdeploy.cost.enforcer import CostEnforcer, CostBudget, BudgetAction


class TestCostEnforcer:
    def test_within_budget(self):
        enforcer = CostEnforcer(CostBudget(daily_budget=100.0))
        result = enforcer.check_request("sess-1", estimated_cost=0.01)
        assert result.allowed
        assert result.action == BudgetAction.ALLOW

    def test_exceeds_per_request(self):
        enforcer = CostEnforcer(CostBudget(max_cost_per_request=0.10))
        result = enforcer.check_request("sess-1", estimated_cost=0.50)
        assert not result.allowed
        assert result.action == BudgetAction.BLOCK

    def test_exceeds_session_budget(self):
        enforcer = CostEnforcer(CostBudget(max_cost_per_session=0.05))
        enforcer.record_cost("sess-1", 0.04)
        result = enforcer.check_request("sess-1", estimated_cost=0.02)
        assert not result.allowed

    def test_exceeds_daily_budget(self):
        enforcer = CostEnforcer(CostBudget(daily_budget=0.10))
        enforcer.record_cost("sess-1", 0.09)
        result = enforcer.check_request("sess-2", estimated_cost=0.02)
        assert not result.allowed

    def test_record_cost(self):
        enforcer = CostEnforcer()
        enforcer.record_cost("sess-1", 0.05)
        enforcer.record_cost("sess-1", 0.03)
        assert enforcer.get_session_cost("sess-1") == 0.08
        assert enforcer.get_daily_cost() == 0.08

    def test_daily_remaining(self):
        enforcer = CostEnforcer(CostBudget(daily_budget=10.0))
        enforcer.record_cost("sess-1", 3.0)
        assert enforcer.get_daily_remaining() == 7.0

    def test_reset_daily(self):
        enforcer = CostEnforcer()
        enforcer.record_cost("sess-1", 5.0)
        enforcer.reset_daily()
        assert enforcer.get_daily_cost() == 0.0

    def test_downgrade_action(self):
        budget = CostBudget(daily_budget=0.01, on_exceed="downgrade")
        enforcer = CostEnforcer(budget)
        enforcer.record_cost("sess-1", 0.02)
        result = enforcer.check_request("sess-2", estimated_cost=0.01)
        assert not result.allowed
        assert result.action == BudgetAction.DOWNGRADE
        assert result.suggested_model == "bedrock/claude-3-haiku"
