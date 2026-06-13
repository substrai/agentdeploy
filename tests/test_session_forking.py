"""Tests for session forking for parallel conversation branches."""

import pytest

from agentdeploy.sessions.forking import (
    Branch,
    BranchResult,
    BranchStatus,
    Fork,
    ForkConfig,
    MergeResult,
    MergeStrategy,
    SessionForker,
)


@pytest.fixture
def forker():
    return SessionForker(max_concurrent_forks=5, max_branches_per_fork=10)


@pytest.fixture
def basic_fork(forker):
    """Create a basic fork with 3 branches."""
    return forker.create_fork(
        session_id="session-123",
        config=ForkConfig(
            num_branches=3,
            merge_strategy=MergeStrategy.BEST_SCORE,
            branch_names=["branch-a", "branch-b", "branch-c"],
        ),
        base_context={"history": ["msg1", "msg2"]},
    )


class TestForkCreation:
    def test_create_basic_fork(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=2),
        )
        assert fork.fork_id != ""
        assert len(fork.branches) == 2
        assert fork.status == BranchStatus.RUNNING

    def test_branches_have_names(self, basic_fork):
        names = [b.name for b in basic_fork.branches]
        assert "branch-a" in names
        assert "branch-b" in names
        assert "branch-c" in names

    def test_branches_get_base_context(self, basic_fork):
        for branch in basic_fork.branches:
            assert branch.context == {"history": ["msg1", "msg2"]}

    def test_branches_have_isolated_context(self, basic_fork):
        # Modifying one branch's context should not affect others
        basic_fork.branches[0].context["extra"] = "data"
        assert "extra" not in basic_fork.branches[1].context

    def test_max_branches_limit(self, forker):
        with pytest.raises(ValueError, match="Cannot create"):
            forker.create_fork(
                session_id="sess-1",
                config=ForkConfig(num_branches=15),
            )

    def test_max_concurrent_forks_limit(self):
        forker = SessionForker(max_concurrent_forks=2)
        forker.create_fork(session_id="sess-1", config=ForkConfig(num_branches=2))
        forker.create_fork(session_id="sess-1", config=ForkConfig(num_branches=2))
        with pytest.raises(ValueError, match="max concurrent forks"):
            forker.create_fork(session_id="sess-1", config=ForkConfig(num_branches=2))

    def test_default_branch_names(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=3),
        )
        names = [b.name for b in fork.branches]
        assert names == ["branch-0", "branch-1", "branch-2"]


class TestBranchCompletion:
    def test_complete_branch(self, forker, basic_fork):
        branch = basic_fork.branches[0]
        result = forker.complete_branch(
            basic_fork.fork_id,
            branch.branch_id,
            output="Result A",
            score=0.85,
        )
        assert result is True
        assert branch.status == BranchStatus.COMPLETED
        assert branch.result.output == "Result A"
        assert branch.result.score == 0.85

    def test_fail_branch(self, forker, basic_fork):
        branch = basic_fork.branches[0]
        result = forker.fail_branch(
            basic_fork.fork_id,
            branch.branch_id,
            error="Something went wrong",
        )
        assert result is True
        assert branch.status == BranchStatus.FAILED
        assert branch.result.error == "Something went wrong"
        assert branch.result.success is False

    def test_cancel_branch(self, forker, basic_fork):
        branch = basic_fork.branches[0]
        result = forker.cancel_branch(basic_fork.fork_id, branch.branch_id)
        assert result is True
        assert branch.status == BranchStatus.CANCELLED

    def test_cannot_complete_inactive_branch(self, forker, basic_fork):
        branch = basic_fork.branches[0]
        forker.complete_branch(basic_fork.fork_id, branch.branch_id, output="Done", score=0.5)
        # Try to complete again
        result = forker.complete_branch(basic_fork.fork_id, branch.branch_id, output="Again", score=0.9)
        assert result is False

    def test_complete_nonexistent_branch(self, forker, basic_fork):
        result = forker.complete_branch(basic_fork.fork_id, "fake-id", output="x")
        assert result is False


class TestMergeStrategies:
    def test_best_score_merge(self, forker, basic_fork):
        branches = basic_fork.branches
        forker.complete_branch(basic_fork.fork_id, branches[0].branch_id, output="Low", score=0.3)
        forker.complete_branch(basic_fork.fork_id, branches[1].branch_id, output="High", score=0.9)
        forker.complete_branch(basic_fork.fork_id, branches[2].branch_id, output="Mid", score=0.6)

        result = forker.merge(basic_fork.fork_id)
        assert result.merged_output == "High"
        assert result.strategy_used == MergeStrategy.BEST_SCORE
        assert result.winning_branch_id == branches[1].branch_id

    def test_first_wins_merge(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=3, merge_strategy=MergeStrategy.FIRST_WINS),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="First", score=0.5)
        forker.complete_branch(fork.fork_id, branches[1].branch_id, output="Second", score=0.9)

        result = forker.merge(fork.fork_id)
        assert result.merged_output == "First"
        assert result.strategy_used == MergeStrategy.FIRST_WINS

    def test_all_merge(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=3, merge_strategy=MergeStrategy.ALL),
        )
        branches = fork.branches
        for i, branch in enumerate(branches):
            forker.complete_branch(fork.fork_id, branch.branch_id, output=f"Result-{i}", score=0.5)

        result = forker.merge(fork.fork_id)
        assert result.merged_output == ["Result-0", "Result-1", "Result-2"]
        assert result.branches_merged == 3

    def test_majority_merge(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=3, merge_strategy=MergeStrategy.MAJORITY),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="yes", score=0.5)
        forker.complete_branch(fork.fork_id, branches[1].branch_id, output="yes", score=0.5)
        forker.complete_branch(fork.fork_id, branches[2].branch_id, output="no", score=0.5)

        result = forker.merge(fork.fork_id)
        assert result.merged_output == "yes"
        assert result.strategy_used == MergeStrategy.MAJORITY

    def test_custom_merge_strategy(self, forker):
        def custom_merger(results):
            return " + ".join(r.output for r in results)

        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(
                num_branches=2,
                merge_strategy=MergeStrategy.CUSTOM,
                custom_merger=custom_merger,
            ),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="A", score=0.5)
        forker.complete_branch(fork.fork_id, branches[1].branch_id, output="B", score=0.5)

        result = forker.merge(fork.fork_id)
        assert result.merged_output == "A + B"

    def test_merge_not_ready(self, forker, basic_fork):
        # No branches completed yet, min_branches_required defaults to 1
        # Actually need at least 1 completed
        with pytest.raises(ValueError, match="not ready to merge"):
            forker.merge(basic_fork.fork_id)


class TestAutoCancel:
    def test_auto_cancel_on_merge(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(
                num_branches=3,
                merge_strategy=MergeStrategy.FIRST_WINS,
                auto_cancel_on_merge=True,
                min_branches_required=1,
            ),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="Done", score=0.5)
        forker.merge(fork.fork_id)

        # Remaining branches should be cancelled
        assert branches[1].status == BranchStatus.CANCELLED
        assert branches[2].status == BranchStatus.CANCELLED

    def test_no_auto_cancel(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(
                num_branches=3,
                merge_strategy=MergeStrategy.FIRST_WINS,
                auto_cancel_on_merge=False,
                min_branches_required=1,
            ),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="Done", score=0.5)
        forker.merge(fork.fork_id)

        # Remaining branches should still be running
        assert branches[1].status == BranchStatus.RUNNING
        assert branches[2].status == BranchStatus.RUNNING


class TestForkManagement:
    def test_get_fork(self, forker, basic_fork):
        retrieved = forker.get_fork(basic_fork.fork_id)
        assert retrieved is not None
        assert retrieved.fork_id == basic_fork.fork_id

    def test_get_nonexistent_fork(self, forker):
        assert forker.get_fork("nonexistent") is None

    def test_list_forks_by_session(self, forker):
        forker.create_fork(session_id="sess-a", config=ForkConfig(num_branches=2))
        forker.create_fork(session_id="sess-a", config=ForkConfig(num_branches=2))
        forker.create_fork(session_id="sess-b", config=ForkConfig(num_branches=2))

        forks_a = forker.list_forks(session_id="sess-a")
        assert len(forks_a) == 2

        forks_b = forker.list_forks(session_id="sess-b")
        assert len(forks_b) == 1

    def test_cleanup_completed(self, forker):
        fork = forker.create_fork(
            session_id="sess-1",
            config=ForkConfig(num_branches=2, min_branches_required=1),
        )
        branches = fork.branches
        forker.complete_branch(fork.fork_id, branches[0].branch_id, output="Done", score=0.5)
        forker.merge(fork.fork_id)

        cleaned = forker.cleanup_completed()
        assert cleaned == 1
        assert forker.get_fork(fork.fork_id) is None

    def test_active_fork_count(self, forker):
        forker.create_fork(session_id="sess-1", config=ForkConfig(num_branches=2))
        forker.create_fork(session_id="sess-2", config=ForkConfig(num_branches=2))
        assert forker.active_fork_count == 2


class TestBranch:
    def test_branch_is_active(self):
        branch = Branch(status=BranchStatus.RUNNING)
        assert branch.is_active is True

        branch.complete(output="done")
        assert branch.is_active is False

    def test_branch_duration(self):
        branch = Branch()
        # Duration should be positive since creation
        assert branch.duration_ms >= 0

    def test_branch_result_success(self):
        result = BranchResult(branch_id="b1", output="data")
        assert result.success is True

        failed = BranchResult(branch_id="b2", error="oops")
        assert failed.success is False
