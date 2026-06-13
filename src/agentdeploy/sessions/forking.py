"""Session forking for parallel conversation branches.

Fork a session into multiple branches, merge results back.
Branch tracking, merge strategies (first-wins, best-score, all).

Features:
- Fork an existing session into N parallel branches
- Each branch executes independently with isolated context
- Configurable merge strategies for combining branch results
- Branch tracking with status, scores, and metadata
- Support for nested forks (fork within a fork)
- Automatic cleanup of completed branches
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BranchStatus(Enum):
    """Status of a forked branch."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MERGED = "merged"


class MergeStrategy(Enum):
    """Strategy for merging branch results back to parent."""

    FIRST_WINS = "first_wins"
    BEST_SCORE = "best_score"
    ALL = "all"
    MAJORITY = "majority"
    CUSTOM = "custom"


@dataclass
class BranchResult:
    """Result from a completed branch.

    Attributes:
        branch_id: Unique branch identifier.
        output: The branch's output value.
        score: Optional quality score (0.0 to 1.0) for ranking.
        duration_ms: How long the branch took to execute.
        metadata: Additional result metadata.
        error: Error message if branch failed.
    """

    branch_id: str
    output: Any = None
    score: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the branch completed successfully."""
        return self.error is None


@dataclass
class Branch:
    """A forked conversation branch.

    Attributes:
        branch_id: Unique branch identifier.
        parent_session_id: ID of the session this was forked from.
        fork_id: ID of the fork operation that created this branch.
        name: Human-readable branch name.
        status: Current branch status.
        context: Branch-specific context (isolated from other branches).
        created_at: When the branch was created.
        completed_at: When the branch completed (if finished).
        result: Branch result (if completed).
        metadata: Additional branch metadata.
    """

    branch_id: str = ""
    parent_session_id: str = ""
    fork_id: str = ""
    name: str = ""
    status: BranchStatus = BranchStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    completed_at: Optional[float] = None
    result: Optional[BranchResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.branch_id:
            self.branch_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    @property
    def duration_ms(self) -> float:
        """Time elapsed since branch creation."""
        end = self.completed_at or time.time()
        return (end - self.created_at) * 1000

    @property
    def is_active(self) -> bool:
        """Whether the branch is still active."""
        return self.status in (BranchStatus.PENDING, BranchStatus.RUNNING)

    def complete(self, output: Any, score: float = 0.0, metadata: Optional[Dict] = None) -> None:
        """Mark the branch as completed with a result."""
        self.completed_at = time.time()
        self.status = BranchStatus.COMPLETED
        self.result = BranchResult(
            branch_id=self.branch_id,
            output=output,
            score=score,
            duration_ms=self.duration_ms,
            metadata=metadata or {},
        )

    def fail(self, error: str) -> None:
        """Mark the branch as failed."""
        self.completed_at = time.time()
        self.status = BranchStatus.FAILED
        self.result = BranchResult(
            branch_id=self.branch_id,
            error=error,
            duration_ms=self.duration_ms,
        )

    def cancel(self) -> None:
        """Cancel this branch."""
        self.completed_at = time.time()
        self.status = BranchStatus.CANCELLED


@dataclass
class ForkConfig:
    """Configuration for a fork operation.

    Attributes:
        num_branches: Number of branches to create.
        merge_strategy: How to merge branch results.
        timeout_ms: Maximum time to wait for branches (0 = no timeout).
        min_branches_required: Minimum branches that must complete before merge.
        auto_cancel_on_merge: Whether to cancel remaining branches after merge.
        custom_merger: Custom merge function (for CUSTOM strategy).
        branch_names: Optional names for each branch.
    """

    num_branches: int = 2
    merge_strategy: MergeStrategy = MergeStrategy.BEST_SCORE
    timeout_ms: float = 0.0
    min_branches_required: int = 1
    auto_cancel_on_merge: bool = True
    custom_merger: Optional[Callable[[List[BranchResult]], Any]] = None
    branch_names: Optional[List[str]] = None


@dataclass
class MergeResult:
    """Result of merging branch results.

    Attributes:
        merged_output: The final merged output value.
        strategy_used: Which merge strategy was applied.
        branches_merged: Number of branches included in merge.
        winning_branch_id: ID of the winning branch (for first-wins/best-score).
        all_results: All branch results that were considered.
        duration_ms: Total fork-to-merge duration.
    """

    merged_output: Any = None
    strategy_used: MergeStrategy = MergeStrategy.BEST_SCORE
    branches_merged: int = 0
    winning_branch_id: Optional[str] = None
    all_results: List[BranchResult] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class Fork:
    """A fork operation tracking multiple branches.

    Attributes:
        fork_id: Unique fork identifier.
        parent_session_id: Session this fork belongs to.
        config: Fork configuration.
        branches: All branches in this fork.
        created_at: When the fork was created.
        merged_at: When the merge completed.
        merge_result: The merge result (if merged).
        status: Overall fork status.
    """

    fork_id: str = ""
    parent_session_id: str = ""
    config: ForkConfig = field(default_factory=ForkConfig)
    branches: List[Branch] = field(default_factory=list)
    created_at: float = 0.0
    merged_at: Optional[float] = None
    merge_result: Optional[MergeResult] = None
    status: BranchStatus = BranchStatus.PENDING

    def __post_init__(self):
        if not self.fork_id:
            self.fork_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    @property
    def active_branches(self) -> List[Branch]:
        """Get currently active branches."""
        return [b for b in self.branches if b.is_active]

    @property
    def completed_branches(self) -> List[Branch]:
        """Get completed branches."""
        return [b for b in self.branches if b.status == BranchStatus.COMPLETED]

    @property
    def is_ready_to_merge(self) -> bool:
        """Whether enough branches completed to perform merge."""
        completed = len(self.completed_branches)
        return completed >= self.config.min_branches_required

    @property
    def all_done(self) -> bool:
        """Whether all branches are finished (completed, failed, or cancelled)."""
        return all(not b.is_active for b in self.branches)


class SessionForker:
    """Manages session forking and merging operations.

    Creates parallel conversation branches from a session, tracks their
    execution, and merges results using configurable strategies.

    Args:
        max_concurrent_forks: Maximum number of active forks per session.
        max_branches_per_fork: Maximum branches allowed in a single fork.

    Example:
        forker = SessionForker()

        # Create a fork with 3 branches
        fork = forker.create_fork(
            session_id="sess-123",
            config=ForkConfig(
                num_branches=3,
                merge_strategy=MergeStrategy.BEST_SCORE,
            ),
            base_context={"conversation": [...]}
        )

        # Complete branches with results
        forker.complete_branch(fork.fork_id, fork.branches[0].branch_id, output="Result A", score=0.8)
        forker.complete_branch(fork.fork_id, fork.branches[1].branch_id, output="Result B", score=0.9)

        # Merge results
        merge_result = forker.merge(fork.fork_id)
        print(merge_result.merged_output)  # "Result B" (highest score)
    """

    def __init__(
        self,
        max_concurrent_forks: int = 5,
        max_branches_per_fork: int = 10,
    ):
        self.max_concurrent_forks = max_concurrent_forks
        self.max_branches_per_fork = max_branches_per_fork
        self._forks: Dict[str, Fork] = {}

    @property
    def active_fork_count(self) -> int:
        """Number of currently active forks."""
        return sum(1 for f in self._forks.values() if f.status in (BranchStatus.PENDING, BranchStatus.RUNNING))

    def create_fork(
        self,
        session_id: str,
        config: Optional[ForkConfig] = None,
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Fork:
        """Create a new fork from a session.

        Args:
            session_id: The parent session to fork from.
            config: Fork configuration.
            base_context: Context to copy into each branch.

        Returns:
            The created Fork with initialized branches.

        Raises:
            ValueError: If fork limits are exceeded.
        """
        config = config or ForkConfig()

        if config.num_branches > self.max_branches_per_fork:
            raise ValueError(
                f"Cannot create {config.num_branches} branches "
                f"(max: {self.max_branches_per_fork})"
            )

        # Check concurrent fork limit for this session
        session_forks = [
            f for f in self._forks.values()
            if f.parent_session_id == session_id and f.status in (BranchStatus.PENDING, BranchStatus.RUNNING)
        ]
        if len(session_forks) >= self.max_concurrent_forks:
            raise ValueError(
                f"Session '{session_id}' has reached max concurrent forks "
                f"({self.max_concurrent_forks})"
            )

        fork = Fork(
            parent_session_id=session_id,
            config=config,
            status=BranchStatus.RUNNING,
        )

        # Create branches
        for i in range(config.num_branches):
            branch_name = ""
            if config.branch_names and i < len(config.branch_names):
                branch_name = config.branch_names[i]
            else:
                branch_name = f"branch-{i}"

            branch = Branch(
                parent_session_id=session_id,
                fork_id=fork.fork_id,
                name=branch_name,
                status=BranchStatus.RUNNING,
                context=dict(base_context) if base_context else {},
            )
            fork.branches.append(branch)

        self._forks[fork.fork_id] = fork
        return fork

    def get_fork(self, fork_id: str) -> Optional[Fork]:
        """Get a fork by ID."""
        return self._forks.get(fork_id)

    def get_branch(self, fork_id: str, branch_id: str) -> Optional[Branch]:
        """Get a specific branch within a fork."""
        fork = self._forks.get(fork_id)
        if fork is None:
            return None
        for branch in fork.branches:
            if branch.branch_id == branch_id:
                return branch
        return None

    def complete_branch(
        self,
        fork_id: str,
        branch_id: str,
        output: Any,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a branch as completed with its result.

        Args:
            fork_id: The fork containing the branch.
            branch_id: The branch to complete.
            output: The branch's output value.
            score: Quality score for ranking (0.0 to 1.0).
            metadata: Additional result metadata.

        Returns:
            True if branch was found and completed, False otherwise.
        """
        branch = self.get_branch(fork_id, branch_id)
        if branch is None or not branch.is_active:
            return False

        branch.complete(output, score, metadata)
        return True

    def fail_branch(self, fork_id: str, branch_id: str, error: str) -> bool:
        """Mark a branch as failed.

        Args:
            fork_id: The fork containing the branch.
            branch_id: The branch that failed.
            error: Error description.

        Returns:
            True if branch was found and marked failed, False otherwise.
        """
        branch = self.get_branch(fork_id, branch_id)
        if branch is None or not branch.is_active:
            return False

        branch.fail(error)
        return True

    def cancel_branch(self, fork_id: str, branch_id: str) -> bool:
        """Cancel a specific branch.

        Args:
            fork_id: The fork containing the branch.
            branch_id: The branch to cancel.

        Returns:
            True if branch was found and cancelled, False otherwise.
        """
        branch = self.get_branch(fork_id, branch_id)
        if branch is None or not branch.is_active:
            return False

        branch.cancel()
        return True

    def merge(self, fork_id: str) -> Optional[MergeResult]:
        """Merge completed branch results using the configured strategy.

        Args:
            fork_id: The fork to merge.

        Returns:
            MergeResult with the merged output, or None if fork not found.

        Raises:
            ValueError: If not enough branches have completed.
        """
        fork = self._forks.get(fork_id)
        if fork is None:
            return None

        if not fork.is_ready_to_merge:
            raise ValueError(
                f"Fork '{fork_id}' not ready to merge: "
                f"{len(fork.completed_branches)}/{fork.config.min_branches_required} "
                f"branches completed"
            )

        completed_results = [
            b.result for b in fork.completed_branches
            if b.result is not None
        ]

        merge_result = self._apply_merge_strategy(fork.config, completed_results)
        merge_result.duration_ms = (time.time() - fork.created_at) * 1000

        # Auto-cancel remaining branches
        if fork.config.auto_cancel_on_merge:
            for branch in fork.active_branches:
                branch.cancel()

        fork.merge_result = merge_result
        fork.merged_at = time.time()
        fork.status = BranchStatus.MERGED

        return merge_result

    def list_forks(self, session_id: Optional[str] = None) -> List[Fork]:
        """List all forks, optionally filtered by session.

        Args:
            session_id: Optional session ID to filter by.

        Returns:
            List of matching forks.
        """
        if session_id:
            return [f for f in self._forks.values() if f.parent_session_id == session_id]
        return list(self._forks.values())

    def cleanup_completed(self, session_id: Optional[str] = None) -> int:
        """Remove completed/merged forks from tracking.

        Args:
            session_id: Optional session ID to filter cleanup.

        Returns:
            Number of forks cleaned up.
        """
        to_remove = []
        for fork_id, fork in self._forks.items():
            if session_id and fork.parent_session_id != session_id:
                continue
            if fork.status == BranchStatus.MERGED or fork.all_done:
                to_remove.append(fork_id)

        for fork_id in to_remove:
            del self._forks[fork_id]

        return len(to_remove)

    def _apply_merge_strategy(
        self,
        config: ForkConfig,
        results: List[BranchResult],
    ) -> MergeResult:
        """Apply the configured merge strategy to branch results."""
        if not results:
            return MergeResult(strategy_used=config.merge_strategy)

        if config.merge_strategy == MergeStrategy.FIRST_WINS:
            winner = results[0]
            return MergeResult(
                merged_output=winner.output,
                strategy_used=MergeStrategy.FIRST_WINS,
                branches_merged=1,
                winning_branch_id=winner.branch_id,
                all_results=results,
            )

        elif config.merge_strategy == MergeStrategy.BEST_SCORE:
            winner = max(results, key=lambda r: r.score)
            return MergeResult(
                merged_output=winner.output,
                strategy_used=MergeStrategy.BEST_SCORE,
                branches_merged=1,
                winning_branch_id=winner.branch_id,
                all_results=results,
            )

        elif config.merge_strategy == MergeStrategy.ALL:
            all_outputs = [r.output for r in results]
            return MergeResult(
                merged_output=all_outputs,
                strategy_used=MergeStrategy.ALL,
                branches_merged=len(results),
                all_results=results,
            )

        elif config.merge_strategy == MergeStrategy.MAJORITY:
            # Count occurrences of each output
            output_counts: Dict[str, int] = {}
            for r in results:
                key = str(r.output)
                output_counts[key] = output_counts.get(key, 0) + 1
            majority_key = max(output_counts, key=output_counts.get)
            majority_result = next(r for r in results if str(r.output) == majority_key)
            return MergeResult(
                merged_output=majority_result.output,
                strategy_used=MergeStrategy.MAJORITY,
                branches_merged=output_counts[majority_key],
                winning_branch_id=majority_result.branch_id,
                all_results=results,
            )

        elif config.merge_strategy == MergeStrategy.CUSTOM and config.custom_merger:
            merged = config.custom_merger(results)
            return MergeResult(
                merged_output=merged,
                strategy_used=MergeStrategy.CUSTOM,
                branches_merged=len(results),
                all_results=results,
            )

        # Default fallback to best_score
        winner = max(results, key=lambda r: r.score)
        return MergeResult(
            merged_output=winner.output,
            strategy_used=MergeStrategy.BEST_SCORE,
            branches_merged=1,
            winning_branch_id=winner.branch_id,
            all_results=results,
        )
