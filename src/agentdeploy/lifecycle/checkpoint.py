"""Agent execution timeout with checkpoint/resume.

Saves agent state on timeout and resumes from the last checkpoint
on the next invocation. Enables long-running agent workflows to
survive Lambda's 15-minute timeout limit.

Usage:
    from agentdeploy.lifecycle.checkpoint import CheckpointManager

    manager = CheckpointManager(timeout_buffer_ms=30000)

    # Save checkpoint before timeout
    if manager.should_checkpoint(context):
        manager.save(execution_id="exec-123", state=agent_state)

    # Resume on next invocation
    state = manager.load(execution_id="exec-123")
    if state:
        agent.resume(state)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    """Status of an agent execution."""

    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class CheckpointData:
    """Serializable checkpoint state."""

    execution_id: str
    step_index: int
    total_steps: int
    status: ExecutionStatus
    timestamp: float
    state: Dict[str, Any]
    completed_steps: List[str]
    pending_steps: List[str]
    accumulated_cost: float = 0.0
    elapsed_ms: float = 0.0
    resume_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "execution_id": self.execution_id,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "state": self.state,
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "accumulated_cost": self.accumulated_cost,
            "elapsed_ms": self.elapsed_ms,
            "resume_count": self.resume_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Deserialize from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            step_index=data["step_index"],
            total_steps=data["total_steps"],
            status=ExecutionStatus(data["status"]),
            timestamp=data["timestamp"],
            state=data["state"],
            completed_steps=data["completed_steps"],
            pending_steps=data["pending_steps"],
            accumulated_cost=data.get("accumulated_cost", 0.0),
            elapsed_ms=data.get("elapsed_ms", 0.0),
            resume_count=data.get("resume_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointResult:
    """Result of a checkpoint save operation."""

    success: bool
    execution_id: str
    step_index: int
    storage_path: str
    size_bytes: int
    duration_ms: float
    error: Optional[str] = None


class CheckpointManager:
    """Manages agent execution checkpoints for timeout resilience.

    Monitors remaining execution time and saves agent state before
    Lambda timeout. On the next invocation, loads the checkpoint
    and resumes from where execution left off.

    Args:
        timeout_buffer_ms: Milliseconds to reserve for checkpoint save (default 30s).
        storage_dir: Directory for checkpoint files (local dev).
        max_checkpoints: Maximum stored checkpoints per execution.
        auto_cleanup: Whether to delete checkpoints after completion.
    """

    def __init__(
        self,
        timeout_buffer_ms: int = 30000,
        storage_dir: str = ".agentdeploy/checkpoints",
        max_checkpoints: int = 10,
        auto_cleanup: bool = True,
    ):
        self._timeout_buffer_ms = timeout_buffer_ms
        self._storage_dir = Path(storage_dir)
        self._max_checkpoints = max_checkpoints
        self._auto_cleanup = auto_cleanup
        self._start_time: Optional[float] = None
        self._execution_id: Optional[str] = None

    @property
    def timeout_buffer_ms(self) -> int:
        """Milliseconds reserved for checkpoint operations."""
        return self._timeout_buffer_ms

    @property
    def storage_dir(self) -> Path:
        """Directory where checkpoints are stored."""
        return self._storage_dir

    def start_execution(self, execution_id: str) -> None:
        """Mark the start of an agent execution.

        Args:
            execution_id: Unique identifier for this execution.
        """
        self._start_time = time.time()
        self._execution_id = execution_id

    def should_checkpoint(
        self,
        remaining_time_ms: Optional[int] = None,
        max_execution_ms: int = 900000,  # 15 minutes default
    ) -> bool:
        """Determine if a checkpoint should be saved now.

        Args:
            remaining_time_ms: Remaining time from Lambda context.
                If None, calculates from start_time and max_execution_ms.
            max_execution_ms: Maximum execution time in ms (if no remaining_time provided).

        Returns:
            True if checkpoint should be saved immediately.
        """
        if remaining_time_ms is not None:
            return remaining_time_ms <= self._timeout_buffer_ms

        # Calculate from elapsed time
        if self._start_time is None:
            return False

        elapsed_ms = (time.time() - self._start_time) * 1000
        remaining = max_execution_ms - elapsed_ms
        return remaining <= self._timeout_buffer_ms

    def save(
        self,
        execution_id: str,
        step_index: int,
        total_steps: int,
        state: Dict[str, Any],
        completed_steps: Optional[List[str]] = None,
        pending_steps: Optional[List[str]] = None,
        cost: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointResult:
        """Save a checkpoint to storage.

        Args:
            execution_id: Unique execution identifier.
            step_index: Current step index (0-based).
            total_steps: Total number of steps in the workflow.
            state: Serializable agent state dictionary.
            completed_steps: Names of completed steps.
            pending_steps: Names of remaining steps.
            cost: Accumulated cost so far.
            metadata: Additional metadata.

        Returns:
            CheckpointResult with save status.
        """
        start = time.time()

        checkpoint = CheckpointData(
            execution_id=execution_id,
            step_index=step_index,
            total_steps=total_steps,
            status=ExecutionStatus.CHECKPOINTED,
            timestamp=time.time(),
            state=state,
            completed_steps=completed_steps or [],
            pending_steps=pending_steps or [],
            accumulated_cost=cost,
            elapsed_ms=(time.time() - self._start_time) * 1000 if self._start_time else 0.0,
            metadata=metadata or {},
        )

        try:
            # Ensure storage directory exists
            self._storage_dir.mkdir(parents=True, exist_ok=True)

            # Write checkpoint file
            file_path = self._storage_dir / f"{execution_id}.json"
            content = json.dumps(checkpoint.to_dict(), default=str)
            file_path.write_text(content)

            duration_ms = (time.time() - start) * 1000

            return CheckpointResult(
                success=True,
                execution_id=execution_id,
                step_index=step_index,
                storage_path=str(file_path),
                size_bytes=len(content.encode()),
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return CheckpointResult(
                success=False,
                execution_id=execution_id,
                step_index=step_index,
                storage_path="",
                size_bytes=0,
                duration_ms=duration_ms,
                error=str(e),
            )

    def load(self, execution_id: str) -> Optional[CheckpointData]:
        """Load a checkpoint from storage.

        Args:
            execution_id: The execution to resume.

        Returns:
            CheckpointData if found, None otherwise.
        """
        file_path = self._storage_dir / f"{execution_id}.json"

        if not file_path.exists():
            return None

        try:
            content = file_path.read_text()
            data = json.loads(content)
            checkpoint = CheckpointData.from_dict(data)
            checkpoint.resume_count += 1
            return checkpoint
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def complete(self, execution_id: str) -> None:
        """Mark an execution as completed and optionally clean up.

        Args:
            execution_id: The completed execution.
        """
        if self._auto_cleanup:
            self.delete(execution_id)

    def delete(self, execution_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            execution_id: The execution checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self._storage_dir / f"{execution_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_checkpoints(self) -> List[str]:
        """List all stored checkpoint execution IDs.

        Returns:
            List of execution IDs with stored checkpoints.
        """
        if not self._storage_dir.exists():
            return []

        return [
            f.stem
            for f in self._storage_dir.glob("*.json")
        ]

    def get_elapsed_ms(self) -> float:
        """Get elapsed time since execution start.

        Returns:
            Elapsed milliseconds, or 0 if not started.
        """
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000
