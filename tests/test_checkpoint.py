"""Tests for agent execution timeout with checkpoint/resume."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from agentdeploy.lifecycle.checkpoint import (
    CheckpointData,
    CheckpointManager,
    CheckpointResult,
    ExecutionStatus,
)


class TestCheckpointManagerInit:
    """Test manager initialization."""

    def test_default_config(self):
        manager = CheckpointManager()
        assert manager.timeout_buffer_ms == 30000

    def test_custom_buffer(self):
        manager = CheckpointManager(timeout_buffer_ms=60000)
        assert manager.timeout_buffer_ms == 60000


class TestShouldCheckpoint:
    """Test timeout detection logic."""

    def test_should_checkpoint_when_low_time(self):
        manager = CheckpointManager(timeout_buffer_ms=30000)
        assert manager.should_checkpoint(remaining_time_ms=20000) is True

    def test_should_not_checkpoint_when_plenty_of_time(self):
        manager = CheckpointManager(timeout_buffer_ms=30000)
        assert manager.should_checkpoint(remaining_time_ms=600000) is False

    def test_should_checkpoint_at_exact_boundary(self):
        manager = CheckpointManager(timeout_buffer_ms=30000)
        assert manager.should_checkpoint(remaining_time_ms=30000) is True

    def test_should_checkpoint_from_elapsed_time(self):
        manager = CheckpointManager(timeout_buffer_ms=30000)
        manager.start_execution("exec-1")
        # With 900s max and buffer of 30s, should checkpoint after 870s
        # Since we just started, should NOT checkpoint
        assert manager.should_checkpoint(max_execution_ms=900000) is False

    def test_no_start_returns_false(self):
        manager = CheckpointManager(timeout_buffer_ms=30000)
        assert manager.should_checkpoint() is False


class TestCheckpointSaveLoad:
    """Test checkpoint persistence."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            manager.start_execution("exec-1")

            result = manager.save(
                execution_id="exec-1",
                step_index=2,
                total_steps=5,
                state={"messages": ["hello", "world"], "context": {"key": "val"}},
                completed_steps=["step_1", "step_2"],
                pending_steps=["step_3", "step_4", "step_5"],
                cost=0.003,
            )

            assert result.success is True
            assert result.size_bytes > 0

            # Load it back
            loaded = manager.load("exec-1")
            assert loaded is not None
            assert loaded.execution_id == "exec-1"
            assert loaded.step_index == 2
            assert loaded.total_steps == 5
            assert loaded.state["messages"] == ["hello", "world"]
            assert loaded.completed_steps == ["step_1", "step_2"]
            assert loaded.pending_steps == ["step_3", "step_4", "step_5"]
            assert loaded.accumulated_cost == 0.003

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            assert manager.load("does-not-exist") is None

    def test_load_increments_resume_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            manager.start_execution("exec-2")
            manager.save(
                execution_id="exec-2", step_index=1,
                total_steps=3, state={},
            )

            loaded = manager.load("exec-2")
            assert loaded is not None
            assert loaded.resume_count == 1

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested_dir = str(Path(tmp) / "deep" / "nested")
            manager = CheckpointManager(storage_dir=nested_dir)
            manager.start_execution("exec-3")

            result = manager.save(
                execution_id="exec-3", step_index=0,
                total_steps=2, state={"x": 1},
            )
            assert result.success is True
            assert Path(nested_dir).exists()

    def test_save_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            manager.start_execution("exec-4")
            manager.save(
                execution_id="exec-4", step_index=0,
                total_steps=3, state={},
                metadata={"agent": "support-bot", "tenant": "acme"},
            )

            loaded = manager.load("exec-4")
            assert loaded is not None
            assert loaded.metadata["agent"] == "support-bot"
            assert loaded.metadata["tenant"] == "acme"


class TestCheckpointLifecycle:
    """Test complete lifecycle operations."""

    def test_complete_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp, auto_cleanup=True)
            manager.start_execution("exec-cleanup")
            manager.save(
                execution_id="exec-cleanup", step_index=2,
                total_steps=3, state={},
            )

            manager.complete("exec-cleanup")
            assert manager.load("exec-cleanup") is None

    def test_complete_without_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp, auto_cleanup=False)
            manager.start_execution("exec-keep")
            manager.save(
                execution_id="exec-keep", step_index=2,
                total_steps=3, state={},
            )

            manager.complete("exec-keep")
            assert manager.load("exec-keep") is not None

    def test_delete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            manager.start_execution("exec-del")
            manager.save(execution_id="exec-del", step_index=0, total_steps=1, state={})

            assert manager.delete("exec-del") is True
            assert manager.delete("exec-del") is False  # Already deleted

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            manager.start_execution("a")
            manager.save(execution_id="exec-a", step_index=0, total_steps=1, state={})
            manager.save(execution_id="exec-b", step_index=0, total_steps=1, state={})
            manager.save(execution_id="exec-c", step_index=0, total_steps=1, state={})

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3
            assert "exec-a" in checkpoints
            assert "exec-b" in checkpoints

    def test_list_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(storage_dir=tmp)
            assert manager.list_checkpoints() == []


class TestCheckpointData:
    """Test CheckpointData serialization."""

    def test_to_dict_and_back(self):
        original = CheckpointData(
            execution_id="e1",
            step_index=3,
            total_steps=7,
            status=ExecutionStatus.CHECKPOINTED,
            timestamp=time.time(),
            state={"key": "value", "nested": {"a": 1}},
            completed_steps=["s1", "s2", "s3"],
            pending_steps=["s4", "s5", "s6", "s7"],
            accumulated_cost=0.015,
            elapsed_ms=45000.0,
            resume_count=2,
            metadata={"agent": "test"},
        )

        data = original.to_dict()
        restored = CheckpointData.from_dict(data)

        assert restored.execution_id == "e1"
        assert restored.step_index == 3
        assert restored.total_steps == 7
        assert restored.status == ExecutionStatus.CHECKPOINTED
        assert restored.state == {"key": "value", "nested": {"a": 1}}
        assert restored.accumulated_cost == 0.015
        assert restored.resume_count == 2


class TestElapsedTime:
    """Test elapsed time tracking."""

    def test_elapsed_before_start(self):
        manager = CheckpointManager()
        assert manager.get_elapsed_ms() == 0.0

    def test_elapsed_after_start(self):
        manager = CheckpointManager()
        manager.start_execution("exec-time")
        time.sleep(0.01)
        elapsed = manager.get_elapsed_ms()
        assert elapsed >= 10.0  # At least 10ms
