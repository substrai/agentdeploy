"""Tests for graceful shutdown with in-flight request draining."""

import json
import os
import signal
import tempfile
import threading
import time

import pytest

from agentdeploy.lifecycle.shutdown import (
    GracefulShutdown,
    HealthStatus,
    RequestTracker,
    SessionState,
    ShutdownState,
)


class TestRequestTracker:
    """Tests for in-flight request tracking."""

    def test_acquire_and_release(self):
        """Test basic request acquire and release."""
        tracker = RequestTracker()
        tracker.acquire()
        assert tracker.active_count == 1
        tracker.release()
        assert tracker.active_count == 0
        assert tracker.total_completed == 1

    def test_is_drained_when_empty(self):
        """Test drained state with no active requests."""
        tracker = RequestTracker()
        assert tracker.is_drained() is True

    def test_is_not_drained_with_active(self):
        """Test not drained with active requests."""
        tracker = RequestTracker()
        tracker.acquire()
        assert tracker.is_drained() is False

    def test_reject_increments_counter(self):
        """Test that reject increments the rejected counter."""
        tracker = RequestTracker()
        tracker.reject()
        tracker.reject()
        assert tracker.total_rejected == 2


class TestSessionState:
    """Tests for session state persistence."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        state = SessionState()
        state.set("model_id", "gpt-4")
        assert state.get("model_id") == "gpt-4"

    def test_get_default_value(self):
        """Test get with default value for missing key."""
        state = SessionState()
        assert state.get("missing", "default") == "default"

    def test_persist_and_load(self):
        """Test persisting and loading state from disk."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            state = SessionState(checkpoint_path=path)
            state.set("key1", "value1")
            state.set("key2", 42)
            state.persist()

            loaded = SessionState.load(path)
            assert loaded.get("key1") == "value1"
            assert loaded.get("key2") == 42
        finally:
            os.unlink(path)


class TestHealthStatus:
    """Tests for health endpoint status."""

    def test_healthy_when_running(self):
        """Test health status is healthy when running."""
        status = HealthStatus(state=ShutdownState.RUNNING)
        assert status.is_healthy is True

    def test_unhealthy_when_draining(self):
        """Test health status is unhealthy when draining."""
        status = HealthStatus(state=ShutdownState.DRAINING)
        assert status.is_healthy is False

    def test_to_dict_format(self):
        """Test health status dictionary format."""
        status = HealthStatus(state=ShutdownState.RUNNING, active_requests=5)
        result = status.to_dict()
        assert result["status"] == "healthy"
        assert result["active_requests"] == 5
        assert result["accepting_requests"] is True


class TestGracefulShutdown:
    """Tests for the full graceful shutdown lifecycle."""

    def test_initial_state_is_running(self):
        """Test that initial state is RUNNING."""
        shutdown = GracefulShutdown()
        assert shutdown.state == ShutdownState.RUNNING
        assert shutdown.is_shutting_down is False

    def test_initiate_shutdown_transitions_to_draining(self):
        """Test that initiating shutdown transitions state."""
        shutdown = GracefulShutdown()
        shutdown.initiate_shutdown()
        assert shutdown.state == ShutdownState.DRAINING
        assert shutdown.is_shutting_down is True
        assert shutdown.health_status.accepting_requests is False

    def test_try_accept_request_rejected_during_shutdown(self):
        """Test that new requests are rejected during shutdown."""
        shutdown = GracefulShutdown()
        assert shutdown.try_accept_request() is True
        shutdown.complete_request()

        shutdown.initiate_shutdown()
        assert shutdown.try_accept_request() is False
        assert shutdown.request_tracker.total_rejected == 1

    def test_wait_for_drain_succeeds(self):
        """Test successful drain of in-flight requests."""
        shutdown = GracefulShutdown(drain_timeout=5.0)
        shutdown.try_accept_request()

        # Complete request in background
        def complete():
            time.sleep(0.1)
            shutdown.complete_request()

        threading.Thread(target=complete).start()
        shutdown.initiate_shutdown()
        result = shutdown.wait_for_drain()
        assert result is True

    def test_wait_for_drain_timeout(self):
        """Test drain timeout with stuck requests."""
        shutdown = GracefulShutdown(drain_timeout=0.2)
        shutdown.try_accept_request()  # Never completed
        shutdown.initiate_shutdown()
        result = shutdown.wait_for_drain()
        assert result is False

    def test_full_shutdown_sequence(self):
        """Test the complete shutdown sequence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            shutdown = GracefulShutdown(drain_timeout=1.0, state_path=path)
            shutdown.session_state.set("conversation_id", "abc-123")

            # Simulate a request that completes quickly
            shutdown.try_accept_request()
            shutdown.complete_request()

            result = shutdown.shutdown()
            assert result["drained"] is True
            assert result["total_completed"] == 1
            assert result["state_persisted_to"] == path
            assert shutdown.state == ShutdownState.STOPPED

            # Verify state was persisted
            with open(path) as f:
                data = json.load(f)
            assert data["data"]["conversation_id"] == "abc-123"
        finally:
            os.unlink(path)

    def test_on_shutdown_callback(self):
        """Test that on_shutdown callback is invoked."""
        called = []
        shutdown = GracefulShutdown(on_shutdown=lambda: called.append(True))
        shutdown.initiate_shutdown()
        assert len(called) == 1

    def test_on_drained_callback(self):
        """Test that on_drained callback is invoked."""
        called = []
        shutdown = GracefulShutdown(
            drain_timeout=1.0,
            on_drained=lambda: called.append(True),
        )
        shutdown.initiate_shutdown()
        shutdown.wait_for_drain()
        assert len(called) == 1
