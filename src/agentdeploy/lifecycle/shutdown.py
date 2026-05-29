"""Graceful shutdown with in-flight request draining.

Provides SIGTERM handling, request draining with configurable timeout,
session state persistence before exit, and health endpoint updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ShutdownState(str, Enum):
    """Current state of the shutdown lifecycle."""
    RUNNING = "running"
    DRAINING = "draining"
    PERSISTING = "persisting"
    STOPPED = "stopped"


@dataclass
class RequestTracker:
    """Tracks in-flight requests for graceful draining."""
    _active_requests: int = 0
    _total_completed: int = 0
    _total_rejected: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def active_count(self) -> int:
        """Number of currently active requests."""
        with self._lock:
            return self._active_requests

    @property
    def total_completed(self) -> int:
        """Total number of completed requests."""
        with self._lock:
            return self._total_completed

    @property
    def total_rejected(self) -> int:
        """Total number of rejected requests during drain."""
        with self._lock:
            return self._total_rejected

    def acquire(self) -> bool:
        """Register a new in-flight request.

        Returns:
            True if the request was accepted, False if draining.
        """
        with self._lock:
            self._active_requests += 1
            return True

    def release(self) -> None:
        """Mark a request as completed."""
        with self._lock:
            self._active_requests -= 1
            self._total_completed += 1

    def reject(self) -> None:
        """Record a rejected request."""
        with self._lock:
            self._total_rejected += 1

    def is_drained(self) -> bool:
        """Check if all in-flight requests have completed."""
        with self._lock:
            return self._active_requests <= 0


@dataclass
class SessionState:
    """Session state to persist before shutdown."""
    data: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None

    def set(self, key: str, value: Any) -> None:
        """Set a session state value."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a session state value."""
        return self.data.get(key, default)

    def persist(self, path: Optional[str] = None) -> str:
        """Persist session state to disk.

        Args:
            path: Optional file path override.

        Returns:
            The path where state was persisted.
        """
        target_path = path or self.checkpoint_path or "/tmp/agent_session_state.json"
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

        state_data = {
            "timestamp": time.time(),
            "data": self.data,
        }

        with open(target_path, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        self.checkpoint_path = target_path
        return target_path

    @classmethod
    def load(cls, path: str) -> "SessionState":
        """Load session state from disk.

        Args:
            path: File path to load from.

        Returns:
            A SessionState instance with loaded data.
        """
        with open(path, "r") as f:
            state_data = json.load(f)

        instance = cls(checkpoint_path=path)
        instance.data = state_data.get("data", {})
        return instance


@dataclass
class HealthStatus:
    """Health endpoint status for load balancer integration."""
    state: ShutdownState = ShutdownState.RUNNING
    accepting_requests: bool = True
    active_requests: int = 0
    drain_start_time: Optional[float] = None
    drain_timeout: float = 30.0

    @property
    def is_healthy(self) -> bool:
        """Whether the service should be considered healthy."""
        return self.state == ShutdownState.RUNNING

    @property
    def drain_elapsed(self) -> float:
        """Seconds elapsed since drain started."""
        if self.drain_start_time is None:
            return 0.0
        return time.time() - self.drain_start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for health endpoint response."""
        return {
            "status": "healthy" if self.is_healthy else "draining",
            "state": self.state.value,
            "accepting_requests": self.accepting_requests,
            "active_requests": self.active_requests,
            "drain_elapsed_seconds": round(self.drain_elapsed, 2) if self.drain_start_time else None,
        }


class GracefulShutdown:
    """Manages graceful shutdown with request draining and state persistence.

    Handles SIGTERM signals, drains in-flight requests with a configurable
    timeout, persists session state, and updates health endpoint status.

    Args:
        drain_timeout: Maximum seconds to wait for in-flight requests to complete.
        state_path: Path to persist session state on shutdown.
        on_shutdown: Optional callback invoked when shutdown begins.
        on_drained: Optional callback invoked when all requests are drained.
    """

    def __init__(
        self,
        drain_timeout: float = 30.0,
        state_path: Optional[str] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
        on_drained: Optional[Callable[[], None]] = None,
    ):
        self.drain_timeout = drain_timeout
        self.state_path = state_path or "/tmp/agent_session_state.json"
        self.on_shutdown = on_shutdown
        self.on_drained = on_drained

        self._request_tracker = RequestTracker()
        self._session_state = SessionState(checkpoint_path=self.state_path)
        self._health_status = HealthStatus(drain_timeout=drain_timeout)
        self._shutdown_event = threading.Event()
        self._original_sigterm = None
        self._original_sigint = None

    @property
    def state(self) -> ShutdownState:
        """Current shutdown state."""
        return self._health_status.state

    @property
    def health_status(self) -> HealthStatus:
        """Current health status."""
        self._health_status.active_requests = self._request_tracker.active_count
        return self._health_status

    @property
    def session_state(self) -> SessionState:
        """Session state for persistence."""
        return self._session_state

    @property
    def request_tracker(self) -> RequestTracker:
        """Request tracker for in-flight monitoring."""
        return self._request_tracker

    @property
    def is_shutting_down(self) -> bool:
        """Whether shutdown has been initiated."""
        return self._shutdown_event.is_set()

    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT signal handlers.

        Should be called from the main thread.
        """
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def uninstall_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.initiate_shutdown()

    def initiate_shutdown(self) -> None:
        """Begin the graceful shutdown process.

        Transitions to DRAINING state and stops accepting new requests.
        """
        if self._shutdown_event.is_set():
            return

        self._shutdown_event.set()
        self._health_status.state = ShutdownState.DRAINING
        self._health_status.accepting_requests = False
        self._health_status.drain_start_time = time.time()

        logger.info(
            f"Shutdown initiated. Draining {self._request_tracker.active_count} "
            f"in-flight requests (timeout: {self.drain_timeout}s)"
        )

        if self.on_shutdown:
            self.on_shutdown()

    def try_accept_request(self) -> bool:
        """Try to accept a new request.

        Returns:
            True if the request is accepted, False if shutting down.
        """
        if self._shutdown_event.is_set():
            self._request_tracker.reject()
            return False
        return self._request_tracker.acquire()

    def complete_request(self) -> None:
        """Mark a request as completed."""
        self._request_tracker.release()

    def wait_for_drain(self) -> bool:
        """Wait for all in-flight requests to complete.

        Returns:
            True if all requests drained within timeout, False if timed out.
        """
        start_time = time.time()
        poll_interval = 0.1

        while not self._request_tracker.is_drained():
            elapsed = time.time() - start_time
            if elapsed >= self.drain_timeout:
                logger.warning(
                    f"Drain timeout reached ({self.drain_timeout}s). "
                    f"{self._request_tracker.active_count} requests still active."
                )
                return False
            time.sleep(poll_interval)

        logger.info("All in-flight requests drained successfully")
        if self.on_drained:
            self.on_drained()
        return True

    def persist_state(self) -> str:
        """Persist session state to disk.

        Returns:
            Path where state was persisted.
        """
        self._health_status.state = ShutdownState.PERSISTING
        logger.info(f"Persisting session state to {self.state_path}")
        path = self._session_state.persist(self.state_path)
        return path

    def shutdown(self) -> dict[str, Any]:
        """Execute the full shutdown sequence.

        1. Initiate shutdown (stop accepting requests)
        2. Wait for in-flight requests to drain
        3. Persist session state
        4. Transition to STOPPED state

        Returns:
            Summary of the shutdown process.
        """
        self.initiate_shutdown()
        drained = self.wait_for_drain()
        state_path = self.persist_state()

        self._health_status.state = ShutdownState.STOPPED
        self._health_status.active_requests = self._request_tracker.active_count

        return {
            "drained": drained,
            "active_requests_remaining": self._request_tracker.active_count,
            "total_completed": self._request_tracker.total_completed,
            "total_rejected": self._request_tracker.total_rejected,
            "state_persisted_to": state_path,
            "drain_elapsed_seconds": self._health_status.drain_elapsed,
        }
