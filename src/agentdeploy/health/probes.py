"""Health check endpoint and readiness probes for AgentDeploy.

Provides /health and /ready endpoints with component-level status checks,
structured JSON responses, and configurable health check registration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger("agentdeploy.health")


class HealthStatus(Enum):
    """Health status values for components and overall system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component.

    Attributes:
        name: Component identifier.
        status: Current health status.
        message: Human-readable status message.
        latency_ms: Time taken to check this component.
        details: Additional component-specific details.
        last_checked: Timestamp of last health check.
    """
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    last_checked: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "last_checked": self.last_checked,
        }


@dataclass
class HealthResponse:
    """Structured health check response.

    Attributes:
        status: Overall system health status.
        components: Individual component health statuses.
        version: Application version.
        uptime_seconds: Time since the service started.
        timestamp: Current timestamp.
    """
    status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    version: str = ""
    uptime_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Check if the overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def http_status_code(self) -> int:
        """Get the appropriate HTTP status code."""
        if self.status == HealthStatus.HEALTHY:
            return 200
        elif self.status == HealthStatus.DEGRADED:
            return 200  # Still serving, but degraded
        else:
            return 503

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp,
            "components": [c.to_dict() for c in self.components],
        }


# Type alias for check functions
CheckFunction = Union[
    Callable[[], bool],
    Callable[[], Dict[str, Any]],
    Callable[[], ComponentHealth],
    Callable[[], Coroutine[Any, Any, bool]],
    Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
    Callable[[], Coroutine[Any, Any, ComponentHealth]],
]


@dataclass
class RegisteredCheck:
    """A registered health check with metadata."""
    name: str
    check_fn: CheckFunction
    critical: bool = True  # If critical, failure = UNHEALTHY; else DEGRADED
    timeout_ms: float = 5000.0
    is_async: bool = False


class HealthCheck:
    """Health check manager with component-level status tracking.

    Manages registered health checks, executes them on demand,
    and aggregates results into structured responses.

    Usage:
        health = HealthCheck(version="1.2.3")

        # Register checks
        health.register("database", check_database, critical=True)
        health.register("cache", check_cache, critical=False)

        # Run all checks
        response = health.check_health()
        print(response.to_dict())

        # Check readiness
        response = health.check_ready()
    """

    def __init__(
        self,
        version: str = "",
        start_time: Optional[float] = None,
    ):
        self._version = version
        self._start_time = start_time or time.time()
        self._checks: Dict[str, RegisteredCheck] = {}
        self._readiness_checks: Dict[str, RegisteredCheck] = {}
        self._last_response: Optional[HealthResponse] = None

    @property
    def version(self) -> str:
        """Get the application version."""
        return self._version

    @property
    def uptime_seconds(self) -> float:
        """Get the uptime in seconds."""
        return time.time() - self._start_time

    @property
    def registered_checks(self) -> List[str]:
        """List all registered health check names."""
        return list(self._checks.keys())

    @property
    def last_response(self) -> Optional[HealthResponse]:
        """Get the last health check response."""
        return self._last_response

    def register(
        self,
        name: str,
        check_fn: CheckFunction,
        critical: bool = True,
        timeout_ms: float = 5000.0,
        readiness: bool = False,
    ) -> None:
        """Register a health check function.

        Args:
            name: Unique name for this check.
            check_fn: Function that performs the health check.
            critical: If True, failure makes overall status UNHEALTHY.
            timeout_ms: Maximum time allowed for this check.
            readiness: If True, also include in readiness checks.
        """
        is_async = asyncio.iscoroutinefunction(check_fn)

        registered = RegisteredCheck(
            name=name,
            check_fn=check_fn,
            critical=critical,
            timeout_ms=timeout_ms,
            is_async=is_async,
        )

        self._checks[name] = registered

        if readiness:
            self._readiness_checks[name] = registered

    def unregister(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: The check name to remove.

        Returns:
            True if the check was found and removed.
        """
        removed = name in self._checks
        self._checks.pop(name, None)
        self._readiness_checks.pop(name, None)
        return removed

    def check_health(self) -> HealthResponse:
        """Run all health checks and return aggregated response.

        Returns:
            HealthResponse with overall status and component details.
        """
        components = []

        for name, check in self._checks.items():
            component = self._execute_check(check)
            components.append(component)

        overall_status = self._aggregate_status(components)

        response = HealthResponse(
            status=overall_status,
            components=components,
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

        self._last_response = response
        return response

    def check_ready(self) -> HealthResponse:
        """Run readiness checks and return response.

        Readiness checks determine if the service is ready to accept traffic.
        This is typically used by load balancers and orchestrators.

        Returns:
            HealthResponse for readiness status.
        """
        if not self._readiness_checks:
            # If no readiness checks registered, use all checks
            return self.check_health()

        components = []

        for name, check in self._readiness_checks.items():
            component = self._execute_check(check)
            components.append(component)

        overall_status = self._aggregate_status(components)

        return HealthResponse(
            status=overall_status,
            components=components,
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

    def check_liveness(self) -> HealthResponse:
        """Simple liveness check - just confirms the process is running.

        Returns:
            HealthResponse that is always healthy (process is alive).
        """
        return HealthResponse(
            status=HealthStatus.HEALTHY,
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

    def _execute_check(self, check: RegisteredCheck) -> ComponentHealth:
        """Execute a single health check with timeout handling."""
        start = time.time()

        try:
            if check.is_async:
                result = self._run_async_check(check)
            else:
                result = check.check_fn()

            latency_ms = (time.time() - start) * 1000
            return self._normalize_result(check.name, result, latency_ms)

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.warning(f"Health check '{check.name}' failed: {e}")
            return ComponentHealth(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
            )

    def _run_async_check(self, check: RegisteredCheck) -> Any:
        """Run an async check function."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, check.check_fn())
                    return future.result(timeout=check.timeout_ms / 1000)
            else:
                return loop.run_until_complete(check.check_fn())
        except RuntimeError:
            return asyncio.run(check.check_fn())

    def _normalize_result(
        self, name: str, result: Any, latency_ms: float
    ) -> ComponentHealth:
        """Normalize various check return types into ComponentHealth."""
        if isinstance(result, ComponentHealth):
            result.latency_ms = latency_ms
            result.last_checked = time.time()
            return result

        if isinstance(result, bool):
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message="OK" if result else "Check returned False",
                latency_ms=latency_ms,
            )

        if isinstance(result, dict):
            status_str = result.get("status", "healthy" if result.get("healthy", True) else "unhealthy")
            try:
                status = HealthStatus(status_str)
            except ValueError:
                status = HealthStatus.HEALTHY if result.get("healthy", True) else HealthStatus.UNHEALTHY

            return ComponentHealth(
                name=name,
                status=status,
                message=result.get("message", ""),
                latency_ms=latency_ms,
                details={k: v for k, v in result.items() if k not in ("status", "message", "healthy")},
            )

        # Fallback: truthy = healthy
        return ComponentHealth(
            name=name,
            status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
        )

    def _aggregate_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Aggregate component statuses into overall status."""
        if not components:
            return HealthStatus.HEALTHY

        has_unhealthy_critical = False
        has_degraded = False

        for component in components:
            check = self._checks.get(component.name)
            is_critical = check.critical if check else True

            if component.status == HealthStatus.UNHEALTHY:
                if is_critical:
                    has_unhealthy_critical = True
                else:
                    has_degraded = True
            elif component.status == HealthStatus.DEGRADED:
                has_degraded = True

        if has_unhealthy_critical:
            return HealthStatus.UNHEALTHY
        elif has_degraded:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


class HealthEndpoint:
    """HTTP endpoint handler for health checks.

    Provides /health and /ready endpoint handlers that can be integrated
    with any ASGI/WSGI framework.

    Usage with a generic framework:
        health_check = HealthCheck(version="1.0.0")
        endpoint = HealthEndpoint(health_check)

        # Register routes
        app.get("/health")(endpoint.health_handler)
        app.get("/ready")(endpoint.ready_handler)
        app.get("/live")(endpoint.liveness_handler)
    """

    def __init__(self, health_check: HealthCheck):
        self._health_check = health_check

    def health_handler(self) -> Dict[str, Any]:
        """Handle /health endpoint requests.

        Returns:
            Tuple of (response_dict, http_status_code).
        """
        response = self._health_check.check_health()
        return {
            "body": response.to_dict(),
            "status_code": response.http_status_code,
        }

    def ready_handler(self) -> Dict[str, Any]:
        """Handle /ready endpoint requests.

        Returns:
            Tuple of (response_dict, http_status_code).
        """
        response = self._health_check.check_ready()
        return {
            "body": response.to_dict(),
            "status_code": response.http_status_code,
        }

    def liveness_handler(self) -> Dict[str, Any]:
        """Handle /live endpoint requests.

        Returns:
            Tuple of (response_dict, http_status_code) - always 200.
        """
        response = self._health_check.check_liveness()
        return {
            "body": response.to_dict(),
            "status_code": 200,
        }


def health_check(
    name: Optional[str] = None,
    critical: bool = True,
    timeout_ms: float = 5000.0,
) -> Callable:
    """Decorator to mark a function as a health check.

    The decorated function will have metadata attached that can be
    used for auto-registration with a HealthCheck instance.

    Example:
        @health_check(name="database", critical=True)
        def check_database():
            conn = get_db_connection()
            conn.execute("SELECT 1")
            return True
    """
    def decorator(func: Callable) -> Callable:
        func._health_check_name = name or func.__name__
        func._health_check_critical = critical
        func._health_check_timeout_ms = timeout_ms
        return func

    return decorator
