"""Tests for health check endpoint and readiness probes."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agentdeploy.health.probes import (
    ComponentHealth,
    HealthCheck,
    HealthEndpoint,
    HealthResponse,
    HealthStatus,
    health_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_to_dict(self):
        component = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=5.2,
            details={"pool_size": 10},
        )
        d = component.to_dict()
        assert d["name"] == "database"
        assert d["status"] == "healthy"
        assert d["message"] == "Connected"
        assert d["latency_ms"] == 5.2
        assert d["details"]["pool_size"] == 10

    def test_default_values(self):
        component = ComponentHealth(name="test")
        assert component.status == HealthStatus.UNKNOWN
        assert component.message == ""
        assert component.latency_ms == 0.0
        assert component.details == {}


class TestHealthResponse:
    """Tests for HealthResponse."""

    def test_healthy_response(self):
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            uptime_seconds=120.5,
        )
        assert response.is_healthy is True
        assert response.http_status_code == 200

    def test_unhealthy_response(self):
        response = HealthResponse(status=HealthStatus.UNHEALTHY)
        assert response.is_healthy is False
        assert response.http_status_code == 503

    def test_degraded_response(self):
        response = HealthResponse(status=HealthStatus.DEGRADED)
        assert response.is_healthy is False
        assert response.http_status_code == 200  # Still serving

    def test_to_dict(self):
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY, message="OK"),
        ]
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            components=components,
            version="2.0.0",
            uptime_seconds=60.0,
        )
        d = response.to_dict()
        assert d["status"] == "healthy"
        assert d["version"] == "2.0.0"
        assert len(d["components"]) == 1
        assert d["components"][0]["name"] == "db"


class TestHealthCheck:
    """Tests for HealthCheck manager."""

    def test_register_and_check(self):
        """Test registering a check and running it."""
        hc = HealthCheck(version="1.0.0")
        hc.register("database", lambda: True, critical=True)

        response = hc.check_health()
        assert response.status == HealthStatus.HEALTHY
        assert len(response.components) == 1
        assert response.components[0].name == "database"
        assert response.components[0].status == HealthStatus.HEALTHY

    def test_failing_critical_check(self):
        """Test that a failing critical check makes overall status unhealthy."""
        hc = HealthCheck()
        hc.register("database", lambda: False, critical=True)
        hc.register("cache", lambda: True, critical=False)

        response = hc.check_health()
        assert response.status == HealthStatus.UNHEALTHY

    def test_failing_non_critical_check(self):
        """Test that a failing non-critical check makes status degraded."""
        hc = HealthCheck()
        hc.register("database", lambda: True, critical=True)
        hc.register("cache", lambda: False, critical=False)

        response = hc.check_health()
        assert response.status == HealthStatus.DEGRADED

    def test_all_healthy(self):
        """Test all checks passing."""
        hc = HealthCheck(version="1.2.3")
        hc.register("db", lambda: True)
        hc.register("cache", lambda: True)
        hc.register("queue", lambda: True)

        response = hc.check_health()
        assert response.status == HealthStatus.HEALTHY
        assert response.version == "1.2.3"
        assert len(response.components) == 3

    def test_dict_return_value(self):
        """Test check that returns a dictionary."""
        def check_db():
            return {
                "status": "healthy",
                "message": "Connected to primary",
                "connections": 5,
                "max_connections": 20,
            }

        hc = HealthCheck()
        hc.register("database", check_db)

        response = hc.check_health()
        assert response.status == HealthStatus.HEALTHY
        db_component = response.components[0]
        assert db_component.message == "Connected to primary"
        assert db_component.details["connections"] == 5

    def test_component_health_return(self):
        """Test check that returns a ComponentHealth directly."""
        def check_custom():
            return ComponentHealth(
                name="custom",
                status=HealthStatus.DEGRADED,
                message="High latency detected",
                details={"avg_latency_ms": 500},
            )

        hc = HealthCheck()
        hc.register("custom", check_custom, critical=False)

        response = hc.check_health()
        assert response.status == HealthStatus.DEGRADED
        assert response.components[0].message == "High latency detected"

    def test_exception_handling(self):
        """Test that exceptions in checks are handled gracefully."""
        def failing_check():
            raise ConnectionError("Connection refused")

        hc = HealthCheck()
        hc.register("database", failing_check, critical=True)

        response = hc.check_health()
        assert response.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in response.components[0].message

    def test_latency_tracking(self):
        """Test that check latency is measured."""
        def slow_check():
            time.sleep(0.05)
            return True

        hc = HealthCheck()
        hc.register("slow", slow_check)

        response = hc.check_health()
        assert response.components[0].latency_ms >= 40  # At least 40ms

    def test_uptime_tracking(self):
        """Test uptime calculation."""
        start = time.time() - 100  # Started 100 seconds ago
        hc = HealthCheck(start_time=start)

        assert hc.uptime_seconds >= 99.0

    def test_readiness_checks(self):
        """Test readiness-specific checks."""
        hc = HealthCheck()
        hc.register("db", lambda: True, readiness=True)
        hc.register("cache", lambda: True, readiness=False)
        hc.register("warmup", lambda: True, readiness=True)

        response = hc.check_ready()
        # Only readiness checks should be included
        assert len(response.components) == 2
        component_names = [c.name for c in response.components]
        assert "db" in component_names
        assert "warmup" in component_names
        assert "cache" not in component_names

    def test_liveness_always_healthy(self):
        """Test that liveness check always returns healthy."""
        hc = HealthCheck(version="1.0.0")
        hc.register("db", lambda: False, critical=True)

        response = hc.check_liveness()
        assert response.status == HealthStatus.HEALTHY

    def test_unregister_check(self):
        """Test unregistering a health check."""
        hc = HealthCheck()
        hc.register("db", lambda: True)
        hc.register("cache", lambda: True)

        assert hc.unregister("db") is True
        assert "db" not in hc.registered_checks
        assert hc.unregister("db") is False

    def test_no_checks_is_healthy(self):
        """Test that no registered checks means healthy."""
        hc = HealthCheck()
        response = hc.check_health()
        assert response.status == HealthStatus.HEALTHY


class TestHealthEndpoint:
    """Tests for HealthEndpoint HTTP handler."""

    def test_health_endpoint_healthy(self):
        """Test /health endpoint when all checks pass."""
        hc = HealthCheck(version="1.0.0")
        hc.register("db", lambda: True)
        endpoint = HealthEndpoint(hc)

        result = endpoint.health_handler()
        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"

    def test_health_endpoint_unhealthy(self):
        """Test /health endpoint when critical check fails."""
        hc = HealthCheck()
        hc.register("db", lambda: False, critical=True)
        endpoint = HealthEndpoint(hc)

        result = endpoint.health_handler()
        assert result["status_code"] == 503
        assert result["body"]["status"] == "unhealthy"

    def test_ready_endpoint(self):
        """Test /ready endpoint."""
        hc = HealthCheck()
        hc.register("warmup", lambda: True, readiness=True)
        endpoint = HealthEndpoint(hc)

        result = endpoint.ready_handler()
        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"

    def test_liveness_endpoint(self):
        """Test /live endpoint always returns 200."""
        hc = HealthCheck()
        hc.register("db", lambda: False, critical=True)
        endpoint = HealthEndpoint(hc)

        result = endpoint.liveness_handler()
        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"

    def test_endpoint_includes_components(self):
        """Test that endpoint response includes component details."""
        hc = HealthCheck(version="2.0.0")
        hc.register("database", lambda: {"status": "healthy", "pool": 5})
        hc.register("redis", lambda: True)
        endpoint = HealthEndpoint(hc)

        result = endpoint.health_handler()
        body = result["body"]
        assert len(body["components"]) == 2
        assert body["version"] == "2.0.0"


class TestHealthCheckDecorator:
    """Tests for the @health_check decorator."""

    def test_decorator_attaches_metadata(self):
        """Test that decorator attaches health check metadata."""
        @health_check(name="my_check", critical=True, timeout_ms=3000)
        def my_check():
            return True

        assert my_check._health_check_name == "my_check"
        assert my_check._health_check_critical is True
        assert my_check._health_check_timeout_ms == 3000

    def test_decorator_default_name(self):
        """Test that decorator uses function name as default."""
        @health_check()
        def check_database():
            return True

        assert check_database._health_check_name == "check_database"

    def test_decorated_function_still_callable(self):
        """Test that decorated function can still be called normally."""
        @health_check(name="test")
        def check_something():
            return {"status": "healthy", "detail": "all good"}

        result = check_something()
        assert result["status"] == "healthy"
