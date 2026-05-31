"""Tests for sliding window rate limiter."""

import time
import pytest
from unittest.mock import patch

from agentdeploy.ratelimit.sliding_window import (
    SlidingWindowRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitExceeded,
)


@pytest.fixture
def limiter():
    """Create a rate limiter with default config."""
    return SlidingWindowRateLimiter(
        default_config=RateLimitConfig(max_requests=10, window_seconds=60)
    )


@pytest.fixture
def multi_endpoint_limiter():
    """Create a rate limiter with multiple endpoint configs."""
    return SlidingWindowRateLimiter(
        default_config=RateLimitConfig(max_requests=100, window_seconds=60),
        configs=[
            RateLimitConfig(max_requests=5, window_seconds=60, endpoint="/api/chat"),
            RateLimitConfig(max_requests=20, window_seconds=60, endpoint="/api/search"),
        ],
    )


class TestBasicRateLimiting:
    """Test basic rate limiting functionality."""

    def test_allows_requests_within_limit(self, limiter):
        """Test that requests within the limit are allowed."""
        for i in range(10):
            result = limiter.consume("tenant-1")
            assert result.allowed is True

    def test_blocks_requests_over_limit(self, limiter):
        """Test that requests over the limit are blocked."""
        for _ in range(10):
            limiter.consume("tenant-1")

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.consume("tenant-1")

        assert exc_info.value.limit == 10
        assert exc_info.value.retry_after > 0

    def test_different_tenants_have_separate_limits(self, limiter):
        """Test that each tenant has independent rate limits."""
        # Exhaust tenant-1's limit
        for _ in range(10):
            limiter.consume("tenant-1")

        # tenant-2 should still be allowed
        result = limiter.consume("tenant-2")
        assert result.allowed is True

    def test_remaining_count_decreases(self, limiter):
        """Test that remaining count decreases with each request."""
        result1 = limiter.consume("tenant-1")
        result2 = limiter.consume("tenant-1")

        assert result1.remaining > result2.remaining

    def test_check_does_not_consume(self, limiter):
        """Test that check() doesn't consume from the budget."""
        result1 = limiter.check("tenant-1")
        result2 = limiter.check("tenant-1")

        assert result1.remaining == result2.remaining


class TestPerEndpointLimits:
    """Test per-endpoint rate limiting."""

    def test_endpoint_specific_limits(self, multi_endpoint_limiter):
        """Test that endpoint-specific limits are applied."""
        # /api/chat has limit of 5
        for _ in range(5):
            multi_endpoint_limiter.consume("tenant-1", "/api/chat")

        with pytest.raises(RateLimitExceeded):
            multi_endpoint_limiter.consume("tenant-1", "/api/chat")

        # /api/search should still work (limit of 20)
        result = multi_endpoint_limiter.consume("tenant-1", "/api/search")
        assert result.allowed is True

    def test_default_config_for_unknown_endpoint(self, multi_endpoint_limiter):
        """Test that unknown endpoints use the default config."""
        result = multi_endpoint_limiter.check("tenant-1", "/api/unknown")
        assert result.limit == 100  # Default limit


class TestSlidingWindow:
    """Test sliding window behavior."""

    def test_window_resets_after_expiry(self, limiter):
        """Test that the window resets after the window period."""
        now = time.time()

        with patch("agentdeploy.ratelimit.sliding_window.time.time", return_value=now):
            for _ in range(10):
                limiter.consume("tenant-1")

        # Move time forward past the window
        with patch("agentdeploy.ratelimit.sliding_window.time.time", return_value=now + 61):
            result = limiter.consume("tenant-1")
            assert result.allowed is True

    def test_sliding_window_interpolation(self, limiter):
        """Test that the sliding window correctly interpolates between windows."""
        now = time.time()
        # Align to window boundary
        window_start = now - (now % 60)

        # Fill up in the previous window
        with patch("agentdeploy.ratelimit.sliding_window.time.time", return_value=window_start - 30):
            for _ in range(8):
                limiter.consume("tenant-1")

        # At the start of the new window, previous count should still partially apply
        with patch("agentdeploy.ratelimit.sliding_window.time.time", return_value=window_start + 1):
            result = limiter.check("tenant-1")
            # Most of the previous window's count should still be counted
            assert result.remaining < 10


class TestRetryAfter:
    """Test retry-after header generation."""

    def test_retry_after_in_exception(self, limiter):
        """Test that RateLimitExceeded includes retry_after."""
        for _ in range(10):
            limiter.consume("tenant-1")

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.consume("tenant-1")

        assert exc_info.value.retry_after > 0
        assert exc_info.value.retry_after <= 60

    def test_result_headers_format(self, limiter):
        """Test that result headers are properly formatted."""
        result = limiter.consume("tenant-1")
        headers = result.to_headers()

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Limit"] == "10"

    def test_blocked_result_includes_retry_after_header(self, limiter):
        """Test that blocked results include Retry-After header."""
        for _ in range(10):
            limiter.consume("tenant-1")

        result = limiter.check("tenant-1")
        headers = result.to_headers()
        assert "Retry-After" in headers


class TestUsageAndReset:
    """Test usage reporting and reset functionality."""

    def test_get_usage_reports_current_state(self, limiter):
        """Test that get_usage returns accurate state."""
        limiter.consume("tenant-1")
        limiter.consume("tenant-1")

        usage = limiter.get_usage("tenant-1")
        assert usage["tenant_id"] == "tenant-1"
        assert usage["current_count"] == 2
        assert usage["limit"] == 10
        assert usage["remaining"] == 8

    def test_reset_clears_counters(self, limiter):
        """Test that reset clears all counters for a tenant."""
        for _ in range(5):
            limiter.consume("tenant-1")

        limiter.reset("tenant-1")

        result = limiter.check("tenant-1")
        assert result.remaining == 10

    def test_burst_multiplier(self):
        """Test that burst multiplier increases effective limit."""
        config = RateLimitConfig(max_requests=10, window_seconds=60, burst_multiplier=1.5)
        limiter = SlidingWindowRateLimiter(default_config=config)

        # Should allow 15 requests (10 * 1.5)
        for _ in range(15):
            result = limiter.consume("tenant-1")
            assert result.allowed is True

        with pytest.raises(RateLimitExceeded):
            limiter.consume("tenant-1")
