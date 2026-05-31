"""Request rate limiting with sliding window counter algorithm.

Provides per-tenant, per-endpoint rate limiting with configurable windows.
Returns 429-style responses with retry-after headers when limits are exceeded.

The sliding window counter algorithm combines the benefits of fixed window
(memory efficiency) and sliding log (accuracy) approaches by interpolating
between the current and previous window counts.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("agentdeploy.ratelimit")


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded.

    Attributes:
        retry_after: Seconds until the client can retry.
        limit: The configured limit that was exceeded.
        remaining: Number of requests remaining (always 0 when raised).
    """

    def __init__(self, retry_after: float, limit: int, endpoint: str = ""):
        self.retry_after = retry_after
        self.limit = limit
        self.endpoint = endpoint
        super().__init__(
            f"Rate limit exceeded for endpoint '{endpoint}': "
            f"{limit} requests per window. Retry after {retry_after:.1f}s"
        )


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule.

    Attributes:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Duration of the sliding window in seconds.
        endpoint: The endpoint pattern this rule applies to (e.g., "/api/chat").
        burst_multiplier: Multiplier for burst allowance (default: 1.0, no burst).
    """
    max_requests: int
    window_seconds: int = 60
    endpoint: str = "*"
    burst_multiplier: float = 1.0

    @property
    def effective_limit(self) -> int:
        """Return the effective limit including burst allowance."""
        return int(self.max_requests * self.burst_multiplier)


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        remaining: Number of requests remaining in the current window.
        limit: The configured limit.
        retry_after: Seconds until the window resets (only set if not allowed).
        reset_at: Unix timestamp when the current window resets.
    """
    allowed: bool
    remaining: int
    limit: int
    retry_after: float = 0.0
    reset_at: float = 0.0

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP response headers (RFC 6585 style)."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(int(self.retry_after) + 1)
        return headers


@dataclass
class _WindowCounter:
    """Internal counter for a single fixed window."""
    count: int = 0
    window_start: float = 0.0


class SlidingWindowRateLimiter:
    """Sliding window counter rate limiter.

    Implements the sliding window counter algorithm which provides a good
    balance between accuracy and memory efficiency. It works by:

    1. Maintaining counters for the current and previous fixed windows
    2. Estimating the request count in the sliding window by interpolating:
       count = prev_window_count * overlap_ratio + current_window_count

    This avoids the boundary burst problem of fixed windows while using
    O(1) memory per tenant/endpoint combination.

    Args:
        default_config: Default rate limit config applied to all endpoints.
        configs: List of endpoint-specific rate limit configs.

    Example:
        limiter = SlidingWindowRateLimiter(
            default_config=RateLimitConfig(max_requests=100, window_seconds=60),
            configs=[
                RateLimitConfig(max_requests=10, window_seconds=60, endpoint="/api/chat"),
            ]
        )

        result = limiter.check("tenant-123", "/api/chat")
        if not result.allowed:
            return Response(status=429, headers=result.to_headers())
    """

    def __init__(
        self,
        default_config: RateLimitConfig | None = None,
        configs: list[RateLimitConfig] | None = None,
    ):
        self.default_config = default_config or RateLimitConfig(
            max_requests=100, window_seconds=60
        )
        self._endpoint_configs: dict[str, RateLimitConfig] = {}
        for config in (configs or []):
            self._endpoint_configs[config.endpoint] = config

        # Storage: {(tenant_id, endpoint): {window_key: _WindowCounter}}
        self._counters: dict[tuple[str, str], dict[str, _WindowCounter]] = defaultdict(dict)

    def check(self, tenant_id: str, endpoint: str = "*") -> RateLimitResult:
        """Check if a request is within rate limits (does not consume).

        Args:
            tenant_id: The tenant making the request.
            endpoint: The endpoint being accessed.

        Returns:
            RateLimitResult indicating whether the request is allowed.
        """
        config = self._get_config(endpoint)
        now = time.time()
        count = self._get_sliding_count(tenant_id, endpoint, config, now)
        limit = config.effective_limit

        remaining = max(0, limit - int(count))
        window_end = self._get_window_end(now, config.window_seconds)

        if count >= limit:
            retry_after = window_end - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                retry_after=retry_after,
                reset_at=window_end,
            )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=limit,
            reset_at=window_end,
        )

    def consume(self, tenant_id: str, endpoint: str = "*") -> RateLimitResult:
        """Consume a request from the rate limit budget.

        Args:
            tenant_id: The tenant making the request.
            endpoint: The endpoint being accessed.

        Returns:
            RateLimitResult indicating whether the request was allowed.

        Raises:
            RateLimitExceeded: If the rate limit has been exceeded.
        """
        config = self._get_config(endpoint)
        now = time.time()
        count = self._get_sliding_count(tenant_id, endpoint, config, now)
        limit = config.effective_limit

        window_end = self._get_window_end(now, config.window_seconds)

        if count >= limit:
            retry_after = window_end - now
            logger.warning(
                f"Rate limit exceeded: tenant={tenant_id} endpoint={endpoint} "
                f"count={count:.1f} limit={limit}"
            )
            raise RateLimitExceeded(
                retry_after=retry_after,
                limit=limit,
                endpoint=endpoint,
            )

        # Increment current window counter
        self._increment(tenant_id, endpoint, config, now)

        remaining = max(0, limit - int(count) - 1)
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=limit,
            reset_at=window_end,
        )

    def get_usage(self, tenant_id: str, endpoint: str = "*") -> dict:
        """Get current usage statistics for a tenant/endpoint.

        Returns:
            Dict with current count, limit, remaining, and reset time.
        """
        config = self._get_config(endpoint)
        now = time.time()
        count = self._get_sliding_count(tenant_id, endpoint, config, now)
        limit = config.effective_limit

        return {
            "tenant_id": tenant_id,
            "endpoint": endpoint,
            "current_count": int(count),
            "limit": limit,
            "remaining": max(0, limit - int(count)),
            "window_seconds": config.window_seconds,
            "reset_at": self._get_window_end(now, config.window_seconds),
        }

    def reset(self, tenant_id: str, endpoint: str = "*") -> None:
        """Reset rate limit counters for a tenant/endpoint.

        Args:
            tenant_id: The tenant to reset.
            endpoint: The endpoint to reset (or "*" for all).
        """
        key = (tenant_id, endpoint)
        if key in self._counters:
            del self._counters[key]

    def add_config(self, config: RateLimitConfig) -> None:
        """Add or update an endpoint-specific rate limit config."""
        self._endpoint_configs[config.endpoint] = config

    def _get_config(self, endpoint: str) -> RateLimitConfig:
        """Get the rate limit config for an endpoint."""
        if endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]
        # Check for wildcard/prefix matches
        for pattern, config in self._endpoint_configs.items():
            if pattern.endswith("*") and endpoint.startswith(pattern[:-1]):
                return config
        return self.default_config

    def _get_sliding_count(
        self, tenant_id: str, endpoint: str, config: RateLimitConfig, now: float
    ) -> float:
        """Calculate the sliding window count using interpolation."""
        window_size = config.window_seconds
        current_window_start = self._get_window_start(now, window_size)
        prev_window_start = current_window_start - window_size

        key = (tenant_id, endpoint)
        counters = self._counters[key]

        current_key = str(int(current_window_start))
        prev_key = str(int(prev_window_start))

        current_count = counters.get(current_key, _WindowCounter()).count
        prev_count = counters.get(prev_key, _WindowCounter()).count

        # Calculate overlap ratio: how much of the previous window is in our sliding window
        elapsed_in_current = now - current_window_start
        overlap_ratio = 1.0 - (elapsed_in_current / window_size)

        # Sliding window estimate
        sliding_count = prev_count * overlap_ratio + current_count
        return sliding_count

    def _increment(
        self, tenant_id: str, endpoint: str, config: RateLimitConfig, now: float
    ) -> None:
        """Increment the counter for the current window."""
        window_size = config.window_seconds
        current_window_start = self._get_window_start(now, window_size)
        current_key = str(int(current_window_start))

        key = (tenant_id, endpoint)
        if current_key not in self._counters[key]:
            self._counters[key][current_key] = _WindowCounter(
                count=0, window_start=current_window_start
            )

        self._counters[key][current_key].count += 1

        # Cleanup old windows (keep only current and previous)
        prev_window_start = current_window_start - window_size
        prev_key = str(int(prev_window_start))
        keys_to_remove = [
            k for k in self._counters[key]
            if k != current_key and k != prev_key
        ]
        for k in keys_to_remove:
            del self._counters[key][k]

    @staticmethod
    def _get_window_start(now: float, window_size: int) -> float:
        """Get the start time of the current fixed window."""
        return now - (now % window_size)

    @staticmethod
    def _get_window_end(now: float, window_size: int) -> float:
        """Get the end time of the current fixed window."""
        return SlidingWindowRateLimiter._get_window_start(now, window_size) + window_size
