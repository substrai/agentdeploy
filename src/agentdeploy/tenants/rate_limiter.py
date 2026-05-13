"""Per-tenant rate limiting."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    tenant_id: str
    current_rate: int  # requests in current window
    limit: int  # max requests per window
    remaining: int
    retry_after_seconds: float = 0.0
    message: str = ""


class RateLimiter:
    """Token bucket rate limiter per tenant.

    Usage:
        limiter = RateLimiter()
        limiter.set_limit("tenant-a", 100)  # 100 req/min
        result = limiter.check("tenant-a")
        if not result.allowed:
            return 429, result.message
    """

    def __init__(self, default_limit: int = 100, window_seconds: int = 60):
        self._limits: Dict[str, int] = {}
        self._windows: Dict[str, deque] = {}
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    def set_limit(self, tenant_id: str, requests_per_minute: int) -> None:
        """Set rate limit for a tenant."""
        self._limits[tenant_id] = requests_per_minute

    def check(self, tenant_id: str) -> RateLimitResult:
        """Check if a request is within rate limits.

        Args:
            tenant_id: Tenant making the request

        Returns:
            RateLimitResult with allow/deny decision
        """
        limit = self._limits.get(tenant_id, self.default_limit)
        now = time.time()
        cutoff = now - self.window_seconds

        # Initialize window if needed
        if tenant_id not in self._windows:
            self._windows[tenant_id] = deque()

        window = self._windows[tenant_id]

        # Remove expired entries
        while window and window[0] < cutoff:
            window.popleft()

        current_rate = len(window)

        if current_rate >= limit:
            # Calculate retry after
            oldest = window[0] if window else now
            retry_after = oldest + self.window_seconds - now
            return RateLimitResult(
                allowed=False,
                tenant_id=tenant_id,
                current_rate=current_rate,
                limit=limit,
                remaining=0,
                retry_after_seconds=max(retry_after, 0),
                message=f"Rate limit exceeded: {current_rate}/{limit} requests per {self.window_seconds}s",
            )

        # Record this request
        window.append(now)

        return RateLimitResult(
            allowed=True,
            tenant_id=tenant_id,
            current_rate=current_rate + 1,
            limit=limit,
            remaining=limit - current_rate - 1,
            message="OK",
        )

    def check_only(self, tenant_id: str) -> RateLimitResult:
        """Check without recording (peek)."""
        limit = self._limits.get(tenant_id, self.default_limit)
        now = time.time()
        cutoff = now - self.window_seconds

        if tenant_id not in self._windows:
            return RateLimitResult(allowed=True, tenant_id=tenant_id, current_rate=0, limit=limit, remaining=limit, message="OK")

        window = self._windows[tenant_id]
        current = sum(1 for t in window if t >= cutoff)

        return RateLimitResult(
            allowed=current < limit,
            tenant_id=tenant_id,
            current_rate=current,
            limit=limit,
            remaining=max(limit - current, 0),
        )

    def get_usage(self, tenant_id: str) -> Dict[str, int]:
        """Get current usage for a tenant."""
        limit = self._limits.get(tenant_id, self.default_limit)
        now = time.time()
        cutoff = now - self.window_seconds

        if tenant_id not in self._windows:
            return {"current": 0, "limit": limit, "remaining": limit}

        current = sum(1 for t in self._windows[tenant_id] if t >= cutoff)
        return {"current": current, "limit": limit, "remaining": max(limit - current, 0)}

    def reset(self, tenant_id: str) -> None:
        """Reset rate limit window for a tenant."""
        if tenant_id in self._windows:
            self._windows[tenant_id].clear()
