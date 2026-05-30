"""Per-tenant API key management with rotation.

Provides generation, revocation, and automatic rotation of API keys
per tenant. Keys are stored as hashes with expiry tracking and
rotation notifications.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class KeyStatus(str, Enum):
    """Status of an API key."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATING = "rotating"


@dataclass
class APIKey:
    """Represents an API key with metadata."""

    key_id: str
    tenant_id: str
    key_hash: str
    prefix: str  # First 8 chars for identification
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used_at: Optional[float] = None
    rotation_scheduled_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if the key is currently active and not expired."""
        return self.status == KeyStatus.ACTIVE and not self.is_expired

    @property
    def days_until_expiry(self) -> Optional[float]:
        """Days remaining until expiry."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0.0, remaining / 86400)


@dataclass
class RotationPolicy:
    """Policy for automatic key rotation."""

    rotation_interval_days: float = 90.0
    grace_period_days: float = 7.0
    notify_before_days: float = 14.0
    auto_rotate: bool = True
    max_active_keys_per_tenant: int = 2

    @property
    def rotation_interval_seconds(self) -> float:
        return self.rotation_interval_days * 86400

    @property
    def grace_period_seconds(self) -> float:
        return self.grace_period_days * 86400

    @property
    def notify_before_seconds(self) -> float:
        return self.notify_before_days * 86400


@dataclass
class RotationEvent:
    """Event emitted during key rotation."""

    tenant_id: str
    old_key_id: str
    new_key_id: str
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class KeyValidationResult:
    """Result of validating an API key."""

    valid: bool
    tenant_id: Optional[str] = None
    key_id: Optional[str] = None
    error: Optional[str] = None


class APIKeyManager:
    """Manages per-tenant API keys with rotation support.

    Handles key generation, validation, revocation, and automatic
    rotation scheduling. Keys are stored as SHA-256 hashes.

    Example:
        >>> manager = APIKeyManager(rotation_policy=RotationPolicy(rotation_interval_days=90))
        >>> raw_key = manager.generate_key("tenant-123")
        >>> result = manager.validate_key(raw_key)
        >>> assert result.valid and result.tenant_id == "tenant-123"
    """

    KEY_PREFIX = "ad_"  # agentdeploy prefix
    KEY_LENGTH = 48  # bytes of randomness

    def __init__(
        self,
        rotation_policy: Optional[RotationPolicy] = None,
        on_rotation: Optional[Callable[[RotationEvent], None]] = None,
        on_expiry_warning: Optional[Callable[[APIKey, float], None]] = None,
    ) -> None:
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey
        self._tenant_keys: dict[str, list[str]] = {}  # tenant_id -> [key_ids]
        self._rotation_policy = rotation_policy or RotationPolicy()
        self._on_rotation = on_rotation
        self._on_expiry_warning = on_expiry_warning

    @property
    def rotation_policy(self) -> RotationPolicy:
        """Current rotation policy."""
        return self._rotation_policy

    def generate_key(
        self,
        tenant_id: str,
        expires_in_days: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a new API key for a tenant.

        Args:
            tenant_id: The tenant to generate the key for.
            expires_in_days: Optional custom expiry in days.
            metadata: Optional metadata to attach to the key.

        Returns:
            The raw API key string (only returned once).
        """
        # Generate cryptographically secure random key
        raw_bytes = secrets.token_bytes(self.KEY_LENGTH)
        raw_key = self.KEY_PREFIX + secrets.token_urlsafe(self.KEY_LENGTH)

        # Compute hash for storage
        key_hash = self._hash_key(raw_key)
        key_id = self._generate_key_id()
        prefix = raw_key[:12]

        # Calculate expiry
        expires_at = None
        if expires_in_days is not None:
            expires_at = time.time() + (expires_in_days * 86400)
        elif self._rotation_policy.rotation_interval_days:
            expires_at = time.time() + self._rotation_policy.rotation_interval_seconds

        api_key = APIKey(
            key_id=key_id,
            tenant_id=tenant_id,
            key_hash=key_hash,
            prefix=prefix,
            status=KeyStatus.ACTIVE,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        self._keys[key_id] = api_key
        self._tenant_keys.setdefault(tenant_id, []).append(key_id)

        return raw_key

    def validate_key(self, raw_key: str) -> KeyValidationResult:
        """Validate an API key.

        Args:
            raw_key: The raw API key to validate.

        Returns:
            KeyValidationResult indicating validity and tenant.
        """
        if not raw_key.startswith(self.KEY_PREFIX):
            return KeyValidationResult(valid=False, error="Invalid key format")

        key_hash = self._hash_key(raw_key)

        for api_key in self._keys.values():
            if hmac.compare_digest(api_key.key_hash, key_hash):
                if api_key.status == KeyStatus.REVOKED:
                    return KeyValidationResult(
                        valid=False,
                        key_id=api_key.key_id,
                        error="Key has been revoked",
                    )
                if api_key.is_expired:
                    return KeyValidationResult(
                        valid=False,
                        key_id=api_key.key_id,
                        tenant_id=api_key.tenant_id,
                        error="Key has expired",
                    )
                # Update last used timestamp
                api_key.last_used_at = time.time()
                return KeyValidationResult(
                    valid=True,
                    tenant_id=api_key.tenant_id,
                    key_id=api_key.key_id,
                )

        return KeyValidationResult(valid=False, error="Key not found")

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: The ID of the key to revoke.

        Returns:
            True if the key was revoked, False if not found.
        """
        if key_id not in self._keys:
            return False

        self._keys[key_id].status = KeyStatus.REVOKED
        return True

    def revoke_all_tenant_keys(self, tenant_id: str) -> int:
        """Revoke all keys for a tenant.

        Args:
            tenant_id: The tenant whose keys to revoke.

        Returns:
            Number of keys revoked.
        """
        key_ids = self._tenant_keys.get(tenant_id, [])
        count = 0
        for key_id in key_ids:
            if self._keys[key_id].status == KeyStatus.ACTIVE:
                self._keys[key_id].status = KeyStatus.REVOKED
                count += 1
        return count

    def rotate_key(self, tenant_id: str, old_key_id: Optional[str] = None) -> str:
        """Rotate a key for a tenant.

        Generates a new key and marks the old one for expiry after
        the grace period.

        Args:
            tenant_id: The tenant to rotate keys for.
            old_key_id: Specific key to rotate (or oldest active key).

        Returns:
            The new raw API key.
        """
        # Find the key to rotate
        if old_key_id is None:
            active_keys = self.get_active_keys(tenant_id)
            if active_keys:
                old_key_id = active_keys[0].key_id

        # Mark old key as rotating (grace period)
        if old_key_id and old_key_id in self._keys:
            old_key = self._keys[old_key_id]
            old_key.status = KeyStatus.ROTATING
            old_key.expires_at = time.time() + self._rotation_policy.grace_period_seconds

        # Generate new key
        new_raw_key = self.generate_key(tenant_id)

        # Find new key ID
        new_key_id = self._tenant_keys[tenant_id][-1]

        # Emit rotation event
        if self._on_rotation and old_key_id:
            event = RotationEvent(
                tenant_id=tenant_id,
                old_key_id=old_key_id,
                new_key_id=new_key_id,
                reason="scheduled_rotation",
            )
            self._on_rotation(event)

        return new_raw_key

    def get_active_keys(self, tenant_id: str) -> list[APIKey]:
        """Get all active keys for a tenant."""
        key_ids = self._tenant_keys.get(tenant_id, [])
        return [
            self._keys[kid]
            for kid in key_ids
            if self._keys[kid].is_active
        ]

    def get_tenant_keys(self, tenant_id: str) -> list[APIKey]:
        """Get all keys for a tenant regardless of status."""
        key_ids = self._tenant_keys.get(tenant_id, [])
        return [self._keys[kid] for kid in key_ids]

    def check_rotation_needed(self) -> list[APIKey]:
        """Check which keys need rotation based on policy.

        Returns:
            List of keys that should be rotated.
        """
        needs_rotation: list[APIKey] = []
        notify_threshold = self._rotation_policy.notify_before_seconds

        for api_key in self._keys.values():
            if api_key.status != KeyStatus.ACTIVE:
                continue
            if api_key.expires_at is None:
                continue

            time_until_expiry = api_key.expires_at - time.time()
            if time_until_expiry <= notify_threshold:
                needs_rotation.append(api_key)

                if self._on_expiry_warning:
                    self._on_expiry_warning(api_key, time_until_expiry)

        return needs_rotation

    def cleanup_expired_keys(self) -> int:
        """Remove expired and revoked keys older than grace period.

        Returns:
            Number of keys cleaned up.
        """
        to_remove: list[str] = []
        grace = self._rotation_policy.grace_period_seconds

        for key_id, api_key in self._keys.items():
            if api_key.status == KeyStatus.REVOKED:
                to_remove.append(key_id)
            elif api_key.is_expired:
                if api_key.expires_at and (time.time() - api_key.expires_at) > grace:
                    to_remove.append(key_id)

        for key_id in to_remove:
            tenant_id = self._keys[key_id].tenant_id
            del self._keys[key_id]
            if tenant_id in self._tenant_keys:
                self._tenant_keys[tenant_id] = [
                    k for k in self._tenant_keys[tenant_id] if k != key_id
                ]

        return len(to_remove)

    def get_key_stats(self) -> dict[str, Any]:
        """Get statistics about managed keys."""
        stats: dict[str, int] = {
            "total_keys": len(self._keys),
            "active": 0,
            "expired": 0,
            "revoked": 0,
            "rotating": 0,
            "tenants": len(self._tenant_keys),
        }
        for api_key in self._keys.values():
            if api_key.is_expired:
                stats["expired"] += 1
            elif api_key.status == KeyStatus.ACTIVE:
                stats["active"] += 1
            elif api_key.status == KeyStatus.REVOKED:
                stats["revoked"] += 1
            elif api_key.status == KeyStatus.ROTATING:
                stats["rotating"] += 1

        return stats

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _generate_key_id(self) -> str:
        """Generate a unique key ID."""
        return f"key_{secrets.token_hex(8)}"
