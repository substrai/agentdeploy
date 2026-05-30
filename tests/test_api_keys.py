"""Tests for per-tenant API key management with rotation."""

import time
from unittest.mock import MagicMock

import pytest

from agentdeploy.auth.api_keys import (
    APIKey,
    APIKeyManager,
    KeyStatus,
    KeyValidationResult,
    RotationEvent,
    RotationPolicy,
)


@pytest.fixture
def manager() -> APIKeyManager:
    """Create an API key manager with default policy."""
    policy = RotationPolicy(rotation_interval_days=90)
    return APIKeyManager(rotation_policy=policy)


@pytest.fixture
def manager_short_expiry() -> APIKeyManager:
    """Create a manager with very short expiry for testing."""
    policy = RotationPolicy(
        rotation_interval_days=0.001,  # ~86 seconds
        grace_period_days=0.0001,
    )
    return APIKeyManager(rotation_policy=policy)


class TestKeyGeneration:
    """Test API key generation."""

    def test_generate_key_returns_string(self, manager: APIKeyManager) -> None:
        """Should return a raw key string."""
        raw_key = manager.generate_key("tenant-1")
        assert isinstance(raw_key, str)
        assert raw_key.startswith("ad_")
        assert len(raw_key) > 20

    def test_generate_key_unique(self, manager: APIKeyManager) -> None:
        """Should generate unique keys each time."""
        key1 = manager.generate_key("tenant-1")
        key2 = manager.generate_key("tenant-1")
        assert key1 != key2

    def test_generate_key_tracks_tenant(self, manager: APIKeyManager) -> None:
        """Should track keys per tenant."""
        manager.generate_key("tenant-1")
        manager.generate_key("tenant-1")
        manager.generate_key("tenant-2")

        assert len(manager.get_tenant_keys("tenant-1")) == 2
        assert len(manager.get_tenant_keys("tenant-2")) == 1

    def test_generate_key_with_custom_expiry(self, manager: APIKeyManager) -> None:
        """Should respect custom expiry."""
        manager.generate_key("tenant-1", expires_in_days=30)
        keys = manager.get_active_keys("tenant-1")
        assert len(keys) == 1
        assert keys[0].days_until_expiry is not None
        assert 29 < keys[0].days_until_expiry < 31


class TestKeyValidation:
    """Test API key validation."""

    def test_validate_valid_key(self, manager: APIKeyManager) -> None:
        """Should validate a correct key."""
        raw_key = manager.generate_key("tenant-1")
        result = manager.validate_key(raw_key)
        assert result.valid is True
        assert result.tenant_id == "tenant-1"

    def test_validate_invalid_key(self, manager: APIKeyManager) -> None:
        """Should reject an invalid key."""
        result = manager.validate_key("ad_invalid_key_here")
        assert result.valid is False
        assert result.error == "Key not found"

    def test_validate_wrong_prefix(self, manager: APIKeyManager) -> None:
        """Should reject keys with wrong prefix."""
        result = manager.validate_key("wrong_prefix_key")
        assert result.valid is False
        assert "Invalid key format" in result.error

    def test_validate_revoked_key(self, manager: APIKeyManager) -> None:
        """Should reject revoked keys."""
        raw_key = manager.generate_key("tenant-1")
        keys = manager.get_active_keys("tenant-1")
        manager.revoke_key(keys[0].key_id)

        result = manager.validate_key(raw_key)
        assert result.valid is False
        assert "revoked" in result.error

    def test_validate_expired_key(self, manager: APIKeyManager) -> None:
        """Should reject expired keys."""
        raw_key = manager.generate_key("tenant-1", expires_in_days=0.00001)
        # Force expiry
        keys = manager.get_tenant_keys("tenant-1")
        keys[0].expires_at = time.time() - 100

        result = manager.validate_key(raw_key)
        assert result.valid is False
        assert "expired" in result.error


class TestKeyRevocation:
    """Test key revocation."""

    def test_revoke_key(self, manager: APIKeyManager) -> None:
        """Should revoke a specific key."""
        manager.generate_key("tenant-1")
        keys = manager.get_active_keys("tenant-1")
        assert len(keys) == 1

        manager.revoke_key(keys[0].key_id)
        assert len(manager.get_active_keys("tenant-1")) == 0

    def test_revoke_all_tenant_keys(self, manager: APIKeyManager) -> None:
        """Should revoke all keys for a tenant."""
        manager.generate_key("tenant-1")
        manager.generate_key("tenant-1")
        manager.generate_key("tenant-2")

        count = manager.revoke_all_tenant_keys("tenant-1")
        assert count == 2
        assert len(manager.get_active_keys("tenant-1")) == 0
        assert len(manager.get_active_keys("tenant-2")) == 1

    def test_revoke_nonexistent_key(self, manager: APIKeyManager) -> None:
        """Should return False for nonexistent key."""
        assert manager.revoke_key("nonexistent") is False


class TestKeyRotation:
    """Test key rotation functionality."""

    def test_rotate_key_generates_new(self, manager: APIKeyManager) -> None:
        """Should generate a new key during rotation."""
        old_key = manager.generate_key("tenant-1")
        new_key = manager.rotate_key("tenant-1")

        assert new_key != old_key
        assert new_key.startswith("ad_")

        # New key should be valid
        result = manager.validate_key(new_key)
        assert result.valid is True

    def test_rotate_key_marks_old_as_rotating(self, manager: APIKeyManager) -> None:
        """Should mark old key as rotating with grace period."""
        manager.generate_key("tenant-1")
        old_keys = manager.get_active_keys("tenant-1")
        old_key_id = old_keys[0].key_id

        manager.rotate_key("tenant-1")

        old_key = manager._keys[old_key_id]
        assert old_key.status == KeyStatus.ROTATING

    def test_rotation_callback_fires(self) -> None:
        """Should fire rotation callback."""
        callback = MagicMock()
        manager = APIKeyManager(on_rotation=callback)
        manager.generate_key("tenant-1")

        manager.rotate_key("tenant-1")
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, RotationEvent)
        assert event.tenant_id == "tenant-1"


class TestKeyStats:
    """Test key statistics."""

    def test_get_key_stats(self, manager: APIKeyManager) -> None:
        """Should return accurate statistics."""
        manager.generate_key("tenant-1")
        manager.generate_key("tenant-2")
        manager.generate_key("tenant-2")

        keys = manager.get_active_keys("tenant-2")
        manager.revoke_key(keys[0].key_id)

        stats = manager.get_key_stats()
        assert stats["total_keys"] == 3
        assert stats["active"] == 2
        assert stats["revoked"] == 1
        assert stats["tenants"] == 2

    def test_check_rotation_needed(self, manager: APIKeyManager) -> None:
        """Should identify keys needing rotation."""
        manager.generate_key("tenant-1")
        keys = manager.get_active_keys("tenant-1")

        # Set expiry to within notification window
        keys[0].expires_at = time.time() + 100  # Less than 14 days

        needs_rotation = manager.check_rotation_needed()
        assert len(needs_rotation) == 1
        assert needs_rotation[0].tenant_id == "tenant-1"
