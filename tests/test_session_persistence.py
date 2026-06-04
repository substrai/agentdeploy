"""Tests for session persistence with configurable TTL and cleanup."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agentdeploy.sessions.persistence import (
    InMemoryBackend,
    SessionCleanupResult,
    SessionConfig,
    SessionPersistence,
    SessionRecord,
    SessionStatus,
)


class TestSessionRecord:
    """Tests for the SessionRecord dataclass."""

    def test_auto_generates_id(self):
        """Test that session ID is auto-generated."""
        record = SessionRecord(agent_id="agent-1")
        assert record.session_id != ""
        assert len(record.session_id) == 36  # UUID format

    def test_auto_sets_timestamps(self):
        """Test that timestamps are auto-set on creation."""
        record = SessionRecord(agent_id="agent-1")
        assert record.created_at > 0
        assert record.last_accessed_at > 0
        assert record.expires_at > record.created_at

    def test_is_expired(self):
        """Test expiry detection."""
        record = SessionRecord(
            agent_id="agent-1",
            expires_at=time.time() - 100,
        )
        assert record.is_expired is True

        record2 = SessionRecord(
            agent_id="agent-1",
            expires_at=time.time() + 1000,
        )
        assert record2.is_expired is False

    def test_touch_extends_ttl(self):
        """Test that touch updates timestamps and extends TTL."""
        record = SessionRecord(agent_id="agent-1", ttl_seconds=3600)
        old_expires = record.expires_at

        time.sleep(0.01)
        record.touch()

        assert record.last_accessed_at > record.created_at
        assert record.expires_at >= old_expires

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        record = SessionRecord(
            agent_id="agent-1",
            tenant_id="tenant-x",
            metadata={"key": "value"},
            context={"messages": []},
            ttl_seconds=7200,
        )
        data = record.to_dict()
        restored = SessionRecord.from_dict(data)

        assert restored.session_id == record.session_id
        assert restored.agent_id == "agent-1"
        assert restored.tenant_id == "tenant-x"
        assert restored.metadata == {"key": "value"}
        assert restored.ttl_seconds == 7200

    def test_remaining_ttl(self):
        """Test remaining TTL calculation."""
        record = SessionRecord(
            agent_id="agent-1",
            expires_at=time.time() + 500,
        )
        assert 499 <= record.remaining_ttl <= 501

        expired = SessionRecord(
            agent_id="agent-1",
            expires_at=time.time() - 100,
        )
        assert expired.remaining_ttl == 0.0


class TestSessionPersistence:
    """Tests for the SessionPersistence manager."""

    def _create_persistence(self, **kwargs) -> SessionPersistence:
        """Helper to create a persistence instance with in-memory backend."""
        config = SessionConfig(**kwargs)
        return SessionPersistence(config=config, backend=InMemoryBackend())

    def test_create_session(self):
        """Test creating a new session."""
        sp = self._create_persistence(default_ttl_seconds=3600)
        session = sp.create_session(
            agent_id="agent-1",
            tenant_id="tenant-x",
            metadata={"user": "alice"},
        )

        assert session.agent_id == "agent-1"
        assert session.tenant_id == "tenant-x"
        assert session.status == SessionStatus.ACTIVE
        assert session.metadata["user"] == "alice"

    def test_get_session(self):
        """Test retrieving a session by ID."""
        sp = self._create_persistence()
        created = sp.create_session(agent_id="agent-1")

        retrieved = sp.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session_returns_none(self):
        """Test that getting a nonexistent session returns None."""
        sp = self._create_persistence()
        assert sp.get_session("nonexistent-id") is None

    def test_get_expired_session_returns_expired_status(self):
        """Test that expired sessions are returned with expired status."""
        sp = self._create_persistence(
            default_ttl_seconds=1,
            enable_auto_extend=False,
        )
        session = sp.create_session(agent_id="agent-1")

        # Manually expire it
        session.expires_at = time.time() - 10
        sp.backend.put_item(sp.config.table_name, session.to_dict())

        retrieved = sp.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.status == SessionStatus.EXPIRED

    def test_update_session(self):
        """Test updating session metadata and context."""
        sp = self._create_persistence()
        session = sp.create_session(
            agent_id="agent-1",
            metadata={"version": "1"},
        )

        updated = sp.update_session(
            session.session_id,
            metadata={"version": "2", "extra": "data"},
            context={"messages": ["hello"]},
        )

        assert updated is not None
        assert updated.metadata["version"] == "2"
        assert updated.metadata["extra"] == "data"
        assert updated.context["messages"] == ["hello"]

    def test_terminate_session(self):
        """Test terminating a session."""
        sp = self._create_persistence()
        session = sp.create_session(agent_id="agent-1")

        result = sp.terminate_session(session.session_id)
        assert result is True

        # Verify it's terminated
        retrieved = sp.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.status in (SessionStatus.TERMINATED, SessionStatus.EXPIRED)

    def test_delete_session(self):
        """Test permanently deleting a session."""
        sp = self._create_persistence()
        session = sp.create_session(agent_id="agent-1")

        assert sp.delete_session(session.session_id) is True
        assert sp.get_session(session.session_id) is None
        assert sp.delete_session(session.session_id) is False

    def test_list_sessions(self):
        """Test listing sessions with filtering."""
        sp = self._create_persistence()
        sp.create_session(agent_id="agent-1", tenant_id="t1")
        sp.create_session(agent_id="agent-1", tenant_id="t2")
        sp.create_session(agent_id="agent-2", tenant_id="t1")

        all_sessions = sp.list_sessions()
        assert len(all_sessions) == 3

        agent1_sessions = sp.list_sessions(agent_id="agent-1")
        assert len(agent1_sessions) == 2

        t1_sessions = sp.list_sessions(tenant_id="t1")
        assert len(t1_sessions) == 2

    def test_cleanup_expired(self):
        """Test cleanup of expired sessions."""
        sp = self._create_persistence(default_ttl_seconds=1)

        # Create sessions that are already expired
        for i in range(5):
            session = sp.create_session(agent_id=f"agent-{i}")
            session.expires_at = time.time() - 100
            sp.backend.put_item(sp.config.table_name, session.to_dict())

        # Create active session
        sp.create_session(agent_id="active-agent")

        result = sp.cleanup_expired()
        assert result.total_scanned == 6
        assert result.total_deleted == 5
        assert result.success is True

    def test_cleanup_dry_run(self):
        """Test cleanup dry run doesn't delete."""
        sp = self._create_persistence()

        session = sp.create_session(agent_id="agent-1")
        session.expires_at = time.time() - 100
        sp.backend.put_item(sp.config.table_name, session.to_dict())

        result = sp.cleanup_expired(dry_run=True)
        assert result.total_deleted == 1  # Would delete 1

        # Verify it's still there
        assert sp.get_session(session.session_id) is not None

    def test_extend_ttl(self):
        """Test extending session TTL."""
        sp = self._create_persistence(max_ttl_seconds=86400)
        session = sp.create_session(agent_id="agent-1")
        original_expires = session.expires_at

        extended = sp.extend_ttl(session.session_id, 7200)
        assert extended is not None
        assert extended.expires_at > original_expires

    def test_ttl_capped_at_max(self):
        """Test that TTL is capped at max_ttl_seconds."""
        sp = self._create_persistence(
            default_ttl_seconds=3600,
            max_ttl_seconds=7200,
        )
        session = sp.create_session(
            agent_id="agent-1",
            ttl_seconds=99999,  # Exceeds max
        )
        # Should be capped
        assert session.ttl_seconds == 7200

    def test_auto_extend_on_access(self):
        """Test that auto-extend updates TTL on get."""
        sp = self._create_persistence(
            default_ttl_seconds=3600,
            enable_auto_extend=True,
        )
        session = sp.create_session(agent_id="agent-1")
        original_last_accessed = session.last_accessed_at

        time.sleep(0.01)
        retrieved = sp.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.last_accessed_at >= original_last_accessed


class TestInMemoryBackend:
    """Tests for the InMemoryBackend."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        backend = InMemoryBackend()
        backend.put_item("table", {"session_id": "s1", "data": "hello"})

        item = backend.get_item("table", {"session_id": "s1"})
        assert item is not None
        assert item["data"] == "hello"

    def test_delete(self):
        """Test delete operation."""
        backend = InMemoryBackend()
        backend.put_item("table", {"session_id": "s1", "data": "hello"})
        backend.delete_item("table", {"session_id": "s1"})

        assert backend.get_item("table", {"session_id": "s1"}) is None

    def test_batch_delete(self):
        """Test batch delete operation."""
        backend = InMemoryBackend()
        for i in range(5):
            backend.put_item("table", {"session_id": f"s{i}"})

        deleted = backend.batch_delete(
            "table",
            [{"session_id": f"s{i}"} for i in range(3)],
        )
        assert deleted == 3
