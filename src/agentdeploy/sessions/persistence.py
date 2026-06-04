"""Session persistence with configurable TTL and cleanup.

Provides DynamoDB-backed session management for deployed agents with
automatic expiry, manual cleanup CLI, and session listing capabilities.

Features:
- DynamoDB session storage with auto-expiry via TTL
- Configurable TTL per session or globally
- Batch cleanup of expired sessions
- Session listing with filtering and pagination
- Session metadata storage and retrieval
- CLI integration for manual session management
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple


class SessionStatus(Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


@dataclass
class SessionRecord:
    """A persisted session record.

    Attributes:
        session_id: Unique session identifier.
        agent_id: The agent this session belongs to.
        tenant_id: Optional tenant identifier for multi-tenant isolation.
        created_at: Unix timestamp of session creation.
        last_accessed_at: Unix timestamp of last access.
        expires_at: Unix timestamp when session expires (TTL).
        status: Current session status.
        metadata: Arbitrary session metadata.
        context: Session context data (conversation state, etc.).
        ttl_seconds: TTL duration in seconds.
    """

    session_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    expires_at: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int = 3600

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed_at:
            self.last_accessed_at = self.created_at
        if not self.expires_at:
            self.expires_at = self.created_at + self.ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return time.time() > self.expires_at

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)

    def touch(self) -> None:
        """Update last access time and extend TTL."""
        now = time.time()
        self.last_accessed_at = now
        self.expires_at = now + self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to a dictionary."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "created_at": int(self.created_at),
            "last_accessed_at": int(self.last_accessed_at),
            "expires_at": int(self.expires_at),
            "status": self.status.value,
            "metadata": self.metadata,
            "context": self.context,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionRecord:
        """Deserialize a session from a dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id", ""),
            tenant_id=data.get("tenant_id", ""),
            created_at=float(data.get("created_at", 0)),
            last_accessed_at=float(data.get("last_accessed_at", 0)),
            expires_at=float(data.get("expires_at", 0)),
            status=SessionStatus(data.get("status", "active")),
            metadata=data.get("metadata", {}),
            context=data.get("context", {}),
            ttl_seconds=int(data.get("ttl_seconds", 3600)),
        )


@dataclass
class SessionConfig:
    """Configuration for session persistence.

    Attributes:
        table_name: DynamoDB table name.
        default_ttl_seconds: Default TTL for new sessions.
        max_ttl_seconds: Maximum allowed TTL.
        cleanup_batch_size: Batch size for cleanup operations.
        enable_auto_extend: Whether to extend TTL on access.
        region: AWS region for DynamoDB.
        endpoint_url: Optional DynamoDB endpoint (for local development).
    """

    table_name: str = "agent_sessions"
    default_ttl_seconds: int = 3600
    max_ttl_seconds: int = 86400
    cleanup_batch_size: int = 25
    enable_auto_extend: bool = True
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None


@dataclass
class SessionCleanupResult:
    """Result of a session cleanup operation.

    Attributes:
        total_scanned: Number of sessions scanned.
        total_deleted: Number of expired sessions deleted.
        errors: List of errors encountered during cleanup.
        duration_seconds: Time taken for the cleanup operation.
    """

    total_scanned: int = 0
    total_deleted: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if cleanup completed without errors."""
        return len(self.errors) == 0


class DynamoDBBackend(Protocol):
    """Protocol for DynamoDB operations (for testing with mocks)."""

    def put_item(self, table_name: str, item: Dict[str, Any]) -> None: ...
    def get_item(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    def delete_item(self, table_name: str, key: Dict[str, Any]) -> None: ...
    def query(self, table_name: str, key_condition: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]: ...
    def scan(self, table_name: str, filter_expression: Optional[Dict] = None, **kwargs) -> List[Dict[str, Any]]: ...
    def batch_delete(self, table_name: str, keys: List[Dict[str, Any]]) -> int: ...


class InMemoryBackend:
    """In-memory backend for testing (implements DynamoDBBackend protocol).

    Stores sessions in a dictionary, simulating DynamoDB behavior.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def put_item(self, table_name: str, item: Dict[str, Any]) -> None:
        key = item.get("session_id", "")
        self._store[key] = item

    def get_item(
        self, table_name: str, key: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        session_id = key.get("session_id", "")
        return self._store.get(session_id)

    def delete_item(self, table_name: str, key: Dict[str, Any]) -> None:
        session_id = key.get("session_id", "")
        self._store.pop(session_id, None)

    def query(
        self, table_name: str, key_condition: Dict[str, Any], **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        for item in self._store.values():
            match = all(
                item.get(k) == v for k, v in key_condition.items()
            )
            if match:
                results.append(item)
        return results

    def scan(
        self, table_name: str, filter_expression: Optional[Dict] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        if filter_expression is None:
            return list(self._store.values())

        results = []
        for item in self._store.values():
            match = True
            for k, v in filter_expression.items():
                if k == "status" and item.get("status") != v:
                    match = False
                elif k == "agent_id" and item.get("agent_id") != v:
                    match = False
                elif k == "tenant_id" and item.get("tenant_id") != v:
                    match = False
            if match:
                results.append(item)
        return results

    def batch_delete(self, table_name: str, keys: List[Dict[str, Any]]) -> int:
        deleted = 0
        for key in keys:
            session_id = key.get("session_id", "")
            if session_id in self._store:
                del self._store[session_id]
                deleted += 1
        return deleted


class SessionPersistence:
    """Main session persistence manager.

    Provides CRUD operations for sessions with TTL management,
    batch cleanup, and listing capabilities.

    Args:
        config: Session configuration.
        backend: Storage backend (DynamoDB or in-memory for testing).
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        backend: Optional[Any] = None,
    ):
        self.config = config or SessionConfig()
        self.backend = backend or InMemoryBackend()

    def create_session(
        self,
        agent_id: str,
        tenant_id: str = "",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SessionRecord:
        """Create a new session.

        Args:
            agent_id: The agent identifier.
            tenant_id: Optional tenant identifier.
            ttl_seconds: Optional TTL override (uses config default if not set).
            metadata: Optional session metadata.
            context: Optional initial session context.

        Returns:
            The created SessionRecord.
        """
        ttl = ttl_seconds or self.config.default_ttl_seconds
        ttl = min(ttl, self.config.max_ttl_seconds)

        session = SessionRecord(
            agent_id=agent_id,
            tenant_id=tenant_id,
            ttl_seconds=ttl,
            metadata=metadata or {},
            context=context or {},
        )

        self.backend.put_item(self.config.table_name, session.to_dict())
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            SessionRecord if found and not expired, None otherwise.
        """
        item = self.backend.get_item(
            self.config.table_name,
            {"session_id": session_id},
        )
        if item is None:
            return None

        session = SessionRecord.from_dict(item)

        # Check expiry
        if session.is_expired:
            session.status = SessionStatus.EXPIRED
            return session

        # Auto-extend TTL on access
        if self.config.enable_auto_extend:
            session.touch()
            self.backend.put_item(self.config.table_name, session.to_dict())

        return session

    def update_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionRecord]:
        """Update session metadata or context.

        Args:
            session_id: The session identifier.
            metadata: New metadata (merged with existing).
            context: New context (merged with existing).

        Returns:
            Updated SessionRecord, or None if not found.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        if metadata:
            session.metadata.update(metadata)
        if context:
            session.context.update(context)

        session.touch()
        self.backend.put_item(self.config.table_name, session.to_dict())
        return session

    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session (mark as terminated).

        Args:
            session_id: The session identifier.

        Returns:
            True if terminated, False if not found.
        """
        item = self.backend.get_item(
            self.config.table_name,
            {"session_id": session_id},
        )
        if item is None:
            return False

        session = SessionRecord.from_dict(item)
        session.status = SessionStatus.TERMINATED
        session.expires_at = time.time()  # Expire immediately
        self.backend.put_item(self.config.table_name, session.to_dict())
        return True

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session.

        Args:
            session_id: The session identifier.

        Returns:
            True if deleted, False if not found.
        """
        item = self.backend.get_item(
            self.config.table_name,
            {"session_id": session_id},
        )
        if item is None:
            return False

        self.backend.delete_item(
            self.config.table_name,
            {"session_id": session_id},
        )
        return True

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        include_expired: bool = False,
    ) -> List[SessionRecord]:
        """List sessions with optional filtering.

        Args:
            agent_id: Filter by agent ID.
            tenant_id: Filter by tenant ID.
            status: Filter by session status.
            include_expired: Whether to include expired sessions.

        Returns:
            List of matching SessionRecords.
        """
        filter_expr: Dict[str, Any] = {}
        if agent_id:
            filter_expr["agent_id"] = agent_id
        if tenant_id:
            filter_expr["tenant_id"] = tenant_id
        if status:
            filter_expr["status"] = status.value

        items = self.backend.scan(
            self.config.table_name,
            filter_expression=filter_expr if filter_expr else None,
        )

        sessions = [SessionRecord.from_dict(item) for item in items]

        if not include_expired:
            now = time.time()
            sessions = [
                s for s in sessions
                if s.expires_at > now or s.status == SessionStatus.ACTIVE
            ]

        return sessions

    def cleanup_expired(
        self, dry_run: bool = False
    ) -> SessionCleanupResult:
        """Clean up expired sessions in batches.

        Args:
            dry_run: If True, only scan without deleting.

        Returns:
            SessionCleanupResult with cleanup statistics.
        """
        start_time = time.time()
        result = SessionCleanupResult()

        items = self.backend.scan(self.config.table_name)
        result.total_scanned = len(items)

        now = time.time()
        expired_keys = []

        for item in items:
            expires_at = float(item.get("expires_at", 0))
            status = item.get("status", "active")
            if expires_at < now or status in ("expired", "terminated"):
                expired_keys.append({"session_id": item["session_id"]})

        if not dry_run and expired_keys:
            # Batch delete in configured batch sizes
            for i in range(0, len(expired_keys), self.config.cleanup_batch_size):
                batch = expired_keys[i:i + self.config.cleanup_batch_size]
                try:
                    deleted = self.backend.batch_delete(
                        self.config.table_name, batch
                    )
                    result.total_deleted += deleted
                except Exception as e:
                    result.errors.append(str(e))
        elif dry_run:
            result.total_deleted = len(expired_keys)

        result.duration_seconds = time.time() - start_time
        return result

    def extend_ttl(
        self, session_id: str, additional_seconds: int
    ) -> Optional[SessionRecord]:
        """Extend a session's TTL.

        Args:
            session_id: The session identifier.
            additional_seconds: Seconds to add to current TTL.

        Returns:
            Updated SessionRecord, or None if not found.
        """
        item = self.backend.get_item(
            self.config.table_name,
            {"session_id": session_id},
        )
        if item is None:
            return None

        session = SessionRecord.from_dict(item)
        new_expires = max(session.expires_at, time.time()) + additional_seconds
        max_expires = time.time() + self.config.max_ttl_seconds

        session.expires_at = min(new_expires, max_expires)
        session.last_accessed_at = time.time()
        self.backend.put_item(self.config.table_name, session.to_dict())
        return session
