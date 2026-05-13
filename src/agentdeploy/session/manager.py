"""Session management - conversation persistence."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


@dataclass
class Session:
    """Agent conversation session with memory."""

    session_id: str
    agent_name: str
    tenant_id: str = "default"
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    turn_count: int = 0

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content, **kwargs))
        self.last_active = time.time()
        if role in ("user", "assistant"):
            self.turn_count += 1

    def get_history(self, strategy: str = "full", window_size: int = 20) -> List[Message]:
        """Get conversation history based on strategy.

        Args:
            strategy: "full", "sliding_window", "last_n"
            window_size: Number of messages for sliding window

        Returns:
            List of messages
        """
        if strategy == "full":
            return list(self.messages)
        elif strategy == "sliding_window":
            return list(self.messages[-window_size:])
        elif strategy == "last_n":
            return list(self.messages[-window_size:])
        return list(self.messages)

    @property
    def duration_seconds(self) -> float:
        return self.last_active - self.created_at

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "tenant_id": self.tenant_id,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": len(self.messages),
        }


class SessionManager:
    """Manages agent sessions (in-memory for local, DynamoDB for prod).

    Usage:
        manager = SessionManager()
        session = manager.get_or_create("session-123", "my-agent")
        session.add_message("user", "Hello!")
        manager.save(session)
    """

    def __init__(self, ttl_hours: int = 24, max_sessions: int = 10000):
        self._sessions: Dict[str, Session] = {}
        self.ttl_seconds = ttl_hours * 3600
        self.max_sessions = max_sessions

    def get_or_create(
        self, session_id: Optional[str], agent_name: str, tenant_id: str = "default"
    ) -> Session:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_active = time.time()
            return session

        # Create new session
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        session = Session(session_id=sid, agent_name=agent_name, tenant_id=tenant_id)
        self._sessions[sid] = session
        self._cleanup()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        """Save/update a session."""
        self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(
        self, agent_name: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> List[Session]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        if agent_name:
            sessions = [s for s in sessions if s.agent_name == agent_name]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        return sorted(sessions, key=lambda s: s.last_active, reverse=True)

    def get_active_count(self) -> int:
        """Count active (non-expired) sessions."""
        cutoff = time.time() - self.ttl_seconds
        return sum(1 for s in self._sessions.values() if s.last_active > cutoff)

    def _cleanup(self) -> None:
        """Remove expired sessions."""
        cutoff = time.time() - self.ttl_seconds
        expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]
