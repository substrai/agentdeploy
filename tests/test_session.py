"""Tests for session management."""

from agentdeploy.session.manager import Session, SessionManager, Message


class TestSession:
    def test_create_session(self):
        session = Session(session_id="test-1", agent_name="agent")
        assert session.is_empty
        assert session.turn_count == 0

    def test_add_messages(self):
        session = Session(session_id="test-1", agent_name="agent")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        assert len(session.messages) == 2
        assert session.turn_count == 2

    def test_get_history_full(self):
        session = Session(session_id="test-1", agent_name="agent")
        for i in range(10):
            session.add_message("user", f"Message {i}")
        history = session.get_history(strategy="full")
        assert len(history) == 10

    def test_get_history_sliding_window(self):
        session = Session(session_id="test-1", agent_name="agent")
        for i in range(20):
            session.add_message("user", f"Message {i}")
        history = session.get_history(strategy="sliding_window", window_size=5)
        assert len(history) == 5
        assert "Message 19" in history[-1].content

    def test_session_to_dict(self):
        session = Session(session_id="test-1", agent_name="agent", tenant_id="tenant-a")
        session.add_message("user", "Hi")
        d = session.to_dict()
        assert d["session_id"] == "test-1"
        assert d["message_count"] == 1


class TestSessionManager:
    def test_get_or_create(self):
        mgr = SessionManager()
        session = mgr.get_or_create("sess-1", "agent")
        assert session.session_id == "sess-1"

    def test_get_existing(self):
        mgr = SessionManager()
        s1 = mgr.get_or_create("sess-1", "agent")
        s1.add_message("user", "Hello")
        s2 = mgr.get_or_create("sess-1", "agent")
        assert len(s2.messages) == 1  # same session

    def test_auto_generate_id(self):
        mgr = SessionManager()
        session = mgr.get_or_create(None, "agent")
        assert session.session_id.startswith("sess-")

    def test_delete_session(self):
        mgr = SessionManager()
        mgr.get_or_create("sess-1", "agent")
        assert mgr.delete("sess-1")
        assert mgr.get("sess-1") is None

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.get_or_create("s1", "agent-a")
        mgr.get_or_create("s2", "agent-b")
        mgr.get_or_create("s3", "agent-a")
        sessions = mgr.list_sessions(agent_name="agent-a")
        assert len(sessions) == 2

    def test_active_count(self):
        mgr = SessionManager(ttl_hours=24)
        mgr.get_or_create("s1", "agent")
        mgr.get_or_create("s2", "agent")
        assert mgr.get_active_count() == 2
