"""Tests for session export/import for debugging and replay."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from agentdeploy.session.export_import import (
    ExportFormat,
    RedactionLevel,
    ReplayResult,
    SessionEvent,
    SessionExport,
    SessionExporter,
    SessionImporter,
)


def _make_events() -> list:
    """Create sample session events."""
    base_time = time.time()
    return [
        SessionEvent(
            timestamp=base_time,
            event_type="user_message",
            data={"content": "Hello, help me search for Python tutorials"},
            metadata={"user_id": "user-123"},
        ),
        SessionEvent(
            timestamp=base_time + 0.5,
            event_type="tool_call",
            data={"tool": "search_kb", "args": {"query": "Python tutorials"}},
            metadata={"cost_usd": 0.0},
            duration_ms=50.0,
        ),
        SessionEvent(
            timestamp=base_time + 0.6,
            event_type="tool_result",
            data={"results": ["Tutorial 1", "Tutorial 2"]},
            metadata={},
        ),
        SessionEvent(
            timestamp=base_time + 1.2,
            event_type="agent_response",
            data={"content": "Here are some Python tutorials I found..."},
            metadata={"cost_usd": 0.001, "tokens": 150},
            duration_ms=600.0,
        ),
    ]


class TestSessionExporter:
    """Test session export functionality."""

    def test_export_basic_session(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.NONE)
        events = _make_events()
        export = exporter.export_session(
            session_id="sess-001",
            events=events,
            agent_name="support-agent",
        )
        assert export.session_id == "sess-001"
        assert export.agent_name == "support-agent"
        assert export.total_events == 4
        assert export.version == "1.0.0"

    def test_export_calculates_duration(self):
        exporter = SessionExporter()
        events = _make_events()
        export = exporter.export_session(session_id="s1", events=events)
        assert export.total_duration_ms > 0

    def test_export_calculates_cost(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.NONE)
        events = _make_events()
        export = exporter.export_session(session_id="s1", events=events)
        assert export.total_cost_usd == 0.001

    def test_export_basic_redaction(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.BASIC)
        events = [
            SessionEvent(
                timestamp=time.time(),
                event_type="tool_call",
                data={"api_key": "sk-secret-123", "query": "hello"},
                metadata={"token": "bearer-abc"},
            ),
        ]
        export = exporter.export_session(session_id="s1", events=events)
        assert export.events[0].data["api_key"] == "[REDACTED]"
        assert export.events[0].data["query"] == "hello"
        assert export.events[0].metadata["token"] == "[REDACTED]"

    def test_export_strict_redaction(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.STRICT)
        events = [
            SessionEvent(
                timestamp=time.time(),
                event_type="user_message",
                data={"content": "hi", "email": "user@example.com"},
                metadata={"user_id": "u-123"},
            ),
        ]
        export = exporter.export_session(
            session_id="s1", events=events, tenant_id="tenant-x"
        )
        assert export.events[0].data["email"] == "[REDACTED]"
        assert export.events[0].metadata["user_id"] == "[REDACTED]"
        assert export.tenant_id == "[REDACTED]"

    def test_export_no_redaction(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.NONE)
        events = [
            SessionEvent(
                timestamp=time.time(),
                event_type="tool_call",
                data={"api_key": "sk-123", "query": "test"},
                metadata={},
            ),
        ]
        export = exporter.export_session(session_id="s1", events=events)
        assert export.events[0].data["api_key"] == "sk-123"


class TestSessionExportSerialization:
    """Test export serialization to/from dict."""

    def test_to_dict_and_back(self):
        exporter = SessionExporter(redaction_level=RedactionLevel.NONE)
        events = _make_events()
        export = exporter.export_session(session_id="s1", events=events, agent_name="bot")

        data = export.to_dict()
        restored = SessionExport.from_dict(data)

        assert restored.session_id == "s1"
        assert restored.agent_name == "bot"
        assert len(restored.events) == 4
        assert restored.events[0].event_type == "user_message"

    def test_save_to_json_file(self):
        exporter = SessionExporter()
        events = _make_events()
        export = exporter.export_session(session_id="s1", events=events)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        exporter.save_to_file(export, path)
        content = Path(path).read_text()
        data = json.loads(content)
        assert data["session_id"] == "s1"
        assert len(data["events"]) == 4

    def test_save_to_jsonl_file(self):
        exporter = SessionExporter(format=ExportFormat.JSONL)
        events = _make_events()
        export = exporter.export_session(session_id="s1", events=events)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        exporter.save_to_file(export, path)
        lines = Path(path).read_text().strip().split("\n")
        # 1 header + 4 events
        assert len(lines) == 5


class TestSessionImporter:
    """Test session import functionality."""

    def test_load_from_json_file(self):
        # Create and export
        exporter = SessionExporter(redaction_level=RedactionLevel.NONE)
        events = _make_events()
        export = exporter.export_session(session_id="s-import", events=events)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        exporter.save_to_file(export, path)

        # Import
        importer = SessionImporter()
        loaded = importer.load_from_file(path)
        assert loaded.session_id == "s-import"
        assert len(loaded.events) == 4

    def test_load_from_dict(self):
        importer = SessionImporter()
        data = {
            "session_id": "s-dict",
            "agent_name": "test-agent",
            "created_at": time.time(),
            "exported_at": time.time(),
            "events": [
                {"timestamp": time.time(), "event_type": "user_message", "data": {"content": "hi"}},
            ],
        }
        session = importer.load_from_dict(data)
        assert session.session_id == "s-dict"
        assert len(session.events) == 1

    def test_load_file_not_found(self):
        importer = SessionImporter()
        with pytest.raises(FileNotFoundError):
            importer.load_from_file("/nonexistent/path.json")

    def test_load_invalid_format(self):
        importer = SessionImporter()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"no_session_id": True}, f)
            path = f.name

        with pytest.raises(ValueError, match="missing 'session_id'"):
            importer.load_from_file(path)


class TestSessionReplay:
    """Test session replay for debugging."""

    def test_replay_without_handler(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s-replay",
            agent_name="bot",
            tenant_id=None,
            created_at=time.time(),
            exported_at=time.time(),
            events=events,
            metadata={},
        )
        result = importer.replay(session)
        assert result.success is True
        assert result.events_replayed == 4
        assert result.events_matched == 4
        assert result.events_diverged == 0

    def test_replay_with_matching_handler(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s-replay",
            agent_name="bot",
            tenant_id=None,
            created_at=time.time(),
            exported_at=time.time(),
            events=events,
            metadata={},
        )

        def handler(event: SessionEvent):
            # Return matching data
            return event.data

        result = importer.replay(session, handler=handler)
        assert result.success is True

    def test_replay_detects_divergence(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s-replay",
            agent_name="bot",
            tenant_id=None,
            created_at=time.time(),
            exported_at=time.time(),
            events=events,
            metadata={},
        )

        def diverging_handler(event: SessionEvent):
            return {"content": "DIFFERENT RESPONSE"}

        result = importer.replay(session, handler=diverging_handler)
        assert result.success is False
        assert result.events_diverged > 0
        assert len(result.divergences) > 0

    def test_replay_stop_on_divergence(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s-replay",
            agent_name="bot",
            tenant_id=None,
            created_at=time.time(),
            exported_at=time.time(),
            events=events,
            metadata={},
        )

        call_count = 0

        def diverging_handler(event: SessionEvent):
            nonlocal call_count
            call_count += 1
            return {"wrong": "data"}

        result = importer.replay(session, handler=diverging_handler, stop_on_divergence=True)
        assert result.events_diverged == 1
        assert call_count == 1  # Stopped after first divergence


class TestSessionUtilities:
    """Test utility methods."""

    def test_get_user_messages(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s1", agent_name="bot", tenant_id=None,
            created_at=time.time(), exported_at=time.time(),
            events=events, metadata={},
        )
        messages = importer.get_user_messages(session)
        assert len(messages) == 1
        assert "Python tutorials" in messages[0]

    def test_get_tool_calls(self):
        importer = SessionImporter()
        events = _make_events()
        session = SessionExport(
            session_id="s1", agent_name="bot", tenant_id=None,
            created_at=time.time(), exported_at=time.time(),
            events=events, metadata={},
        )
        tool_calls = importer.get_tool_calls(session)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search_kb"
