"""Session export/import for debugging and replay.

Enables exporting full agent sessions as JSON for debugging, sharing,
and replay. Supports filtering sensitive data, timestamped snapshots,
and deterministic replay of conversation flows.

Usage:
    from agentdeploy.session.export_import import SessionExporter, SessionImporter

    # Export
    exporter = SessionExporter(redact_secrets=True)
    export_data = exporter.export_session(session_id="sess-123")
    exporter.save_to_file(export_data, "debug_session.json")

    # Import and replay
    importer = SessionImporter()
    session = importer.load_from_file("debug_session.json")
    replay_results = importer.replay(session)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    JSONL = "jsonl"  # One event per line


class RedactionLevel(str, Enum):
    """Level of data redaction for exports."""

    NONE = "none"  # No redaction
    BASIC = "basic"  # Redact API keys, tokens
    STRICT = "strict"  # Redact all PII and identifiers


# Patterns to redact in BASIC mode
BASIC_REDACTION_KEYS = {
    "api_key", "token", "secret", "password", "authorization",
    "access_key", "secret_key", "session_token",
}

# Additional patterns for STRICT mode
STRICT_REDACTION_KEYS = {
    "email", "phone", "address", "ssn", "credit_card",
    "user_id", "account_id", "ip_address",
}


@dataclass
class SessionEvent:
    """A single event in a session timeline."""

    timestamp: float
    event_type: str  # "user_message", "agent_response", "tool_call", "tool_result", "error"
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


@dataclass
class SessionExport:
    """Complete exported session data."""

    session_id: str
    agent_name: str
    tenant_id: Optional[str]
    created_at: float
    exported_at: float
    events: List[SessionEvent]
    metadata: Dict[str, Any]
    version: str = "1.0.0"
    total_events: int = 0
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    redaction_level: str = "none"

    def __post_init__(self):
        self.total_events = len(self.events)
        if self.events:
            first_ts = self.events[0].timestamp
            last_ts = self.events[-1].timestamp
            self.total_duration_ms = (last_ts - first_ts) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "exported_at": self.exported_at,
            "total_events": self.total_events,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "redaction_level": self.redaction_level,
            "metadata": self.metadata,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "data": e.data,
                    "metadata": e.metadata,
                    "duration_ms": e.duration_ms,
                }
                for e in self.events
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionExport":
        """Reconstruct from dictionary."""
        events = [
            SessionEvent(
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                data=e["data"],
                metadata=e.get("metadata", {}),
                duration_ms=e.get("duration_ms"),
            )
            for e in data.get("events", [])
        ]
        return cls(
            session_id=data["session_id"],
            agent_name=data["agent_name"],
            tenant_id=data.get("tenant_id"),
            created_at=data["created_at"],
            exported_at=data["exported_at"],
            events=events,
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0"),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            redaction_level=data.get("redaction_level", "none"),
        )


@dataclass
class ReplayResult:
    """Result of replaying a session."""

    session_id: str
    events_replayed: int
    events_matched: int
    events_diverged: int
    divergences: List[Dict[str, Any]]
    duration_ms: float
    success: bool


class SessionExporter:
    """Exports agent sessions for debugging and sharing.

    Supports redaction of sensitive data and multiple output formats.
    """

    def __init__(
        self,
        redaction_level: RedactionLevel = RedactionLevel.BASIC,
        format: ExportFormat = ExportFormat.JSON,
    ):
        self._redaction_level = redaction_level
        self._format = format

    def export_session(
        self,
        session_id: str,
        events: List[SessionEvent],
        agent_name: str = "unknown",
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionExport:
        """Export a session from its events.

        Args:
            session_id: The session identifier.
            events: List of session events in chronological order.
            agent_name: Name of the agent.
            tenant_id: Optional tenant identifier.
            metadata: Optional session-level metadata.

        Returns:
            SessionExport ready for serialization.
        """
        # Apply redaction
        redacted_events = [self._redact_event(e) for e in events]

        # Calculate total cost
        total_cost = sum(
            e.metadata.get("cost_usd", 0.0) for e in events
        )

        export = SessionExport(
            session_id=session_id,
            agent_name=agent_name,
            tenant_id=tenant_id if self._redaction_level != RedactionLevel.STRICT else "[REDACTED]",
            created_at=events[0].timestamp if events else time.time(),
            exported_at=time.time(),
            events=redacted_events,
            metadata=metadata or {},
            total_cost_usd=total_cost,
            redaction_level=self._redaction_level.value,
        )

        return export

    def save_to_file(self, export: SessionExport, path: str) -> str:
        """Save export to a file.

        Args:
            export: The session export data.
            path: File path to save to.

        Returns:
            Absolute path of the saved file.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if self._format == ExportFormat.JSON:
            with open(file_path, "w") as f:
                json.dump(export.to_dict(), f, indent=2, default=str)
        elif self._format == ExportFormat.JSONL:
            with open(file_path, "w") as f:
                # Write header line
                header = {k: v for k, v in export.to_dict().items() if k != "events"}
                f.write(json.dumps(header, default=str) + "\n")
                # Write one event per line
                for event in export.events:
                    f.write(json.dumps({
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "data": event.data,
                        "metadata": event.metadata,
                    }, default=str) + "\n")

        return str(file_path.resolve())

    def _redact_event(self, event: SessionEvent) -> SessionEvent:
        """Apply redaction to a single event."""
        if self._redaction_level == RedactionLevel.NONE:
            return event

        redacted_data = self._redact_dict(event.data)
        redacted_metadata = self._redact_dict(event.metadata)

        return SessionEvent(
            timestamp=event.timestamp,
            event_type=event.event_type,
            data=redacted_data,
            metadata=redacted_metadata,
            duration_ms=event.duration_ms,
        )

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive fields in a dictionary."""
        redacted: Dict[str, Any] = {}
        keys_to_redact = BASIC_REDACTION_KEYS.copy()

        if self._redaction_level == RedactionLevel.STRICT:
            keys_to_redact |= STRICT_REDACTION_KEYS

        for key, value in data.items():
            if key.lower() in keys_to_redact:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self._redact_dict(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                redacted[key] = value

        return redacted


class SessionImporter:
    """Imports and replays exported sessions for debugging."""

    def load_from_file(self, path: str) -> SessionExport:
        """Load a session export from file.

        Args:
            path: Path to the exported session file.

        Returns:
            SessionExport reconstructed from file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(file_path, "r") as f:
            content = f.read()

        # Detect format
        if content.strip().startswith("{"):
            data = json.loads(content)
        else:
            # JSONL format
            lines = content.strip().split("\n")
            header = json.loads(lines[0])
            events = [json.loads(line) for line in lines[1:]]
            header["events"] = events
            data = header

        if "session_id" not in data:
            raise ValueError("Invalid session export: missing 'session_id'")

        return SessionExport.from_dict(data)

    def load_from_dict(self, data: Dict[str, Any]) -> SessionExport:
        """Load from a dictionary (e.g., from an API response).

        Args:
            data: Dictionary with session export data.

        Returns:
            SessionExport instance.
        """
        return SessionExport.from_dict(data)

    def replay(
        self,
        session: SessionExport,
        handler: Optional[Callable[[SessionEvent], Optional[Dict[str, Any]]]] = None,
        stop_on_divergence: bool = False,
    ) -> ReplayResult:
        """Replay a session for debugging.

        Replays events in sequence, optionally executing a handler
        for each event and comparing results.

        Args:
            session: The session to replay.
            handler: Optional function to execute for each event.
                     Returns the expected output or None to skip.
            stop_on_divergence: Whether to stop on first divergence.

        Returns:
            ReplayResult with comparison data.
        """
        start_time = time.time()
        events_matched = 0
        events_diverged = 0
        divergences: List[Dict[str, Any]] = []

        for i, event in enumerate(session.events):
            if handler is None:
                events_matched += 1
                continue

            result = handler(event)
            if result is None:
                events_matched += 1
                continue

            # Compare with original
            if self._events_match(event.data, result):
                events_matched += 1
            else:
                events_diverged += 1
                divergences.append({
                    "event_index": i,
                    "event_type": event.event_type,
                    "expected": event.data,
                    "actual": result,
                    "timestamp": event.timestamp,
                })

                if stop_on_divergence:
                    break

        duration_ms = (time.time() - start_time) * 1000

        return ReplayResult(
            session_id=session.session_id,
            events_replayed=len(session.events),
            events_matched=events_matched,
            events_diverged=events_diverged,
            divergences=divergences,
            duration_ms=duration_ms,
            success=events_diverged == 0,
        )

    def _events_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Compare two event data dictionaries for equivalence."""
        # Simple deep equality check (ignoring timestamps and metadata)
        for key, value in expected.items():
            if key in ("timestamp", "duration_ms"):
                continue
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True

    def get_user_messages(self, session: SessionExport) -> List[str]:
        """Extract user messages from a session for quick inspection.

        Args:
            session: The session export.

        Returns:
            List of user message strings.
        """
        messages = []
        for event in session.events:
            if event.event_type == "user_message":
                content = event.data.get("content", event.data.get("message", ""))
                if content:
                    messages.append(content)
        return messages

    def get_tool_calls(self, session: SessionExport) -> List[Dict[str, Any]]:
        """Extract tool calls from a session.

        Args:
            session: The session export.

        Returns:
            List of tool call data dictionaries.
        """
        return [
            event.data
            for event in session.events
            if event.event_type == "tool_call"
        ]
