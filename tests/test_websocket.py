"""Tests for WebSocket streaming handler."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from agentdeploy.streaming.websocket import (
    Connection,
    ConnectionManager,
    ConnectionState,
    MessageType,
    StreamMessage,
    WebSocketHandler,
)


# --- Helpers ---


async def mock_token_generator(tokens: list[str]):
    """Create an async generator that yields tokens."""
    for token in tokens:
        yield token


async def slow_token_generator(tokens: list[str], delay: float = 0.01):
    """Token generator with delay between tokens."""
    for token in tokens:
        await asyncio.sleep(delay)
        yield token


# --- Tests ---


class TestConnectionManager:
    """Connection manager tests."""

    def test_connect_creates_connection(self):
        """connect() should create and track a new connection."""
        manager = ConnectionManager()
        conn = manager.connect("conn-1", tenant_id="acme", agent_name="bot")

        assert conn.connection_id == "conn-1"
        assert conn.tenant_id == "acme"
        assert conn.agent_name == "bot"
        assert conn.state == ConnectionState.CONNECTED
        assert manager.total_connections == 1

    def test_disconnect_removes_connection(self):
        """disconnect() should remove the connection."""
        manager = ConnectionManager()
        manager.connect("conn-1")

        result = manager.disconnect("conn-1")
        assert result is True
        assert manager.total_connections == 0

    def test_disconnect_nonexistent_returns_false(self):
        """disconnect() on unknown connection should return False."""
        manager = ConnectionManager()
        assert manager.disconnect("unknown") is False

    def test_tenant_connection_limit(self):
        """Should enforce max connections per tenant."""
        manager = ConnectionManager(max_connections_per_tenant=2)
        manager.connect("conn-1", tenant_id="acme")
        manager.connect("conn-2", tenant_id="acme")

        with pytest.raises(ConnectionError, match="max connections"):
            manager.connect("conn-3", tenant_id="acme")

    def test_tenant_isolation(self):
        """Connections from different tenants should be isolated."""
        manager = ConnectionManager(max_connections_per_tenant=2)
        manager.connect("conn-1", tenant_id="acme")
        manager.connect("conn-2", tenant_id="acme")
        # Different tenant should still be able to connect
        manager.connect("conn-3", tenant_id="other")

        assert manager.get_tenant_connection_count("acme") == 2
        assert manager.get_tenant_connection_count("other") == 1

    def test_get_active_connections(self):
        """get_active_connections should return only active ones."""
        manager = ConnectionManager()
        manager.connect("conn-1")
        manager.connect("conn-2")
        manager.disconnect("conn-1")

        active = manager.get_active_connections()
        assert len(active) == 1
        assert active[0].connection_id == "conn-2"

    def test_stale_connection_cleanup(self):
        """Stale connections should be cleaned up based on TTL."""
        manager = ConnectionManager(ttl_seconds=1)
        conn = manager.connect("old-conn")
        # Simulate old connection
        conn.last_activity = time.time() - 10

        # Trigger cleanup by connecting a new one
        manager.connect("new-conn")

        assert manager.get("old-conn") is None
        assert manager.total_connections == 1


class TestStreamMessage:
    """StreamMessage serialization tests."""

    def test_to_json_serialization(self):
        """to_json should produce valid JSON with all fields."""
        msg = StreamMessage(
            type=MessageType.TOKEN,
            payload="Hello",
            sequence=1,
            connection_id="conn-1",
            metadata={"model": "claude"},
        )
        data = json.loads(msg.to_json())

        assert data["type"] == "token"
        assert data["payload"] == "Hello"
        assert data["sequence"] == 1
        assert data["connectionId"] == "conn-1"
        assert data["metadata"]["model"] == "claude"

    def test_from_json_deserialization(self):
        """from_json should correctly parse a JSON message."""
        raw = json.dumps({
            "type": "stream_start",
            "payload": "stream-abc",
            "sequence": 0,
            "timestamp": 1234567890.0,
            "connectionId": "conn-1",
            "metadata": {},
        })
        msg = StreamMessage.from_json(raw)

        assert msg.type == MessageType.STREAM_START
        assert msg.payload == "stream-abc"
        assert msg.connection_id == "conn-1"

    def test_roundtrip_serialization(self):
        """to_json -> from_json should preserve message content."""
        original = StreamMessage(
            type=MessageType.TOKEN,
            payload="world",
            sequence=5,
            connection_id="conn-x",
        )
        restored = StreamMessage.from_json(original.to_json())

        assert restored.type == original.type
        assert restored.payload == original.payload
        assert restored.sequence == original.sequence


class TestWebSocketHandler:
    """WebSocket handler tests."""

    @pytest.mark.asyncio
    async def test_on_connect_establishes_connection(self):
        """on_connect should create a tracked connection."""
        handler = WebSocketHandler()
        conn = await handler.on_connect("conn-1", tenant_id="acme", agent_name="bot")

        assert conn.connection_id == "conn-1"
        assert conn.state == ConnectionState.CONNECTED
        assert handler.manager.total_connections == 1

    @pytest.mark.asyncio
    async def test_on_disconnect_cleans_up(self):
        """on_disconnect should remove the connection."""
        handler = WebSocketHandler()
        await handler.on_connect("conn-1")
        await handler.on_disconnect("conn-1")

        assert handler.manager.total_connections == 0

    @pytest.mark.asyncio
    async def test_stream_tokens_produces_correct_sequence(self):
        """stream_tokens should yield start, tokens, and end messages."""
        handler = WebSocketHandler(chunk_size=1, flush_interval_ms=0)
        await handler.on_connect("conn-1")

        tokens = ["Hello", " ", "world"]
        messages = []

        async for msg in handler.stream_tokens("conn-1", mock_token_generator(tokens)):
            messages.append(msg)

        # Should have: stream_start + 3 tokens + stream_end = 5 messages
        assert len(messages) == 5
        assert messages[0].type == MessageType.STREAM_START
        assert messages[1].type == MessageType.TOKEN
        assert messages[1].payload == "Hello"
        assert messages[2].type == MessageType.TOKEN
        assert messages[2].payload == " "
        assert messages[3].type == MessageType.TOKEN
        assert messages[3].payload == "world"
        assert messages[4].type == MessageType.STREAM_END

    @pytest.mark.asyncio
    async def test_stream_tokens_with_chunking(self):
        """Tokens should be buffered and sent in chunks."""
        handler = WebSocketHandler(chunk_size=3, flush_interval_ms=0)
        await handler.on_connect("conn-1")

        tokens = ["a", "b", "c", "d", "e"]
        messages = []

        async for msg in handler.stream_tokens("conn-1", mock_token_generator(tokens)):
            messages.append(msg)

        # stream_start + chunk("abc") + chunk("de") + stream_end = 4
        token_messages = [m for m in messages if m.type == MessageType.TOKEN]
        assert len(token_messages) == 2
        assert token_messages[0].payload == "abc"
        assert token_messages[1].payload == "de"

    @pytest.mark.asyncio
    async def test_stream_to_nonexistent_connection_yields_error(self):
        """Streaming to unknown connection should yield an error message."""
        handler = WebSocketHandler()

        messages = []
        async for msg in handler.stream_tokens("unknown", mock_token_generator(["hi"])):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_sequence_numbers_are_monotonic(self):
        """Sequence numbers should increase monotonically."""
        handler = WebSocketHandler(chunk_size=1, flush_interval_ms=0)
        await handler.on_connect("conn-1")

        messages = []
        async for msg in handler.stream_tokens(
            "conn-1", mock_token_generator(["a", "b", "c"])
        ):
            messages.append(msg)

        sequences = [m.sequence for m in messages]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)  # All unique

    @pytest.mark.asyncio
    async def test_on_message_heartbeat(self):
        """Heartbeat messages should return a pong response."""
        handler = WebSocketHandler()
        await handler.on_connect("conn-1")

        response = await handler.on_message(
            "conn-1", json.dumps({"action": "heartbeat"})
        )

        assert response is not None
        assert response.type == MessageType.HEARTBEAT
        assert response.payload == "pong"

    @pytest.mark.asyncio
    async def test_on_message_invalid_json(self):
        """Invalid JSON should return an error message."""
        handler = WebSocketHandler()
        await handler.on_connect("conn-1")

        response = await handler.on_message("conn-1", "not json{{{")

        assert response.type == MessageType.ERROR
        assert "Invalid JSON" in response.payload

    @pytest.mark.asyncio
    async def test_connection_stats_updated_during_streaming(self):
        """Connection stats should be updated as tokens are streamed."""
        handler = WebSocketHandler(chunk_size=1, flush_interval_ms=0)
        conn = await handler.on_connect("conn-1")

        assert conn.messages_sent == 0

        messages = []
        async for msg in handler.stream_tokens(
            "conn-1", mock_token_generator(["a", "b", "c"])
        ):
            messages.append(msg)

        assert conn.messages_sent >= 3
        assert conn.tokens_streamed >= 3

    @pytest.mark.asyncio
    async def test_send_message_to_active_connection(self):
        """send_message should create and return a message."""
        handler = WebSocketHandler()
        await handler.on_connect("conn-1")

        msg = await handler.send_message(
            "conn-1", MessageType.METADATA, "model info", metadata={"model": "v1"}
        )

        assert msg is not None
        assert msg.type == MessageType.METADATA
        assert msg.payload == "model info"

    @pytest.mark.asyncio
    async def test_send_message_to_closed_connection_returns_none(self):
        """send_message to a closed connection should return None."""
        handler = WebSocketHandler()
        await handler.on_connect("conn-1")
        await handler.on_disconnect("conn-1")

        msg = await handler.send_message("conn-1", MessageType.TOKEN, "hello")
        assert msg is None


class TestConnection:
    """Connection dataclass tests."""

    def test_connection_is_active(self):
        """is_active should be True for CONNECTED and STREAMING states."""
        conn = Connection(connection_id="test", state=ConnectionState.CONNECTED)
        assert conn.is_active is True

        conn.state = ConnectionState.STREAMING
        assert conn.is_active is True

        conn.state = ConnectionState.CLOSED
        assert conn.is_active is False

    def test_connection_to_dict(self):
        """to_dict should include all relevant fields."""
        conn = Connection(
            connection_id="conn-1",
            tenant_id="acme",
            agent_name="bot",
            state=ConnectionState.CONNECTED,
        )
        data = conn.to_dict()

        assert data["connection_id"] == "conn-1"
        assert data["tenant_id"] == "acme"
        assert data["agent_name"] == "bot"
        assert data["state"] == "connected"
        assert "duration_seconds" in data

    def test_sequence_counter_increments(self):
        """next_sequence should return monotonically increasing values."""
        conn = Connection(connection_id="test")
        assert conn.next_sequence() == 1
        assert conn.next_sequence() == 2
        assert conn.next_sequence() == 3
