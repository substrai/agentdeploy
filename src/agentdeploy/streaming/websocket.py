"""WebSocket streaming for real-time agent responses.

Provides WebSocket connection management and token-by-token streaming
for deployed agents, designed for AWS API Gateway WebSocket APIs.

Key features:
- Connection lifecycle management (connect, message, disconnect)
- Token-by-token streaming with configurable chunk sizes
- Connection state tracking and heartbeat
- Multi-tenant connection isolation
- Graceful connection cleanup and TTL enforcement
- Message framing with sequence numbers for ordering
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set


class ConnectionState(Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    CLOSING = "closing"
    CLOSED = "closed"


class MessageType(Enum):
    """WebSocket message types for the streaming protocol."""

    STREAM_START = "stream_start"
    TOKEN = "token"
    STREAM_END = "stream_end"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    METADATA = "metadata"


@dataclass
class StreamMessage:
    """A single message in the streaming protocol.

    Attributes:
        type: The message type (token, stream_start, etc.)
        payload: The message content.
        sequence: Monotonically increasing sequence number.
        timestamp: Unix timestamp of message creation.
        connection_id: The WebSocket connection this belongs to.
    """

    type: MessageType
    payload: str = ""
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)
    connection_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket transmission."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "connectionId": self.connection_id,
            "metadata": self.metadata,
        })

    @classmethod
    def from_json(cls, data: str) -> "StreamMessage":
        """Deserialize from JSON."""
        parsed = json.loads(data)
        return cls(
            type=MessageType(parsed["type"]),
            payload=parsed.get("payload", ""),
            sequence=parsed.get("sequence", 0),
            timestamp=parsed.get("timestamp", time.time()),
            connection_id=parsed.get("connectionId", ""),
            metadata=parsed.get("metadata", {}),
        )


@dataclass
class Connection:
    """Represents a single WebSocket connection.

    Tracks connection state, metadata, and streaming statistics.
    """

    connection_id: str
    tenant_id: str = "default"
    agent_name: str = ""
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages_sent: int = 0
    tokens_streamed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _sequence_counter: int = 0

    def next_sequence(self) -> int:
        """Get the next sequence number for this connection."""
        self._sequence_counter += 1
        return self._sequence_counter

    @property
    def duration_seconds(self) -> float:
        """How long this connection has been active."""
        return time.time() - self.connected_at

    @property
    def is_active(self) -> bool:
        """Whether the connection is in an active state."""
        return self.state in (ConnectionState.CONNECTED, ConnectionState.STREAMING)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize connection info for monitoring."""
        return {
            "connection_id": self.connection_id,
            "tenant_id": self.tenant_id,
            "agent_name": self.agent_name,
            "state": self.state.value,
            "connected_at": self.connected_at,
            "duration_seconds": self.duration_seconds,
            "messages_sent": self.messages_sent,
            "tokens_streamed": self.tokens_streamed,
        }


class ConnectionManager:
    """Manages WebSocket connections with TTL and tenant isolation.

    Handles connection lifecycle, cleanup of stale connections,
    and provides tenant-scoped connection queries.

    Example:
        manager = ConnectionManager(ttl_seconds=3600)
        conn = manager.connect("conn-123", tenant_id="acme", agent_name="support-bot")
        manager.disconnect("conn-123")
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_connections_per_tenant: int = 100,
        heartbeat_interval_seconds: int = 30,
    ):
        self._connections: Dict[str, Connection] = {}
        self.ttl_seconds = ttl_seconds
        self.max_connections_per_tenant = max_connections_per_tenant
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    def connect(
        self,
        connection_id: str,
        tenant_id: str = "default",
        agent_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Connection:
        """Register a new WebSocket connection.

        Args:
            connection_id: Unique connection identifier (from API Gateway).
            tenant_id: Tenant this connection belongs to.
            agent_name: The agent being streamed to.
            metadata: Optional connection metadata.

        Returns:
            The created Connection object.

        Raises:
            ConnectionError: If tenant has exceeded max connections.
        """
        # Check tenant connection limit
        tenant_count = self.get_tenant_connection_count(tenant_id)
        if tenant_count >= self.max_connections_per_tenant:
            raise ConnectionError(
                f"Tenant '{tenant_id}' has reached max connections "
                f"({self.max_connections_per_tenant})"
            )

        conn = Connection(
            connection_id=connection_id,
            tenant_id=tenant_id,
            agent_name=agent_name,
            state=ConnectionState.CONNECTED,
            metadata=metadata or {},
        )
        self._connections[connection_id] = conn
        self._cleanup_stale()
        return conn

    def disconnect(self, connection_id: str) -> bool:
        """Close and remove a connection.

        Args:
            connection_id: The connection to close.

        Returns:
            True if the connection existed and was closed.
        """
        conn = self._connections.pop(connection_id, None)
        if conn:
            conn.state = ConnectionState.CLOSED
            return True
        return False

    def get(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    def get_tenant_connections(self, tenant_id: str) -> List[Connection]:
        """Get all active connections for a tenant."""
        return [
            c for c in self._connections.values()
            if c.tenant_id == tenant_id and c.is_active
        ]

    def get_tenant_connection_count(self, tenant_id: str) -> int:
        """Count active connections for a tenant."""
        return sum(
            1 for c in self._connections.values()
            if c.tenant_id == tenant_id and c.is_active
        )

    def get_active_connections(self) -> List[Connection]:
        """Get all active connections across all tenants."""
        return [c for c in self._connections.values() if c.is_active]

    @property
    def total_connections(self) -> int:
        """Total number of tracked connections."""
        return len(self._connections)

    def _cleanup_stale(self) -> None:
        """Remove connections that have exceeded TTL."""
        cutoff = time.time() - self.ttl_seconds
        stale = [
            cid for cid, conn in self._connections.items()
            if conn.last_activity < cutoff
        ]
        for cid in stale:
            self._connections[cid].state = ConnectionState.CLOSED
            del self._connections[cid]


class WebSocketHandler:
    """Handles WebSocket message routing and token streaming.

    Processes API Gateway WebSocket events ($connect, $disconnect, message)
    and provides token-by-token streaming to connected clients.

    Example:
        handler = WebSocketHandler(connection_manager=manager)

        # Handle API Gateway $connect
        conn = await handler.on_connect(connection_id="abc", tenant_id="acme")

        # Stream tokens to client
        async for msg in handler.stream_tokens("abc", token_generator()):
            await send_to_client(msg.to_json())

        # Handle disconnect
        await handler.on_disconnect("abc")
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        chunk_size: int = 1,
        flush_interval_ms: float = 50,
        on_stream_start: Optional[Callable] = None,
        on_stream_end: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.manager = connection_manager or ConnectionManager()
        self.chunk_size = chunk_size
        self.flush_interval_ms = flush_interval_ms
        self.on_stream_start = on_stream_start
        self.on_stream_end = on_stream_end
        self.on_error = on_error
        self._active_streams: Set[str] = set()

    async def on_connect(
        self,
        connection_id: str,
        tenant_id: str = "default",
        agent_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Connection:
        """Handle a new WebSocket connection ($connect route).

        Args:
            connection_id: API Gateway connection ID.
            tenant_id: Tenant identifier from auth context.
            agent_name: The agent this connection is for.
            metadata: Additional connection metadata.

        Returns:
            The established Connection.
        """
        conn = self.manager.connect(
            connection_id=connection_id,
            tenant_id=tenant_id,
            agent_name=agent_name,
            metadata=metadata,
        )
        return conn

    async def on_disconnect(self, connection_id: str) -> None:
        """Handle WebSocket disconnection ($disconnect route).

        Cleans up connection state and stops any active streams.
        """
        self._active_streams.discard(connection_id)
        self.manager.disconnect(connection_id)

    async def on_message(
        self,
        connection_id: str,
        body: str,
    ) -> Optional[StreamMessage]:
        """Handle an incoming WebSocket message.

        Args:
            connection_id: The sending connection.
            body: Raw message body.

        Returns:
            Response message if applicable, None otherwise.
        """
        conn = self.manager.get(connection_id)
        if not conn:
            return StreamMessage(
                type=MessageType.ERROR,
                payload="Connection not found",
                connection_id=connection_id,
            )

        conn.last_activity = time.time()

        # Parse the incoming message
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return StreamMessage(
                type=MessageType.ERROR,
                payload="Invalid JSON",
                connection_id=connection_id,
            )

        # Route based on action
        action = data.get("action", "message")
        if action == "heartbeat":
            return StreamMessage(
                type=MessageType.HEARTBEAT,
                payload="pong",
                sequence=conn.next_sequence(),
                connection_id=connection_id,
            )

        return None

    async def stream_tokens(
        self,
        connection_id: str,
        token_source: AsyncIterator[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamMessage]:
        """Stream tokens to a WebSocket connection.

        Yields StreamMessage objects for each token/chunk, with proper
        framing (stream_start, tokens, stream_end).

        Args:
            connection_id: Target connection.
            token_source: Async iterator yielding tokens.
            metadata: Optional metadata to include in stream_start.

        Yields:
            StreamMessage for each frame in the stream.
        """
        conn = self.manager.get(connection_id)
        if not conn:
            yield StreamMessage(
                type=MessageType.ERROR,
                payload="Connection not found",
                connection_id=connection_id,
            )
            return

        conn.state = ConnectionState.STREAMING
        self._active_streams.add(connection_id)
        stream_id = f"stream-{uuid.uuid4().hex[:8]}"

        # Fire stream start callback
        if self.on_stream_start:
            if asyncio.iscoroutinefunction(self.on_stream_start):
                await self.on_stream_start(connection_id, stream_id)
            else:
                self.on_stream_start(connection_id, stream_id)

        # Emit stream_start
        yield StreamMessage(
            type=MessageType.STREAM_START,
            payload=stream_id,
            sequence=conn.next_sequence(),
            connection_id=connection_id,
            metadata=metadata or {},
        )

        # Stream tokens
        buffer: List[str] = []
        total_tokens = 0

        try:
            async for token in token_source:
                if connection_id not in self._active_streams:
                    break  # Connection was closed during streaming

                buffer.append(token)
                total_tokens += 1

                # Flush when buffer reaches chunk_size
                if len(buffer) >= self.chunk_size:
                    chunk = "".join(buffer)
                    buffer.clear()

                    yield StreamMessage(
                        type=MessageType.TOKEN,
                        payload=chunk,
                        sequence=conn.next_sequence(),
                        connection_id=connection_id,
                    )
                    conn.messages_sent += 1
                    conn.tokens_streamed += 1
                    conn.last_activity = time.time()

                    # Rate limiting via flush interval
                    if self.flush_interval_ms > 0:
                        await asyncio.sleep(self.flush_interval_ms / 1000)

            # Flush remaining buffer
            if buffer:
                chunk = "".join(buffer)
                yield StreamMessage(
                    type=MessageType.TOKEN,
                    payload=chunk,
                    sequence=conn.next_sequence(),
                    connection_id=connection_id,
                )
                conn.messages_sent += 1

        except Exception as e:
            yield StreamMessage(
                type=MessageType.ERROR,
                payload=str(e),
                sequence=conn.next_sequence(),
                connection_id=connection_id,
            )
            if self.on_error:
                if asyncio.iscoroutinefunction(self.on_error):
                    await self.on_error(connection_id, e)
                else:
                    self.on_error(connection_id, e)

        finally:
            self._active_streams.discard(connection_id)
            if conn.is_active:
                conn.state = ConnectionState.CONNECTED

        # Emit stream_end
        yield StreamMessage(
            type=MessageType.STREAM_END,
            payload=stream_id,
            sequence=conn.next_sequence(),
            connection_id=connection_id,
            metadata={"total_tokens": total_tokens},
        )

        # Fire stream end callback
        if self.on_stream_end:
            if asyncio.iscoroutinefunction(self.on_stream_end):
                await self.on_stream_end(connection_id, stream_id, total_tokens)
            else:
                self.on_stream_end(connection_id, stream_id, total_tokens)

    async def send_message(
        self,
        connection_id: str,
        message_type: MessageType,
        payload: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamMessage]:
        """Send a single message to a connection.

        Args:
            connection_id: Target connection.
            message_type: Type of message to send.
            payload: Message content.
            metadata: Optional metadata.

        Returns:
            The sent StreamMessage, or None if connection not found.
        """
        conn = self.manager.get(connection_id)
        if not conn or not conn.is_active:
            return None

        msg = StreamMessage(
            type=message_type,
            payload=payload,
            sequence=conn.next_sequence(),
            connection_id=connection_id,
            metadata=metadata or {},
        )
        conn.messages_sent += 1
        conn.last_activity = time.time()
        return msg

    @property
    def active_stream_count(self) -> int:
        """Number of currently active streams."""
        return len(self._active_streams)

    def is_streaming(self, connection_id: str) -> bool:
        """Check if a connection is currently streaming."""
        return connection_id in self._active_streams
