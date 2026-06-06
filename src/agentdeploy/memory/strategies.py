"""Conversation memory strategies for AgentDeploy.

Provides pluggable memory management with three strategies:
- BufferMemory: Full conversation buffer with configurable window
- SummaryMemory: Rolling summary that compresses older messages
- HybridMemory: Combines buffer for recent + summary for older messages
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class MessageRole(str, Enum):
    """Roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """A single conversation message."""

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
        )


@dataclass
class MemoryState:
    """Serializable state of the memory."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "messages": self.messages,
                "summary": self.summary,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "MemoryState":
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        return cls(
            messages=parsed.get("messages", []),
            summary=parsed.get("summary", ""),
            metadata=parsed.get("metadata", {}),
        )


class MemoryStrategy(ABC):
    """Abstract base class for conversation memory strategies."""

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        ...

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Get messages from memory for context."""
        ...

    @abstractmethod
    def get_context_window(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context window."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        ...

    @abstractmethod
    def serialize(self) -> MemoryState:
        """Serialize memory state."""
        ...

    @abstractmethod
    def restore(self, state: MemoryState) -> None:
        """Restore memory from serialized state."""
        ...

    @property
    @abstractmethod
    def message_count(self) -> int:
        """Number of messages in memory."""
        ...


class BufferMemory(MemoryStrategy):
    """Full conversation buffer with configurable window size.

    Stores all messages up to the window size. When the window is full,
    oldest messages are dropped (FIFO).
    """

    def __init__(self, window_size: int = 50, max_tokens: Optional[int] = None):
        """Initialize buffer memory.

        Args:
            window_size: Maximum number of messages to retain.
            max_tokens: Optional maximum total token count.
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        self._messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        """Add a message, evicting oldest if window is full."""
        self._messages.append(message)
        self._enforce_window()

    def get_messages(self) -> List[Message]:
        """Get all messages in the buffer."""
        return list(self._messages)

    def get_context_window(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context."""
        return [{"role": m.role.value, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        """Clear the buffer."""
        self._messages.clear()

    def serialize(self) -> MemoryState:
        """Serialize buffer state."""
        return MemoryState(
            messages=[m.to_dict() for m in self._messages],
            metadata={"window_size": self.window_size, "strategy": "buffer"},
        )

    def restore(self, state: MemoryState) -> None:
        """Restore buffer from state."""
        self._messages = [Message.from_dict(m) for m in state.messages]

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def _enforce_window(self) -> None:
        """Enforce window size constraint."""
        while len(self._messages) > self.window_size:
            self._messages.pop(0)

        if self.max_tokens is not None:
            total_tokens = sum(
                m.token_count or len(m.content.split()) for m in self._messages
            )
            while total_tokens > self.max_tokens and len(self._messages) > 1:
                removed = self._messages.pop(0)
                total_tokens -= removed.token_count or len(removed.content.split())


class SummaryMemory(MemoryStrategy):
    """Rolling summary memory that compresses older messages.

    Maintains a running summary of the conversation. When messages exceed
    the trigger threshold, they are summarized and replaced.
    """

    def __init__(
        self,
        summary_trigger: int = 10,
        summarize_fn: Optional[Callable[[List[Message], str], str]] = None,
        keep_recent: int = 2,
    ):
        """Initialize summary memory.

        Args:
            summary_trigger: Number of messages before triggering summarization.
            summarize_fn: Function to summarize messages. Uses default if None.
            keep_recent: Number of recent messages to keep unsummarized.
        """
        self.summary_trigger = summary_trigger
        self.keep_recent = keep_recent
        self._summarize_fn = summarize_fn or self._default_summarize
        self._messages: List[Message] = []
        self._summary: str = ""

    def add_message(self, message: Message) -> None:
        """Add message and trigger summarization if needed."""
        self._messages.append(message)
        if len(self._messages) >= self.summary_trigger:
            self._trigger_summarization()

    def get_messages(self) -> List[Message]:
        """Get current messages (post-summarization)."""
        return list(self._messages)

    def get_context_window(self) -> List[Dict[str, str]]:
        """Get context with summary prefix if available."""
        context = []
        if self._summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self._summary}",
            })
        context.extend(
            {"role": m.role.value, "content": m.content} for m in self._messages
        )
        return context

    @property
    def summary(self) -> str:
        """Get the current summary."""
        return self._summary

    def clear(self) -> None:
        """Clear messages and summary."""
        self._messages.clear()
        self._summary = ""

    def serialize(self) -> MemoryState:
        """Serialize summary memory state."""
        return MemoryState(
            messages=[m.to_dict() for m in self._messages],
            summary=self._summary,
            metadata={
                "summary_trigger": self.summary_trigger,
                "strategy": "summary",
            },
        )

    def restore(self, state: MemoryState) -> None:
        """Restore summary memory from state."""
        self._messages = [Message.from_dict(m) for m in state.messages]
        self._summary = state.summary

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def _trigger_summarization(self) -> None:
        """Summarize older messages and keep only recent ones."""
        to_summarize = self._messages[: -self.keep_recent] if self.keep_recent > 0 else self._messages
        to_keep = self._messages[-self.keep_recent:] if self.keep_recent > 0 else []

        self._summary = self._summarize_fn(to_summarize, self._summary)
        self._messages = to_keep

    def _default_summarize(
        self, messages: List[Message], existing_summary: str
    ) -> str:
        """Default summarization: concatenate message previews."""
        parts = []
        if existing_summary:
            parts.append(existing_summary)

        for msg in messages:
            preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
            parts.append(f"{msg.role.value}: {preview}")

        return " | ".join(parts)


class HybridMemory(MemoryStrategy):
    """Hybrid memory combining buffer for recent + summary for older messages.

    Maintains a full buffer of recent messages and summarizes older ones,
    providing both detailed recent context and compressed historical context.
    """

    def __init__(
        self,
        buffer_size: int = 20,
        summary_trigger: int = 10,
        summarize_fn: Optional[Callable[[List[Message], str], str]] = None,
    ):
        """Initialize hybrid memory.

        Args:
            buffer_size: Maximum messages in the active buffer.
            summary_trigger: Messages beyond buffer before summarizing.
            summarize_fn: Custom summarization function.
        """
        self.buffer_size = buffer_size
        self.summary_trigger = summary_trigger
        self._summarize_fn = summarize_fn or self._default_summarize
        self._buffer: List[Message] = []
        self._overflow: List[Message] = []
        self._summary: str = ""
        self._total_messages: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the hybrid memory."""
        self._total_messages += 1
        self._buffer.append(message)

        if len(self._buffer) > self.buffer_size:
            # Move oldest to overflow
            overflow_msg = self._buffer.pop(0)
            self._overflow.append(overflow_msg)

            # Check if overflow needs summarization
            if len(self._overflow) >= self.summary_trigger:
                self._summarize_overflow()

    def get_messages(self) -> List[Message]:
        """Get all buffered messages."""
        return list(self._buffer)

    def get_context_window(self) -> List[Dict[str, str]]:
        """Get context with summary + buffer messages."""
        context = []
        if self._summary:
            context.append({
                "role": "system",
                "content": f"Conversation history summary: {self._summary}",
            })
        # Include overflow messages not yet summarized
        for m in self._overflow:
            context.append({"role": m.role.value, "content": m.content})
        # Include buffer messages
        for m in self._buffer:
            context.append({"role": m.role.value, "content": m.content})
        return context

    @property
    def summary(self) -> str:
        """Get the current summary."""
        return self._summary

    def clear(self) -> None:
        """Clear all memory."""
        self._buffer.clear()
        self._overflow.clear()
        self._summary = ""
        self._total_messages = 0

    def serialize(self) -> MemoryState:
        """Serialize hybrid memory state."""
        return MemoryState(
            messages=[m.to_dict() for m in self._buffer],
            summary=self._summary,
            metadata={
                "overflow": [m.to_dict() for m in self._overflow],
                "buffer_size": self.buffer_size,
                "total_messages": self._total_messages,
                "strategy": "hybrid",
            },
        )

    def restore(self, state: MemoryState) -> None:
        """Restore hybrid memory from state."""
        self._buffer = [Message.from_dict(m) for m in state.messages]
        self._summary = state.summary
        self._overflow = [
            Message.from_dict(m) for m in state.metadata.get("overflow", [])
        ]
        self._total_messages = state.metadata.get("total_messages", len(self._buffer))

    @property
    def message_count(self) -> int:
        return len(self._buffer)

    @property
    def total_messages_seen(self) -> int:
        """Total messages ever added to this memory."""
        return self._total_messages

    def _summarize_overflow(self) -> None:
        """Summarize overflow messages."""
        self._summary = self._summarize_fn(self._overflow, self._summary)
        self._overflow.clear()

    def _default_summarize(
        self, messages: List[Message], existing_summary: str
    ) -> str:
        """Default summarization for overflow messages."""
        parts = []
        if existing_summary:
            parts.append(existing_summary)
        for msg in messages:
            preview = msg.content[:80] + ("..." if len(msg.content) > 80 else "")
            parts.append(f"{msg.role.value}: {preview}")
        return " | ".join(parts)
