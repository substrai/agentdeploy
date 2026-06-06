"""Tests for conversation memory strategies."""

import json
import time

import pytest

from agentdeploy.memory.strategies import (
    BufferMemory,
    HybridMemory,
    MemoryState,
    Message,
    MessageRole,
    SummaryMemory,
)


def make_message(role: str = "user", content: str = "Hello") -> Message:
    return Message(role=MessageRole(role), content=content)


def make_messages(count: int, role: str = "user") -> list:
    return [make_message(role, f"Message {i}") for i in range(count)]


class TestMessage:
    def test_to_dict(self):
        msg = make_message("user", "test content")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "test content"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {"role": "assistant", "content": "response", "timestamp": 123.0}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "response"
        assert msg.timestamp == 123.0

    def test_roundtrip_serialization(self):
        original = make_message("user", "roundtrip test")
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content


class TestMemoryState:
    def test_json_serialization(self):
        state = MemoryState(
            messages=[{"role": "user", "content": "hi"}],
            summary="test summary",
            metadata={"key": "value"},
        )
        json_str = state.to_json()
        restored = MemoryState.from_json(json_str)
        assert restored.messages == state.messages
        assert restored.summary == state.summary
        assert restored.metadata == state.metadata


class TestBufferMemory:
    def test_add_and_get_messages(self):
        mem = BufferMemory(window_size=10)
        msg = make_message("user", "hello")
        mem.add_message(msg)
        assert mem.message_count == 1
        assert mem.get_messages()[0].content == "hello"

    def test_window_size_enforcement(self):
        mem = BufferMemory(window_size=5)
        for msg in make_messages(10):
            mem.add_message(msg)
        assert mem.message_count == 5
        # Should keep the most recent messages
        assert mem.get_messages()[0].content == "Message 5"

    def test_get_context_window(self):
        mem = BufferMemory()
        mem.add_message(make_message("user", "question"))
        mem.add_message(make_message("assistant", "answer"))
        context = mem.get_context_window()
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "question"}
        assert context[1] == {"role": "assistant", "content": "answer"}

    def test_clear(self):
        mem = BufferMemory()
        mem.add_message(make_message())
        mem.clear()
        assert mem.message_count == 0

    def test_serialize_and_restore(self):
        mem = BufferMemory(window_size=10)
        mem.add_message(make_message("user", "saved"))
        state = mem.serialize()

        new_mem = BufferMemory(window_size=10)
        new_mem.restore(state)
        assert new_mem.message_count == 1
        assert new_mem.get_messages()[0].content == "saved"

    def test_max_tokens_enforcement(self):
        mem = BufferMemory(window_size=100, max_tokens=10)
        for i in range(5):
            msg = make_message("user", "word " * 5)  # ~5 tokens each
            mem.add_message(msg)
        # Should have evicted some messages to stay under 10 tokens
        total_tokens = sum(len(m.content.split()) for m in mem.get_messages())
        assert total_tokens <= 10


class TestSummaryMemory:
    def test_add_messages_below_trigger(self):
        mem = SummaryMemory(summary_trigger=10)
        mem.add_message(make_message("user", "hello"))
        assert mem.message_count == 1
        assert mem.summary == ""

    def test_triggers_summarization(self):
        mem = SummaryMemory(summary_trigger=5, keep_recent=2)
        for msg in make_messages(5):
            mem.add_message(msg)
        # After trigger, should keep only recent messages
        assert mem.message_count == 2
        assert mem.summary != ""

    def test_summary_preserves_recent(self):
        mem = SummaryMemory(summary_trigger=5, keep_recent=2)
        for msg in make_messages(5):
            mem.add_message(msg)
        messages = mem.get_messages()
        assert messages[-1].content == "Message 4"
        assert messages[-2].content == "Message 3"

    def test_context_window_includes_summary(self):
        mem = SummaryMemory(summary_trigger=5, keep_recent=2)
        for msg in make_messages(5):
            mem.add_message(msg)
        context = mem.get_context_window()
        # First message should be the summary
        assert context[0]["role"] == "system"
        assert "summary" in context[0]["content"].lower()

    def test_custom_summarize_fn(self):
        def custom_summary(messages, existing):
            return f"Custom: {len(messages)} messages processed"

        mem = SummaryMemory(summary_trigger=3, keep_recent=1, summarize_fn=custom_summary)
        for msg in make_messages(3):
            mem.add_message(msg)
        assert "Custom: 2 messages processed" in mem.summary

    def test_clear_resets_summary(self):
        mem = SummaryMemory(summary_trigger=3, keep_recent=1)
        for msg in make_messages(3):
            mem.add_message(msg)
        mem.clear()
        assert mem.message_count == 0
        assert mem.summary == ""

    def test_serialize_and_restore(self):
        mem = SummaryMemory(summary_trigger=3, keep_recent=1)
        for msg in make_messages(3):
            mem.add_message(msg)

        state = mem.serialize()
        new_mem = SummaryMemory(summary_trigger=3, keep_recent=1)
        new_mem.restore(state)
        assert new_mem.summary == mem.summary
        assert new_mem.message_count == mem.message_count


class TestHybridMemory:
    def test_add_within_buffer(self):
        mem = HybridMemory(buffer_size=10)
        mem.add_message(make_message("user", "hello"))
        assert mem.message_count == 1
        assert mem.total_messages_seen == 1

    def test_overflow_to_summary(self):
        mem = HybridMemory(buffer_size=5, summary_trigger=3)
        for msg in make_messages(10):
            mem.add_message(msg)
        # Buffer should be at max size
        assert mem.message_count == 5
        assert mem.total_messages_seen == 10
        # Summary should have been created from overflow
        assert mem.summary != ""

    def test_context_window_structure(self):
        mem = HybridMemory(buffer_size=3, summary_trigger=2)
        for msg in make_messages(8):
            mem.add_message(msg)
        context = mem.get_context_window()
        # Should have summary + buffer messages
        assert len(context) >= 3  # At least summary + buffer
        assert any("summary" in c.get("content", "").lower() for c in context if c["role"] == "system")

    def test_clear(self):
        mem = HybridMemory(buffer_size=5)
        for msg in make_messages(3):
            mem.add_message(msg)
        mem.clear()
        assert mem.message_count == 0
        assert mem.summary == ""
        assert mem.total_messages_seen == 0

    def test_serialize_and_restore(self):
        mem = HybridMemory(buffer_size=5, summary_trigger=3)
        for msg in make_messages(10):
            mem.add_message(msg)

        state = mem.serialize()
        new_mem = HybridMemory(buffer_size=5, summary_trigger=3)
        new_mem.restore(state)
        assert new_mem.message_count == mem.message_count
        assert new_mem.summary == mem.summary

    def test_custom_summarize_fn(self):
        def custom_fn(messages, existing):
            return f"Hybrid summary: {len(messages)} msgs"

        mem = HybridMemory(buffer_size=3, summary_trigger=2, summarize_fn=custom_fn)
        for msg in make_messages(7):
            mem.add_message(msg)
        assert "Hybrid summary" in mem.summary

    def test_buffer_keeps_most_recent(self):
        mem = HybridMemory(buffer_size=3, summary_trigger=5)
        for msg in make_messages(6):
            mem.add_message(msg)
        messages = mem.get_messages()
        assert messages[-1].content == "Message 5"
        assert messages[0].content == "Message 3"
