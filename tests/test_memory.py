"""Tests for MemoryStore, ConversationMemory, and AgentStateMemory."""

from __future__ import annotations

import pytest

from ai_agent_framework.memory.base import AgentStateMemory, ConversationMemory
from ai_agent_framework.memory.in_memory import InMemoryStore


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        store = InMemoryStore()
        await store.set("ns", "key", {"a": 1})
        assert await store.get("ns", "key") == {"a": 1}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        store = InMemoryStore()
        assert await store.get("ns", "missing") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryStore()
        await store.set("ns", "key", 1)
        await store.delete("ns", "key")
        assert await store.get("ns", "key") is None

    @pytest.mark.asyncio
    async def test_append_and_get_list(self):
        store = InMemoryStore()
        await store.append("ns", "list", "a")
        await store.append("ns", "list", "b")
        assert await store.get_list("ns", "list") == ["a", "b"]

    @pytest.mark.asyncio
    async def test_append_respects_max_length(self):
        store = InMemoryStore()
        for i in range(5):
            await store.append("ns", "list", i, max_length=3)
        assert await store.get_list("ns", "list") == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_stored_values_are_isolated_copies(self):
        store = InMemoryStore()
        payload = {"a": 1}
        await store.set("ns", "key", payload)
        payload["a"] = 999
        stored = await store.get("ns", "key")
        assert stored == {"a": 1}


class TestConversationMemory:
    @pytest.mark.asyncio
    async def test_add_and_get_history(self, conversation_memory: ConversationMemory):
        await conversation_memory.add_message("session1", "user", "hi")
        await conversation_memory.add_message("session1", "assistant", "hello")
        history = await conversation_memory.get_history("session1")
        assert history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    @pytest.mark.asyncio
    async def test_history_isolated_per_session(self, conversation_memory: ConversationMemory):
        await conversation_memory.add_message("session1", "user", "hi")
        await conversation_memory.add_message("session2", "user", "yo")
        assert len(await conversation_memory.get_history("session1")) == 1
        assert len(await conversation_memory.get_history("session2")) == 1

    @pytest.mark.asyncio
    async def test_long_message_is_trimmed(self, conversation_memory: ConversationMemory):
        long_text = "x" * 5000
        await conversation_memory.add_message("session1", "user", long_text)
        history = await conversation_memory.get_history("session1")
        assert len(history[0]["content"]) < 5000

    @pytest.mark.asyncio
    async def test_max_history_trims_oldest(self):
        memory = ConversationMemory(InMemoryStore(), max_history=3)
        for i in range(5):
            await memory.add_message("s1", "user", str(i))
        history = await memory.get_history("s1")
        assert [h["content"] for h in history] == ["2", "3", "4"]

    @pytest.mark.asyncio
    async def test_summary_events(self, conversation_memory: ConversationMemory):
        await conversation_memory.add_summary_event("session1", "email sent")
        await conversation_memory.add_summary_event("session1", "ticket created")
        context = await conversation_memory.get_summary_context("session1")
        assert "email sent" in context
        assert "ticket created" in context

    @pytest.mark.asyncio
    async def test_clear_removes_history_and_summary(self, conversation_memory: ConversationMemory):
        await conversation_memory.add_message("session1", "user", "hi")
        await conversation_memory.add_summary_event("session1", "event")
        await conversation_memory.clear("session1")
        assert await conversation_memory.get_history("session1") == []
        assert await conversation_memory.get_summary_context("session1") == ""


class TestAgentStateMemory:
    @pytest.mark.asyncio
    async def test_get_default_when_unset(self):
        state = AgentStateMemory(InMemoryStore(), "agent_a")
        assert await state.get("key", default="fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        state = AgentStateMemory(InMemoryStore(), "agent_a")
        await state.set("last_query", "hello")
        assert await state.get("last_query") == "hello"

    @pytest.mark.asyncio
    async def test_namespaced_per_agent(self):
        store = InMemoryStore()
        state_a = AgentStateMemory(store, "agent_a")
        state_b = AgentStateMemory(store, "agent_b")
        await state_a.set("key", "a_value")
        await state_b.set("key", "b_value")
        assert await state_a.get("key") == "a_value"
        assert await state_b.get("key") == "b_value"
