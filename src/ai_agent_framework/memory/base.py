"""
Memory interfaces.

The framework distinguishes four kinds of state, all provider-independent
and none tied to any specific host application:

* **Conversation state** -- the message history for a session, used to
  give the model context across turns. Modeled by :class:`ConversationMemory`.
* **Task state** -- the working state of a single task's execution
  (which agents ran, what they returned). This lives on
  :class:`~ai_agent_framework.core.task.TaskResult` /
  :class:`~ai_agent_framework.core.context.ExecutionContext`, not here --
  it is transient by nature, scoped to one task's lifetime.
* **Agent state** -- arbitrary key/value state an individual agent wants
  to persist between invocations (e.g. "last product id discussed").
  Namespaced per agent so different agents can't collide.
* **Long-term memory** -- durable facts that outlive a single session
  (summaries, key facts extracted from conversation), stored separately
  from raw conversation history so it doesn't get evicted by a
  rolling-window trim.

:class:`MemoryStore` is the storage backend interface (in-memory by
default; host applications can implement it against Redis, Postgres,
etc.). :class:`ConversationMemory` is a higher-level, ergonomic wrapper
used by agents/supervisors that speaks in terms of messages and summaries
rather than raw key/value gets and sets.
"""

from __future__ import annotations

from typing import Any, Protocol


class MemoryStore(Protocol):
    """Storage backend interface for all memory kinds.

    A minimal key/value + list-append contract. Concrete stores
    (in-memory, Redis, a database-backed store the host application
    provides) implement this; higher-level memory concepts are built on
    top of it rather than each having their own storage backend.
    """

    async def get(self, namespace: str, key: str) -> Any: ...

    async def set(self, namespace: str, key: str, value: Any) -> None: ...

    async def delete(self, namespace: str, key: str) -> None: ...

    async def append(self, namespace: str, key: str, value: Any, *, max_length: int | None = None) -> None: ...

    async def get_list(self, namespace: str, key: str) -> list[Any]: ...


class ConversationMemory:
    """Ergonomic conversation-history and long-term-summary memory.

    Wraps a :class:`MemoryStore` to provide the operations agents and
    supervisors actually need: append a turn, fetch recent history,
    fetch/append durable summary context. Backed by any ``MemoryStore``
    implementation, so swapping the backend (in-memory -> Redis ->
    database) doesn't change any agent code.
    """

    _HISTORY_NS = "conversation:history"
    _SUMMARY_NS = "conversation:summary"

    def __init__(self, store: MemoryStore, *, max_history: int = 20, max_summary_events: int = 30) -> None:
        self.store = store
        self.max_history = max_history
        self.max_summary_events = max_summary_events

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a single conversation turn, trimmed to a bounded length."""
        trimmed = content if len(content) <= 2000 else content[:2000] + "..."
        await self.store.append(
            self._HISTORY_NS,
            session_id,
            {"role": role, "content": trimmed},
            max_length=self.max_history,
        )

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        return await self.store.get_list(self._HISTORY_NS, session_id)

    async def add_summary_event(self, session_id: str, event: str) -> None:
        """Record a durable fact/event that should survive history trimming."""
        await self.store.append(
            self._SUMMARY_NS,
            session_id,
            event,
            max_length=self.max_summary_events,
        )

    async def get_summary_context(self, session_id: str) -> str:
        events = await self.store.get_list(self._SUMMARY_NS, session_id)
        return "\n".join(events)

    async def clear(self, session_id: str) -> None:
        await self.store.delete(self._HISTORY_NS, session_id)
        await self.store.delete(self._SUMMARY_NS, session_id)


class AgentStateMemory:
    """Namespaced key/value state for a single agent.

    Lets an agent persist small bits of state between invocations
    (e.g. "last_search_query") without colliding with other agents'
    state, since the namespace is derived from the agent's name.
    """

    def __init__(self, store: MemoryStore, agent_name: str) -> None:
        self.store = store
        self._namespace = f"agent_state:{agent_name}"

    async def get(self, key: str, default: Any = None) -> Any:
        value = await self.store.get(self._namespace, key)
        return default if value is None else value

    async def set(self, key: str, value: Any) -> None:
        await self.store.set(self._namespace, key, value)

    async def delete(self, key: str) -> None:
        await self.store.delete(self._namespace, key)
