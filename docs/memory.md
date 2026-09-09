# Memory

## The four kinds of state

| Kind | What it is | Where it lives |
|---|---|---|
| Conversation state | Message history for a session | ConversationMemory |
| Task state | Working state of one task's execution | ExecutionContext.history / TaskResult (transient) |
| Agent state | Small key/value state an agent persists between calls | AgentStateMemory (namespaced per agent) |
| Long-term memory | Durable facts/summaries that outlive a session | ConversationMemory summary methods |

None of these are tied to any specific host application -- they're general
concepts every agentic system needs, backed by a small, swappable storage
interface.

## MemoryStore: the storage interface

```python
class MemoryStore(Protocol):
    async def get(self, namespace: str, key: str) -> Any: ...
    async def set(self, namespace: str, key: str, value: Any) -> None: ...
    async def delete(self, namespace: str, key: str) -> None: ...
    async def append(self, namespace: str, key: str, value: Any, *, max_length: int | None = None) -> None: ...
    async def get_list(self, namespace: str, key: str) -> list[Any]: ...
```

InMemoryStore (process-local, dict-backed) is the default -- fine for
development, tests, and single-process deployments. For persistence across
restarts or multiple processes, implement MemoryStore against Redis, a
database, etc. The interface is deliberately small specifically to make
that easy.

## ConversationMemory: the ergonomic layer

```python
from ai_agent_framework import ConversationMemory
from ai_agent_framework.memory.in_memory import InMemoryStore

memory = ConversationMemory(InMemoryStore(), max_history=20, max_summary_events=30)

await memory.add_message(session_id, "user", "What's my order status?")
await memory.add_message(session_id, "assistant", "Let me check...")

history = await memory.get_history(session_id)

await memory.add_summary_event(session_id, "Customer asked about order 123")
context = await memory.get_summary_context(session_id)
```

- add_message trims any single message to 2000 characters and keeps only
  the most recent max_history messages.
- add_summary_event / get_summary_context are the durable side: facts
  worth keeping even after the raw history around them has been trimmed
  away. Nothing writes to this automatically -- a host application (or an
  agent) decides what's worth summarizing and calls add_summary_event
  explicitly.
- clear(session_id) wipes both history and summary for a session.

## AgentStateMemory: per-agent key/value state

```python
from ai_agent_framework.memory.base import AgentStateMemory

state = AgentStateMemory(store, agent_name="research_agent")
await state.set("last_query", "solar panel pricing")
last_query = await state.get("last_query", default=None)
```

Namespaced by agent name, so two agents using the same key never collide.

## An engine's default memory

AgentEngine builds a ConversationMemory over an InMemoryStore by default
and attaches it to every ExecutionContext it creates (context.memory), so
any agent can read/write conversation state without wiring memory
manually. Supply memory_store= to AgentEngine(...) to use a different
backend.
