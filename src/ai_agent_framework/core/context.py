"""
Execution context.

The :class:`ExecutionContext` is threaded through every layer of a task's
execution (supervisor -> agent -> tool -> model). It carries exactly what
execution needs and nothing that belongs to a specific business domain:
identity of the caller, session/task identifiers, conversation state,
available tools/permissions, and a place to record execution history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_agent_framework.core.events import EventBus
    from ai_agent_framework.memory.base import MemoryStore
    from ai_agent_framework.security.permissions import PermissionSet
    from ai_agent_framework.tools.registry import ToolRegistry


@dataclass
class ExecutionContext:
    """Carries everything a task execution needs, and nothing domain-specific.

    Attributes:
        user_id: Identifier of the human or system on whose behalf the
            task is running. Opaque to the framework.
        session_id: Identifier grouping related tasks/conversation turns.
        task_id: The id of the task currently executing in this context.
        conversation_id: Optional identifier for the conversation this
            task belongs to, if different from the session.
        current_agent: Name of the agent currently executing, if any.
        tool_registry: The tool registry available to this execution.
        permissions: The permission set granted to this execution.
        memory: The memory store backing this execution.
        event_bus: The event bus events are published to.
        metadata: Free-form, host-supplied key/value data. The framework
            never reads business meaning out of this dict; it exists so
            a host application can thread arbitrary extra data through
            without the framework needing to know about it.
        history: Append-only record of execution steps taken so far,
            populated by the orchestration layer for observability and
            for supervisors to reason over past steps.
    """

    user_id: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str | None = None
    conversation_id: str | None = None
    current_agent: str | None = None

    tool_registry: "ToolRegistry | None" = None
    permissions: "PermissionSet | None" = None
    memory: "MemoryStore | None" = None
    event_bus: "EventBus | None" = None

    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, entry: dict[str, Any]) -> None:
        """Append an entry to the execution history."""
        self.history.append(entry)

    def child(self, **overrides: Any) -> "ExecutionContext":
        """Create a copy of this context with the given fields overridden.

        Used when dispatching to a sub-agent or nested execution that
        should share most context but have e.g. a different
        ``current_agent``.
        """
        import dataclasses

        return dataclasses.replace(self, **overrides)
