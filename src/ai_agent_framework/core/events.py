"""
Event system for observability.

The framework never logs to a specific provider (stdout, Sentry, Datadog,
etc.) directly. Instead it publishes structured :class:`Event` objects to
an :class:`EventBus`, and the host application subscribes handlers to
whichever events it cares about. A simple stdout-logging handler is
provided as a convenience, but it is opt-in.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Union

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Well-known lifecycle events emitted by the framework."""

    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"

    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    AGENT_FAILED = "agent_failed"
    AGENT_SELECTED = "agent_selected"

    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"

    MODEL_CALLED = "model_called"
    MODEL_FINISHED = "model_finished"
    MODEL_FAILED = "model_failed"

    PLAN_CREATED = "plan_created"
    RE_PLAN = "re_plan"

    ERROR = "error"


class Event(BaseModel):
    """A single observability event."""

    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


EventHandler = Callable[[Event], Union[None, Awaitable[None]]]


class EventBus:
    """A minimal, provider-independent pub/sub bus for framework events.

    Handlers may be sync or async callables. Handler exceptions are
    swallowed (never allowed to break task execution) but are re-published
    as an :data:`EventType.ERROR` event so observability of a broken
    handler is still possible.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}

    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> None:
        """Subscribe a handler. ``event_type=None`` subscribes to all events."""
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType | None, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        handlers = list(self._handlers.get(event.type, [])) + list(self._handlers.get(None, []))
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - a broken handler must never break execution
                continue

    def publish_sync(self, event: Event) -> None:
        """Synchronous convenience wrapper for non-async call sites."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.publish(event))
            return
        loop.create_task(self.publish(event))


def stdout_logging_handler(event: Event) -> None:
    """An opt-in convenience handler that prints events to stdout.

    Host applications are free to ignore this and register their own
    handlers that forward to whatever logging/observability stack they use.
    """
    parts = [f"[{event.timestamp.isoformat()}]", event.type.value]
    if event.task_id:
        parts.append(f"task={event.task_id}")
    if event.agent_name:
        parts.append(f"agent={event.agent_name}")
    if event.tool_name:
        parts.append(f"tool={event.tool_name}")
    print(" ".join(parts))
