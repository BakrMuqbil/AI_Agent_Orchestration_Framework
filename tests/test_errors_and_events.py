"""Tests for the error hierarchy and the EventBus."""

from __future__ import annotations

import pytest

from ai_agent_framework.core.errors import (
    AgentNotFoundError,
    ConfirmationRequiredError,
    FrameworkError,
    LoopLimitExceededError,
    PermissionDeniedError,
    ToolNotFoundError,
)
from ai_agent_framework.core.events import Event, EventBus, EventType


class TestErrorHierarchy:
    def test_all_errors_inherit_framework_error(self):
        assert issubclass(AgentNotFoundError, FrameworkError)
        assert issubclass(ToolNotFoundError, FrameworkError)
        assert issubclass(PermissionDeniedError, FrameworkError)
        assert issubclass(LoopLimitExceededError, FrameworkError)
        assert issubclass(ConfirmationRequiredError, FrameworkError)

    def test_agent_not_found_carries_agent_name(self):
        err = AgentNotFoundError("ghost")
        assert err.agent_name == "ghost"
        assert "ghost" in str(err)

    def test_tool_not_found_carries_tool_name(self):
        err = ToolNotFoundError("missing_tool")
        assert err.tool_name == "missing_tool"

    def test_confirmation_required_carries_arguments(self):
        err = ConfirmationRequiredError("delete_thing", {"id": "123"})
        assert err.arguments == {"id": "123"}

    def test_framework_error_can_be_caught_generically(self):
        with pytest.raises(FrameworkError):
            raise ToolNotFoundError("x")


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_specific_event_type(self):
        received = []
        bus = EventBus()
        bus.subscribe(EventType.TASK_STARTED, lambda e: received.append(e))
        await bus.publish(Event(type=EventType.TASK_STARTED))
        await bus.publish(Event(type=EventType.TASK_FINISHED))
        assert len(received) == 1
        assert received[0].type == EventType.TASK_STARTED

    @pytest.mark.asyncio
    async def test_subscribe_all_events(self):
        received = []
        bus = EventBus()
        bus.subscribe(None, lambda e: received.append(e))
        await bus.publish(Event(type=EventType.TASK_STARTED))
        await bus.publish(Event(type=EventType.TASK_FINISHED))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = EventBus()
        bus.subscribe(None, handler)
        await bus.publish(Event(type=EventType.AGENT_STARTED))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_receiving_events(self):
        received = []

        def handler(event: Event) -> None:
            received.append(event)

        bus = EventBus()
        bus.subscribe(None, handler)
        bus.unsubscribe(None, handler)
        await bus.publish(Event(type=EventType.TASK_STARTED))
        assert received == []

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self):
        def broken(event: Event) -> None:
            raise RuntimeError("boom")

        bus = EventBus()
        bus.subscribe(None, broken)
        # Should not raise.
        await bus.publish(Event(type=EventType.TASK_STARTED))
