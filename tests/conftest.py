"""Shared pytest fixtures and test doubles for the framework's test suite."""

from __future__ import annotations

from typing import Any

import pytest

from ai_agent_framework.agents.base import Agent, AgentResult
from ai_agent_framework.core.context import ExecutionContext
from ai_agent_framework.memory.base import ConversationMemory
from ai_agent_framework.memory.in_memory import InMemoryStore
from ai_agent_framework.models.base import ModelProvider, ModelRequest, ModelResponse
from ai_agent_framework.security.permissions import Permission, PermissionSet
from ai_agent_framework.tools.registry import ToolRegistry


class FakeModelProvider(ModelProvider):
    """A model provider double that returns pre-programmed responses.

    Configure with a list of ``ModelResponse`` (for ``generate``) and/or
    a queue of structured-output objects (for ``generate_structured``),
    consumed in order. Never makes a network call.
    """

    provider_name = "fake"

    def __init__(self, *, model: str = "fake-model", responses: list[ModelResponse] | None = None) -> None:
        super().__init__(model=model)
        self._responses = list(responses or [])
        self._structured_queue: list[Any] = []
        self.calls: list[ModelRequest] = []

    def queue_response(self, response: ModelResponse) -> None:
        self._responses.append(response)

    def queue_structured(self, value: Any) -> None:
        self._structured_queue.append(value)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if not self._responses:
            return ModelResponse(content="", provider=self.provider_name, model=self.model)
        return self._responses.pop(0)

    async def generate_structured(self, request: ModelRequest, schema):  # type: ignore[override]
        self.calls.append(request)
        if self._structured_queue:
            value = self._structured_queue.pop(0)
            if isinstance(value, schema):
                return value
            return schema(**value)
        return await super().generate_structured(request, schema)


class EchoAgent(Agent):
    """A trivial agent used across tests: returns whatever task text it receives."""

    name = "echo_agent"
    description = "Echoes the task text back as output"
    permission = Permission.READ

    async def run(self, task: str, context: ExecutionContext) -> AgentResult:
        return AgentResult.ok(f"echo: {task}")


class FailingAgent(Agent):
    """An agent that always fails, for testing error handling."""

    name = "failing_agent"
    description = "Always fails"

    async def run(self, task: str, context: ExecutionContext) -> AgentResult:
        return AgentResult.fail("simulated failure")


class RaisingAgent(Agent):
    """An agent whose run() raises, for testing exception normalization."""

    name = "raising_agent"
    description = "Always raises"

    async def run(self, task: str, context: ExecutionContext) -> AgentResult:
        raise RuntimeError("boom")


@pytest.fixture
def memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def conversation_memory(memory_store: InMemoryStore) -> ConversationMemory:
    return ConversationMemory(memory_store)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def admin_context(tool_registry: ToolRegistry, conversation_memory: ConversationMemory) -> ExecutionContext:
    return ExecutionContext(
        user_id="test_user",
        session_id="test_session",
        tool_registry=tool_registry,
        permissions=PermissionSet.admin(),
        memory=conversation_memory,
    )


@pytest.fixture
def read_only_context(tool_registry: ToolRegistry, conversation_memory: ConversationMemory) -> ExecutionContext:
    return ExecutionContext(
        user_id="test_user",
        session_id="test_session",
        tool_registry=tool_registry,
        permissions=PermissionSet.read_only(),
        memory=conversation_memory,
    )
