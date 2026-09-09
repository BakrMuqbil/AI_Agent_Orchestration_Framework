"""Tests for the Agent interface and AgentRegistry."""

from __future__ import annotations

import pytest

from ai_agent_framework.agents.base import Agent, AgentCapability, AgentResult
from ai_agent_framework.agents.registry import AgentRegistry
from ai_agent_framework.core.errors import AgentNotFoundError
from tests.conftest import EchoAgent, FailingAgent


class UnnamedAgent(Agent):
    async def run(self, task, context):  # pragma: no cover - never reached
        return AgentResult.ok("x")


class TestAgentBase:
    def test_agent_requires_name(self):
        with pytest.raises(ValueError):
            UnnamedAgent()

    @pytest.mark.asyncio
    async def test_agent_run_returns_result(self, admin_context):
        agent = EchoAgent()
        result = await agent.run("hello", admin_context)
        assert result.success is True
        assert result.output == "echo: hello"

    def test_describe_includes_metadata(self):
        agent = EchoAgent()
        description = agent.describe()
        assert description["name"] == "echo_agent"
        assert "description" in description
        assert "capabilities" in description
        assert "tools" in description

    def test_agent_result_ok_and_fail(self):
        ok = AgentResult.ok("value", extra=1)
        assert ok.success is True
        assert ok.output == "value"
        assert ok.metadata == {"extra": 1}

        fail = AgentResult.fail("bad thing")
        assert fail.success is False
        assert fail.error == "bad thing"


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = EchoAgent()
        registry.register(agent)
        assert registry.get("echo_agent") is agent

    def test_get_missing_agent_raises(self):
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            registry.get("nope")

    def test_describe_all(self):
        registry = AgentRegistry([EchoAgent(), FailingAgent()])
        descriptions = registry.describe_all()
        names = {d["name"] for d in descriptions}
        assert names == {"echo_agent", "failing_agent"}

    def test_unregister(self):
        registry = AgentRegistry([EchoAgent()])
        registry.unregister("echo_agent")
        assert not registry.has("echo_agent")

    def test_len_and_contains(self):
        registry = AgentRegistry([EchoAgent()])
        assert len(registry) == 1
        assert "echo_agent" in registry
