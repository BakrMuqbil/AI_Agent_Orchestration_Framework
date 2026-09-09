"""Tests for orchestration execution strategies (sequential/parallel)."""

from __future__ import annotations

import pytest

from ai_agent_framework.orchestration.strategies import run_parallel, run_sequential
from tests.conftest import EchoAgent, FailingAgent


class SecondEchoAgent(EchoAgent):
    name = "second_echo_agent"


class TestRunSequential:
    @pytest.mark.asyncio
    async def test_runs_all_agents_in_order(self, admin_context):
        agents = [EchoAgent(), SecondEchoAgent()]
        results = await run_sequential(agents, "hello", admin_context)
        assert [r.agent_name for r in results] == ["echo_agent", "second_echo_agent"]
        assert all(r.result.success for r in results)

    @pytest.mark.asyncio
    async def test_stops_early_on_failure(self, admin_context):
        agents = [FailingAgent(), EchoAgent()]
        results = await run_sequential(agents, "hello", admin_context)
        assert len(results) == 1
        assert results[0].agent_name == "failing_agent"
        assert results[0].result.success is False

    @pytest.mark.asyncio
    async def test_records_history(self, admin_context):
        agents = [EchoAgent()]
        await run_sequential(agents, "hello", admin_context)
        assert len(admin_context.history) == 1


class TestRunParallel:
    @pytest.mark.asyncio
    async def test_runs_all_agents_concurrently(self, admin_context):
        agents = [EchoAgent(), SecondEchoAgent()]
        results = await run_parallel(agents, "hello", admin_context)
        names = {r.agent_name for r in results}
        assert names == {"echo_agent", "second_echo_agent"}

    @pytest.mark.asyncio
    async def test_one_failure_does_not_stop_others(self, admin_context):
        agents = [FailingAgent(), EchoAgent()]
        results = await run_parallel(agents, "hello", admin_context)
        assert len(results) == 2
        by_name = {r.agent_name: r.result for r in results}
        assert by_name["failing_agent"].success is False
        assert by_name["echo_agent"].success is True

    @pytest.mark.asyncio
    async def test_records_history_for_all(self, admin_context):
        agents = [EchoAgent(), SecondEchoAgent()]
        await run_parallel(agents, "hello", admin_context)
        assert len(admin_context.history) == 2
