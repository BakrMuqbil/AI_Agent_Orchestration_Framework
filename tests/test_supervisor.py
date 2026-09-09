"""
Tests for the Supervisor's plan -> select -> execute -> observe -> evaluate
cycle: multi-agent execution, re-planning, loop/safety limits, and error
handling. Uses RuleBasedPlanner and FakeModelProvider-backed ModelPlanner
so no real LLM or network call is ever made.
"""

from __future__ import annotations

import pytest

from ai_agent_framework.agents.registry import AgentRegistry
from ai_agent_framework.core.context import ExecutionContext
from ai_agent_framework.core.task import Task, TaskStatus
from ai_agent_framework.orchestration.supervisor import (
    PlanDecision,
    RuleBasedPlanner,
    Supervisor,
    SupervisorConfig,
)
from tests.conftest import EchoAgent, FailingAgent, FakeModelProvider, RaisingAgent


@pytest.fixture
def agent_registry() -> AgentRegistry:
    return AgentRegistry([EchoAgent(), FailingAgent(), RaisingAgent()])


class TestRuleBasedPlanner:
    @pytest.mark.asyncio
    async def test_selects_best_matching_agent(self, agent_registry, admin_context):
        task = Task(goal="please echo_agent this")
        planner = RuleBasedPlanner()
        decision = await planner.decide(task, agent_registry, admin_context)
        assert decision.action == "call_agent"
        assert decision.agent_name == "echo_agent"

    @pytest.mark.asyncio
    async def test_finishes_after_one_step(self, agent_registry, admin_context):
        task = Task(goal="echo_agent test")
        planner = RuleBasedPlanner()
        admin_context.record({"agent_name": "echo_agent", "output": "echo: test", "success": True})
        decision = await planner.decide(task, agent_registry, admin_context)
        assert decision.action == "finish"

    @pytest.mark.asyncio
    async def test_no_agents_fails(self, admin_context):
        planner = RuleBasedPlanner()
        decision = await planner.decide(Task(goal="anything"), AgentRegistry(), admin_context)
        assert decision.action == "fail"


class TestSupervisorEndToEnd:
    @pytest.mark.asyncio
    async def test_successful_single_agent_run(self, agent_registry, admin_context):
        supervisor = Supervisor(agents=agent_registry, planner=RuleBasedPlanner())
        result = await supervisor.run(Task(goal="echo_agent please"), admin_context)
        assert result.status == TaskStatus.COMPLETED
        assert result.agent_calls == 1
        assert result.steps[0].agent_name == "echo_agent"
        assert result.steps[0].success is True

    @pytest.mark.asyncio
    async def test_multi_step_model_driven_run(self, admin_context):
        registry = AgentRegistry([EchoAgent()])
        model = FakeModelProvider()
        model.queue_structured(
            PlanDecision(action="call_agent", agent_name="echo_agent", agent_task="step one")
        )
        model.queue_structured(
            PlanDecision(action="call_agent", agent_name="echo_agent", agent_task="step two")
        )
        model.queue_structured(PlanDecision(action="finish", final_output="all done"))

        from ai_agent_framework.orchestration.supervisor import ModelPlanner

        supervisor = Supervisor(agents=registry, planner=ModelPlanner(model))
        result = await supervisor.run(Task(goal="multi-step goal"), admin_context)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "all done"
        assert result.agent_calls == 2
        assert [s.agent_name for s in result.steps] == ["echo_agent", "echo_agent"]

    @pytest.mark.asyncio
    async def test_agent_not_found_is_recorded_not_raised(self, admin_context):
        model = FakeModelProvider()
        model.queue_structured(PlanDecision(action="call_agent", agent_name="ghost_agent"))
        model.queue_structured(PlanDecision(action="finish", final_output="recovered"))

        from ai_agent_framework.orchestration.supervisor import ModelPlanner

        supervisor = Supervisor(agents=AgentRegistry([EchoAgent()]), planner=ModelPlanner(model))
        result = await supervisor.run(Task(goal="call a missing agent"), admin_context)

        assert result.steps[0].agent_name == "ghost_agent"
        assert result.steps[0].success is False
        # Supervisor keeps going after a bad dispatch rather than crashing the whole run.
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_raising_agent_is_normalized_into_failed_step(self, admin_context):
        registry = AgentRegistry([RaisingAgent()])
        supervisor = Supervisor(agents=registry, planner=RuleBasedPlanner())
        result = await supervisor.run(Task(goal="raising_agent test"), admin_context)
        assert result.steps[0].success is False
        assert "boom" in result.steps[0].error

    @pytest.mark.asyncio
    async def test_planner_fail_decision_fails_task(self, admin_context):
        supervisor = Supervisor(agents=AgentRegistry(), planner=RuleBasedPlanner())
        result = await supervisor.run(Task(goal="nothing can handle this"), admin_context)
        assert result.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_invalid_decision_shape_fails_gracefully(self, admin_context):
        model = FakeModelProvider()
        model.queue_structured(PlanDecision(action="call_agent", agent_name=None))

        from ai_agent_framework.orchestration.supervisor import ModelPlanner

        supervisor = Supervisor(agents=AgentRegistry([EchoAgent()]), planner=ModelPlanner(model))
        result = await supervisor.run(Task(goal="bad decision"), admin_context)
        assert result.status == TaskStatus.FAILED


class TestSupervisorSafetyLimits:
    @pytest.mark.asyncio
    async def test_max_iterations_stops_infinite_loop(self, admin_context):
        class NeverFinishPlanner:
            async def decide(self, task, agents, context):
                return PlanDecision(action="call_agent", agent_name="echo_agent", agent_task="loop forever")

        registry = AgentRegistry([EchoAgent()])
        config = SupervisorConfig(max_iterations=3, max_agent_calls=100)
        supervisor = Supervisor(agents=registry, planner=NeverFinishPlanner(), config=config)
        result = await supervisor.run(Task(goal="loop"), admin_context)

        assert result.status == TaskStatus.FAILED
        assert "max_iterations" in result.error
        assert result.iterations <= 3

    @pytest.mark.asyncio
    async def test_max_agent_calls_stops_excessive_dispatch(self, admin_context):
        class AlwaysCallPlanner:
            async def decide(self, task, agents, context):
                return PlanDecision(action="call_agent", agent_name="echo_agent", agent_task="again")

        registry = AgentRegistry([EchoAgent()])
        config = SupervisorConfig(max_iterations=100, max_agent_calls=2)
        supervisor = Supervisor(agents=registry, planner=AlwaysCallPlanner(), config=config)
        result = await supervisor.run(Task(goal="loop"), admin_context)

        assert result.status == TaskStatus.FAILED
        assert "max_agent_calls" in result.error
        assert result.agent_calls <= 2

    @pytest.mark.asyncio
    async def test_default_config_has_positive_finite_limits(self):
        config = SupervisorConfig()
        assert config.max_iterations > 0
        assert config.max_agent_calls > 0
        assert config.max_tool_calls > 0


class TestSupervisorEvents:
    @pytest.mark.asyncio
    async def test_task_started_and_finished_events_are_published(self, agent_registry, admin_context):
        from ai_agent_framework.core.events import Event, EventBus, EventType

        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus = EventBus()
        bus.subscribe(None, handler)
        supervisor = Supervisor(agents=agent_registry, planner=RuleBasedPlanner(), event_bus=bus)
        await supervisor.run(Task(goal="echo_agent please"), admin_context)

        types = [e.type for e in received]
        assert EventType.TASK_STARTED in types
        assert EventType.TASK_FINISHED in types
        assert EventType.AGENT_STARTED in types
        assert EventType.AGENT_FINISHED in types

    @pytest.mark.asyncio
    async def test_broken_handler_does_not_break_execution(self, agent_registry, admin_context):
        from ai_agent_framework.core.events import EventBus

        def broken_handler(event):
            raise RuntimeError("handler is broken")

        bus = EventBus()
        bus.subscribe(None, broken_handler)
        supervisor = Supervisor(agents=agent_registry, planner=RuleBasedPlanner(), event_bus=bus)
        result = await supervisor.run(Task(goal="echo_agent please"), admin_context)
        assert result.status == TaskStatus.COMPLETED
