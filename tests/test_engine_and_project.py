"""Tests for AgentEngine (top-level facade) and ProjectRegistration."""

from __future__ import annotations

import pytest

from ai_agent_framework.core.task import TaskStatus
from ai_agent_framework.orchestration.engine import AgentEngine
from ai_agent_framework.orchestration.project import ProjectRegistration
from ai_agent_framework.security.permissions import Permission, PermissionSet
from ai_agent_framework.tools.decorator import tool
from tests.conftest import EchoAgent, FakeModelProvider


class TestAgentEngine:
    @pytest.mark.asyncio
    async def test_register_agent_and_run_without_model(self):
        engine = AgentEngine(model=None)
        engine.register_agent(EchoAgent())
        result = await engine.run("echo_agent please respond")
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_register_agent_returns_self_for_chaining(self):
        engine = AgentEngine(model=None)
        returned = engine.register_agent(EchoAgent())
        assert returned is engine

    @pytest.mark.asyncio
    async def test_run_creates_isolated_sessions(self):
        engine = AgentEngine(model=None)
        engine.register_agent(EchoAgent())
        result_a = await engine.run("echo_agent a", session_id="session-a")
        result_b = await engine.run("echo_agent b", session_id="session-b")
        assert result_a.task_id != result_b.task_id

    @pytest.mark.asyncio
    async def test_engine_with_model_uses_model_planner(self):
        from ai_agent_framework.orchestration.supervisor import ModelPlanner, PlanDecision

        model = FakeModelProvider()
        model.queue_structured(PlanDecision(action="finish", final_output="done via model"))
        engine = AgentEngine(model=model)
        assert isinstance(engine.planner, ModelPlanner)
        result = await engine.run("anything")
        assert result.output == "done via model"

    @pytest.mark.asyncio
    async def test_default_permissions_are_admin(self):
        engine = AgentEngine(model=None)
        assert engine.permissions.allows(Permission.ADMIN)

    @pytest.mark.asyncio
    async def test_custom_permissions_are_used_in_context(self):
        received_permissions = []

        class PermissionCheckingAgent(EchoAgent):
            name = "permission_checking_agent"

            async def run(self, task, context):
                received_permissions.append(context.permissions)
                return await super().run(task, context)

        engine = AgentEngine(model=None, permissions=PermissionSet.read_only())
        engine.register_agent(PermissionCheckingAgent())
        await engine.run("permission_checking_agent test")
        assert received_permissions[0].level == Permission.READ

    @pytest.mark.asyncio
    async def test_register_tool_makes_it_globally_available(self):
        @tool(name="ping", description="ping", permission=Permission.READ)
        async def ping() -> dict:
            return {"pong": True}

        engine = AgentEngine(model=None)
        engine.register_tool(ping)
        assert engine.tools.has("ping")


class TestProjectRegistration:
    @pytest.mark.asyncio
    async def test_apply_registers_agents_and_tools(self):
        @tool(name="dalalti_ping", description="ping", permission=Permission.READ)
        async def dalalti_ping() -> dict:
            return {"pong": True}

        class DalaltiAgent(EchoAgent):
            name = "dalalti.echo_agent"

        project = ProjectRegistration(
            name="dalalti",
            agents=[DalaltiAgent()],
            tools=[dalalti_ping],
        )
        engine = AgentEngine(model=None)
        project.apply(engine)

        assert engine.agents.has("dalalti.echo_agent")
        assert engine.tools.has("dalalti_ping")

    @pytest.mark.asyncio
    async def test_multiple_projects_coexist_with_namespaced_names(self):
        class ProjectAAgent(EchoAgent):
            name = "project_a.agent"

        class ProjectBAgent(EchoAgent):
            name = "project_b.agent"

        project_a = ProjectRegistration(name="project_a", agents=[ProjectAAgent()])
        project_b = ProjectRegistration(name="project_b", agents=[ProjectBAgent()])

        engine = AgentEngine(model=None)
        project_a.apply(engine)
        project_b.apply(engine)

        assert engine.agents.has("project_a.agent")
        assert engine.agents.has("project_b.agent")
        assert len(engine.agents) == 2

    def test_apply_returns_engine_for_chaining(self):
        project = ProjectRegistration(name="empty")
        engine = AgentEngine(model=None)
        assert project.apply(engine) is engine
