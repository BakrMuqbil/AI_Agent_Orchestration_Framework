"""Tests for the FastAPI application: /tasks, /agents, /tools, /status."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ai_agent_framework.api.app import create_app
from ai_agent_framework.orchestration.engine import AgentEngine
from ai_agent_framework.security.permissions import Permission
from ai_agent_framework.tools.decorator import tool
from tests.conftest import EchoAgent, FailingAgent


@pytest.fixture
def engine() -> AgentEngine:
    e = AgentEngine(model=None)
    e.register_agent(EchoAgent())
    e.register_agent(FailingAgent())

    @tool(name="api_test_tool", description="A tool for API tests", permission=Permission.READ)
    async def api_test_tool() -> dict:
        return {"ok": True}

    e.register_tool(api_test_tool)
    return e


@pytest.fixture
async def client(engine: AgentEngine):
    app = create_app(engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRunTaskEndpoint:
    @pytest.mark.asyncio
    async def test_run_task_returns_completed_result(self, client):
        response = await client.post("/api/v1/tasks", json={"goal": "echo_agent please respond"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["agent_calls"] == 1

    @pytest.mark.asyncio
    async def test_run_task_requires_goal(self, client):
        response = await client.post("/api/v1/tasks", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_task_rejects_empty_goal(self, client):
        response = await client.post("/api/v1/tasks", json={"goal": ""})
        assert response.status_code == 422


class TestListAgentsEndpoint:
    @pytest.mark.asyncio
    async def test_list_agents_returns_registered_agents(self, client):
        response = await client.get("/api/v1/agents")
        assert response.status_code == 200
        names = {a["name"] for a in response.json()}
        assert names == {"echo_agent", "failing_agent"}


class TestListToolsEndpoint:
    @pytest.mark.asyncio
    async def test_list_tools_returns_registered_tools(self, client):
        response = await client.get("/api/v1/tools")
        assert response.status_code == 200
        names = {t["name"] for t in response.json()}
        assert "api_test_tool" in names


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_reports_counts(self, client):
        response = await client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["agents_registered"] == 2
        assert data["tools_registered"] == 1
