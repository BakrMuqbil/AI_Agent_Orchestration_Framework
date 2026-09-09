"""
API routes.

A general, framework-level API -- not tied to any specific host
application's domain. Routes operate purely in terms of framework
concepts (tasks, agents, tools), matching the "clean, general API" goal
rather than any fixed literal path list.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ai_agent_framework.api.schemas import (
    AgentInfo,
    AgentStepResponse,
    RunTaskRequest,
    RunTaskResponse,
    ToolInfo,
)
from ai_agent_framework.core.errors import (
    ConfirmationRequiredError,
    PermissionDeniedError,
)
from ai_agent_framework.orchestration.engine import AgentEngine

router = APIRouter(tags=["orchestration"])


def get_engine() -> AgentEngine:
    """Overridden by the application factory with the configured engine.

    Declared here as a placeholder dependency so routes can be defined
    against a stable signature; :func:`ai_agent_framework.api.app.create_app`
    overrides this dependency with the actual engine instance for the app.
    """
    raise RuntimeError("AgentEngine dependency was not configured on this app")


@router.post("/tasks", response_model=RunTaskResponse)
async def run_task(request: RunTaskRequest, engine: AgentEngine = Depends(get_engine)) -> RunTaskResponse:
    """Run a task through the supervisor and return the outcome."""
    try:
        result = await engine.run(
            request.goal,
            session_id=request.session_id,
            user_id=request.user_id,
            input=request.input,
            metadata=request.metadata,
        )
    except (PermissionDeniedError, ConfirmationRequiredError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return RunTaskResponse(
        task_id=result.task_id,
        status=result.status.value,
        output=result.output,
        error=result.error,
        iterations=result.iterations,
        agent_calls=result.agent_calls,
        steps=[
            AgentStepResponse(
                agent_name=s.agent_name, success=s.success, output=s.output, error=s.error
            )
            for s in result.steps
        ],
    )


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents(engine: AgentEngine = Depends(get_engine)) -> list[AgentInfo]:
    """List every agent currently registered with the engine."""
    return [AgentInfo(**a) for a in engine.agents.describe_all()]


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(engine: AgentEngine = Depends(get_engine)) -> list[ToolInfo]:
    """List every globally-registered tool."""
    return [
        ToolInfo(
            name=t.name,
            description=t.description,
            permission=str(t.permission),
            requires_confirmation=t.requires_confirmation,
        )
        for t in engine.tools.list()
    ]


@router.get("/status")
async def status(engine: AgentEngine = Depends(get_engine)) -> dict:
    """Basic liveness/status endpoint."""
    return {
        "status": "online",
        "agents_registered": len(engine.agents),
        "tools_registered": len(engine.tools),
    }
