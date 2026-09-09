"""Request/response schemas for the framework's HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunTaskRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Natural-language description of what to accomplish")
    session_id: str | None = None
    user_id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStepResponse(BaseModel):
    agent_name: str
    success: bool
    output: Any = None
    error: str | None = None


class RunTaskResponse(BaseModel):
    task_id: str
    status: str
    output: Any = None
    error: str | None = None
    iterations: int
    agent_calls: int
    steps: list[AgentStepResponse]


class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: list[str]
    tools: list[str]
    permission: str


class ToolInfo(BaseModel):
    name: str
    description: str
    permission: str
    requires_confirmation: bool
