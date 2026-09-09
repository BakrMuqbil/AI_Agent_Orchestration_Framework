"""
Task representation.

A ``Task`` is the unit of work the supervisor plans over and dispatches
to agents. It is intentionally domain-agnostic: a task is just "do this,
described in natural language or structured input, with this goal".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle states of a task."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_ON_AGENT = "waiting_on_agent"
    RE_PLANNING = "re_planning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """A unit of work submitted to the framework for execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}


class AgentStepResult(BaseModel):
    """The record of a single agent invocation within a task's execution."""

    agent_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    success: bool = True
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class TaskResult(BaseModel):
    """The final outcome of executing a :class:`Task`."""

    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    steps: list[AgentStepResult] = Field(default_factory=list)
    iterations: int = 0
    agent_calls: int = 0
    tool_calls: int = 0

    model_config = {"arbitrary_types_allowed": True}
