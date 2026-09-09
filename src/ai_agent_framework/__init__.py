"""
AI Agent Orchestration Framework
=================================

A provider-independent, domain-independent framework for building,
registering, and orchestrating AI agents with tools, memory, and RAG.

Typical usage::

    from ai_agent_framework import Agent, AgentEngine, Supervisor, tool

    @tool(name="get_time", description="Get the current UTC time")
    async def get_time() -> dict:
        return {"time": "..."}

    class ResearchAgent(Agent):
        name = "research_agent"
        description = "Looks things up"
        tools = [get_time]

        async def run(self, task, context):
            ...

    engine = AgentEngine()
    engine.register_agent(ResearchAgent())
    result = await engine.run("What time is it?")

This top-level package intentionally re-exports only the stable, public
surface of the framework. Internal modules under each subpackage may
change without notice; anything exported here is considered the SDK.
"""

from ai_agent_framework.core.context import ExecutionContext
from ai_agent_framework.core.errors import (
    AgentError,
    AgentNotFoundError,
    ConfigurationError,
    FrameworkError,
    LoopLimitExceededError,
    ModelProviderError,
    PermissionDeniedError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from ai_agent_framework.core.events import Event, EventBus, EventType
from ai_agent_framework.core.task import Task, TaskResult, TaskStatus
from ai_agent_framework.agents.base import Agent, AgentCapability, AgentResult
from ai_agent_framework.agents.registry import AgentRegistry
from ai_agent_framework.tools.base import Tool, ToolResult
from ai_agent_framework.tools.decorator import tool
from ai_agent_framework.tools.registry import ToolRegistry
from ai_agent_framework.models.base import ModelProvider, ModelRequest, ModelResponse
from ai_agent_framework.memory.base import ConversationMemory, MemoryStore
from ai_agent_framework.security.permissions import Permission, PermissionSet
from ai_agent_framework.orchestration.supervisor import Supervisor
from ai_agent_framework.orchestration.engine import AgentEngine
from ai_agent_framework.orchestration.project import ProjectRegistration

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentCapability",
    "AgentEngine",
    "AgentError",
    "AgentNotFoundError",
    "AgentRegistry",
    "AgentResult",
    "ConfigurationError",
    "ConversationMemory",
    "Event",
    "EventBus",
    "EventType",
    "ExecutionContext",
    "FrameworkError",
    "LoopLimitExceededError",
    "MemoryStore",
    "ModelProvider",
    "ModelProviderError",
    "ModelRequest",
    "ModelResponse",
    "Permission",
    "PermissionDeniedError",
    "PermissionSet",
    "ProjectRegistration",
    "Supervisor",
    "Task",
    "TaskResult",
    "TaskStatus",
    "Tool",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationError",
    "tool",
    "__version__",
]
