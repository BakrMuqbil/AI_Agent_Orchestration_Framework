"""
AgentEngine.

The top-level facade a host application interacts with. It owns the
agent registry, tool registry, memory, event bus, and a supervisor, and
exposes the ergonomic surface described in the framework's design goal::

    engine = AgentEngine(model=create_provider("ollama", model="qwen2.5"))
    engine.register_agent(ProductAgent())
    engine.register_agent(ResearchAgent())
    result = await engine.run("Analyze this product and prepare a recommendation")

Everything ``AgentEngine`` does is compose already-independent pieces
(``AgentRegistry``, ``ToolRegistry``, ``Supervisor``, ``ConversationMemory``)
-- it holds no orchestration logic of its own beyond wiring them together
and threading a session's context through calls.
"""

from __future__ import annotations

from typing import Any

from ai_agent_framework.agents.base import Agent
from ai_agent_framework.agents.registry import AgentRegistry
from ai_agent_framework.core.context import ExecutionContext
from ai_agent_framework.core.events import EventBus
from ai_agent_framework.core.task import Task, TaskResult
from ai_agent_framework.memory.base import ConversationMemory, MemoryStore
from ai_agent_framework.memory.in_memory import InMemoryStore
from ai_agent_framework.models.base import ModelProvider
from ai_agent_framework.orchestration.supervisor import (
    ModelPlanner,
    Planner,
    RuleBasedPlanner,
    Supervisor,
    SupervisorConfig,
)
from ai_agent_framework.security.permissions import PermissionSet
from ai_agent_framework.tools.registry import ToolRegistry


class AgentEngine:
    """Top-level entry point for registering agents and running tasks."""

    def __init__(
        self,
        *,
        model: ModelProvider | None = None,
        planner: Planner | None = None,
        agents: AgentRegistry | None = None,
        tools: ToolRegistry | None = None,
        memory_store: MemoryStore | None = None,
        permissions: PermissionSet | None = None,
        config: SupervisorConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.model = model
        self.agents = agents or AgentRegistry()
        self.tools = tools or ToolRegistry()
        self.memory_store = memory_store or InMemoryStore()
        self.memory = ConversationMemory(self.memory_store)
        self.permissions = permissions or PermissionSet.admin()
        self.event_bus = event_bus or EventBus()
        self.config = config or SupervisorConfig()

        resolved_planner = planner or (ModelPlanner(model) if model is not None else RuleBasedPlanner())
        self.planner = resolved_planner

        self.supervisor = Supervisor(
            agents=self.agents,
            planner=resolved_planner,
            config=self.config,
            event_bus=self.event_bus,
        )

    def register_agent(self, agent: Agent) -> "AgentEngine":
        """Register an agent with the engine. Returns ``self`` for chaining."""
        self.agents.register(agent)
        return self

    def register_tool(self, tool: Any) -> "AgentEngine":
        """Register a globally-available tool. Returns ``self`` for chaining."""
        self.tools.register(tool)
        return self

    async def run(
        self,
        goal: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> TaskResult:
        """Run a task through the supervisor and return its result.

        A fresh :class:`~ai_agent_framework.core.context.ExecutionContext`
        is built (or the supplied one reused) with this engine's tool
        registry, permissions, memory, and event bus attached, so any
        agent invoked during the run has access to all of them uniformly.
        """
        task = Task(goal=goal, input=input or {}, metadata=metadata or {})

        exec_context = context or ExecutionContext(
            user_id=user_id,
            session_id=session_id or task.id,
        )
        exec_context.tool_registry = exec_context.tool_registry or self.tools
        exec_context.permissions = exec_context.permissions or self.permissions
        exec_context.memory = exec_context.memory or self.memory
        exec_context.event_bus = exec_context.event_bus or self.event_bus

        return await self.supervisor.run(task, exec_context)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"AgentEngine(agents={len(self.agents)}, tools={len(self.tools)})"
