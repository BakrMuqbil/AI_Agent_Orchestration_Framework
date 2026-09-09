"""
Project integration layer.

Lets a host application (Dalalti, an e-commerce system, a CRM, ...)
register its own agents, tools, and permissions with an
:class:`~ai_agent_framework.orchestration.engine.AgentEngine` as a single
named unit, without the framework's core ever needing to know that
project exists.

Example::

    from ai_agent_framework import AgentEngine, ProjectRegistration

    dalalti = ProjectRegistration(
        name="dalalti",
        agents=[ProductAgent(), OrderAgent(), CustomerAgent()],
        tools=[create_product_tool, update_order_tool],
    )

    engine = AgentEngine(model=my_model)
    dalalti.apply(engine)

This is intentionally a thin composition helper, not a plugin system with
its own lifecycle hooks -- ``apply`` just registers agents and tools onto
the engine's existing registries. Multiple projects can be applied to the
same engine; each project's agents/tools coexist as long as their names
don't collide (a collision overwrites, matching
:meth:`~ai_agent_framework.agents.registry.AgentRegistry.register`'s
behavior, so a host application that wants isolation should namespace
its agent/tool names, e.g. ``"dalalti.product_agent"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ai_agent_framework.agents.base import Agent
from ai_agent_framework.security.permissions import PermissionSet
from ai_agent_framework.tools.base import Tool

if TYPE_CHECKING:
    from ai_agent_framework.orchestration.engine import AgentEngine


@dataclass
class ProjectRegistration:
    """Bundles a project's agents, tools, and configuration for registration.

    Attributes:
        name: A short identifier for the project, used only for
            observability/metadata (e.g. tagging events) -- never
            interpreted by core framework logic.
        agents: Agents this project contributes.
        tools: Globally-available tools this project contributes. Agents
            that need a tool must still declare it in their own
            ``tools`` list; adding it here only makes it available to be
            declared, it does not implicitly grant every agent access.
        permissions: Optional permission set this project's agents should
            run with, if different from the engine's default. Applied by
            setting each agent's ``permission`` attribute is out of scope
            here since ``Agent.permission`` is a class attribute; host
            applications that need per-project permission scoping should
            construct their agents with the desired permission already
            set, or pass a scoped ``ExecutionContext`` per call.
        metadata: Free-form data describing the project (version, owner,
            docs URL, ...). Not read by the framework.
    """

    name: str
    agents: list[Agent] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    permissions: PermissionSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, engine: "AgentEngine") -> "AgentEngine":
        """Register this project's agents and tools onto ``engine``."""
        for agent in self.agents:
            engine.register_agent(agent)
        for tool in self.tools:
            engine.register_tool(tool)
        return engine
