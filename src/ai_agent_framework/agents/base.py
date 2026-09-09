"""
Agent interface.

An ``Agent`` is a named, described unit of capability the supervisor can
dispatch a (sub-)task to. Agents declare what they can do
(``capabilities``), which tools they need (``tools``), what permission
level and model they require, and implement ``run`` to actually do the
work.

The framework core never knows what a concrete agent *does* -- a
``ResearchAgent``, a ``ProductAgent``, a ``ValidatorAgent`` are all just
``Agent`` subclasses from the framework's point of view. Domain-specific
agents belong in a host application or in ``examples/``, never in core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ai_agent_framework.security.permissions import Permission

if TYPE_CHECKING:
    from ai_agent_framework.core.context import ExecutionContext
    from ai_agent_framework.models.base import ModelProvider
    from ai_agent_framework.tools.base import Tool


@dataclass
class AgentCapability:
    """A single declared capability of an agent.

    Capabilities are what the supervisor reasons over when deciding which
    agent to route a task to -- keep them short, specific, and written the
    way you'd describe the capability to another engineer (or a model).
    """

    name: str
    description: str = ""


@dataclass
class AgentResult:
    """The result an agent returns from :meth:`Agent.run`."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any, **metadata: Any) -> "AgentResult":
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "AgentResult":
        return cls(success=False, error=error, metadata=metadata)


class Agent:
    """Base class every agent implements.

    Subclasses set class-level attributes for static metadata and
    implement :meth:`run` for behavior:

    Attributes:
        name: Unique identifier used for registration and routing.
        description: What this agent does, written for a supervisor
            (human or model) deciding whether to route work to it.
        capabilities: Specific things this agent can do.
        tools: The tools this agent is allowed to use. The orchestration
            layer scopes the agent's view of the tool registry to this
            list -- an agent cannot call a tool it hasn't declared.
        permission: The permission level granted to this agent's tool
            calls. Individual tools may require a higher permission than
            an agent holds, in which case calls are rejected -- see
            :mod:`ai_agent_framework.security.permissions`.
        required_model_capability: An optional free-text hint (e.g.
            "supports_function_calling") a host application can use when
            choosing which model to run this agent against. The framework
            does not interpret this value itself.
    """

    name: str = ""
    description: str = ""
    capabilities: list[AgentCapability] = []
    tools: list["Tool"] = []
    permission: Permission = Permission.READ
    required_model_capability: str | None = None

    def __init__(self, *, model: "ModelProvider | None" = None) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a non-empty 'name'")
        self.model = model

    async def run(self, task: str, context: "ExecutionContext") -> AgentResult:
        """Execute this agent against the given (sub-)task.

        Must be implemented by subclasses. ``task`` is the natural-language
        or structured description of what this agent should do -- typically
        the supervisor's decomposition of the overall task goal, not
        necessarily the full original user request.
        """
        raise NotImplementedError

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def describe(self) -> dict[str, Any]:
        """A structured description of this agent, used by the supervisor.

        Intended to be cheap enough to compute for every registered agent
        every time the supervisor needs to decide who to route to (e.g.
        when building a prompt listing available agents).
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [c.name for c in self.capabilities],
            "tools": self.tool_names(),
            "permission": str(self.permission),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
