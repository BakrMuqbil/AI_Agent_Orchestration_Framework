"""Agent registry: registration and lookup, keyed by agent name."""

from __future__ import annotations

from typing import Iterable

from ai_agent_framework.agents.base import Agent
from ai_agent_framework.core.errors import AgentNotFoundError


class AgentRegistry:
    """Holds the set of agents available to the orchestration engine."""

    def __init__(self, agents: Iterable[Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        for a in agents or []:
            self.register(a)

    def register(self, agent: Agent) -> None:
        """Register an agent. Re-registering the same name overwrites it."""
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentNotFoundError(name) from exc

    def has(self, name: str) -> bool:
        return name in self._agents

    def list(self) -> list[Agent]:
        return list(self._agents.values())

    def describe_all(self) -> list[dict]:
        """Structured descriptions of every registered agent.

        This is what the supervisor feeds to its planning step so it can
        reason about which agent(s) fit a given task.
        """
        return [a.describe() for a in self._agents.values()]

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents.values())
