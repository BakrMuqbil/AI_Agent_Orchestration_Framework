"""
Execution strategies.

The Supervisor decides *which* agent(s) to call and *when* to stop, one
decision at a time. These strategies are lower-level helpers for the
common cases of running several already-chosen agents against the same
input, either one after another (each seeing the prior's output) or
concurrently (independent, results collected together). They are used by
the Supervisor's dispatch step when a planner's decision names multiple
agents, and are also exposed directly for host applications that want to
bypass planning and just fan work out explicitly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ai_agent_framework.agents.base import Agent, AgentResult
from ai_agent_framework.core.context import ExecutionContext


@dataclass
class NamedAgentResult:
    agent_name: str
    result: AgentResult


async def run_sequential(
    agents: list[Agent],
    task: str,
    context: ExecutionContext,
) -> list[NamedAgentResult]:
    """Run agents one after another.

    Each agent receives the same ``task`` text unless a caller composes a
    richer task string per step; the previous agent's output is recorded
    in ``context.history`` (via the caller) so later agents can see it if
    their own logic reads context history. This strategy stops early if
    an agent fails, so a broken step doesn't waste calls on agents that
    depend on it.
    """
    results: list[NamedAgentResult] = []
    for agent in agents:
        agent_context = context.child(current_agent=agent.name)
        result = await agent.run(task, agent_context)
        results.append(NamedAgentResult(agent_name=agent.name, result=result))
        context.record(
            {"agent_name": agent.name, "output": result.output, "success": result.success, "error": result.error}
        )
        if not result.success:
            break
    return results


async def run_parallel(
    agents: list[Agent],
    task: str,
    context: ExecutionContext,
) -> list[NamedAgentResult]:
    """Run agents concurrently against the same task, independent of each other."""

    async def _run_one(agent: Agent) -> NamedAgentResult:
        agent_context = context.child(current_agent=agent.name)
        result = await agent.run(task, agent_context)
        return NamedAgentResult(agent_name=agent.name, result=result)

    results = await asyncio.gather(*(_run_one(a) for a in agents))
    for r in results:
        context.record(
            {
                "agent_name": r.agent_name,
                "output": r.result.output,
                "success": r.result.success,
                "error": r.result.error,
            }
        )
    return list(results)
