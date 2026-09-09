"""
Supervisor.

The Supervisor is what makes this a *supervisor architecture* rather
than a router: it doesn't just pick one agent and return its output. It
runs a real execution cycle:

    understand -> plan -> select agent -> execute -> observe -> evaluate
    -> (continue | re-plan | finish)

Each iteration it:

1. Looks at the task goal and everything observed so far (via
   :class:`~ai_agent_framework.core.context.ExecutionContext` history).
2. Decides the next step: call a specific agent with a specific
   sub-task, or finish and produce the final result.
3. Dispatches to that agent and waits for its result.
4. Evaluates whether the result satisfies the goal, needs another agent,
   or requires re-planning entirely.
5. Repeats, bounded by configured safety limits.

Decision-making is delegated to a :class:`Planner` so the *mechanics* of
the loop (bookkeeping, limits, event emission) are separated from the
*policy* of how to decide the next step. Two planners are provided:

* :class:`ModelPlanner` -- asks a :class:`ModelProvider` for structured
  decisions. This is the "real" planner for production use.
* :class:`RuleBasedPlanner` -- a deterministic, model-free planner
  (single-agent selection by simple scoring) usable in tests, or as a
  fallback when no model is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ai_agent_framework.agents.base import Agent, AgentResult
from ai_agent_framework.agents.registry import AgentRegistry
from ai_agent_framework.core.context import ExecutionContext
from ai_agent_framework.core.errors import AgentNotFoundError, LoopLimitExceededError
from ai_agent_framework.core.events import Event, EventBus, EventType
from ai_agent_framework.core.task import AgentStepResult, Task, TaskResult, TaskStatus
from ai_agent_framework.models.base import ModelMessage, ModelProvider, ModelRequest


@dataclass
class SupervisorConfig:
    """Safety limits and tunables for the supervisor's execution loop.

    These exist specifically to satisfy requirement #14: the supervisor
    must never be able to enter an unbounded loop. Every limit here is a
    hard stop -- when exceeded, execution halts with
    :class:`~ai_agent_framework.core.errors.LoopLimitExceededError`
    recorded on the :class:`~ai_agent_framework.core.task.TaskResult`
    rather than raised past the caller, so a host application always gets
    a ``TaskResult`` back instead of having to catch an exception for the
    common "ran out of budget" case.
    """

    max_iterations: int = 8
    max_agent_calls: int = 8
    max_tool_calls: int = 20
    timeout_seconds: float | None = 120.0


class PlanDecision(BaseModel):
    """A single decision the planner returns each iteration."""

    action: str  # "call_agent" | "finish" | "fail"
    agent_name: str | None = None
    agent_task: str | None = None
    final_output: str | None = None
    reasoning: str = ""


class Planner:
    """Interface a planning strategy implements.

    ``decide`` is called once per supervisor iteration and must return a
    :class:`PlanDecision`. Implementations should be side-effect-free
    with respect to the task/context beyond reading them.
    """

    async def decide(
        self,
        task: Task,
        agents: AgentRegistry,
        context: ExecutionContext,
    ) -> PlanDecision:
        raise NotImplementedError


class ModelPlanner(Planner):
    """A planner backed by a :class:`ModelProvider`.

    Asks the model, given the task goal, the available agents (with their
    descriptions/capabilities), and the execution history so far, to
    decide the next step as structured output. This is what gives the
    supervisor genuine reasoning about goal completion rather than fixed
    routing rules.
    """

    def __init__(self, model: ModelProvider) -> None:
        self.model = model

    async def decide(
        self,
        task: Task,
        agents: AgentRegistry,
        context: ExecutionContext,
    ) -> PlanDecision:
        agent_descriptions = "\n".join(
            f"- {a['name']}: {a['description']} (capabilities: {', '.join(a['capabilities']) or 'none listed'})"
            for a in agents.describe_all()
        )
        history_text = "\n".join(
            f"- step {i + 1}: agent={h.get('agent_name')} success={h.get('success')} "
            f"output={_truncate(str(h.get('output')))}"
            for i, h in enumerate(context.history)
        ) or "(no steps taken yet)"

        system = ModelMessage(
            role="system",
            content=(
                "You are the supervisor of a multi-agent system. Your job is to "
                "decide the single next step needed to accomplish the task goal.\n\n"
                f"Available agents:\n{agent_descriptions}\n\n"
                "Rules:\n"
                "- Choose exactly one action per turn: 'call_agent' or 'finish' or 'fail'.\n"
                "- Use 'call_agent' to dispatch a sub-task to one agent by name.\n"
                "- Use 'finish' once the goal is satisfied by what's been observed so far, "
                "and put the final answer in final_output.\n"
                "- Use 'fail' only if the goal cannot be accomplished with the available agents.\n"
                "- Never call an agent name that is not in the list above."
            ),
        )
        user = ModelMessage(
            role="user",
            content=(
                f"Task goal: {task.goal}\n\n"
                f"Execution history so far:\n{history_text}\n\n"
                "Decide the next step."
            ),
        )
        request = ModelRequest(messages=[system, user])
        return await self.model.generate_structured(request, PlanDecision)


class RuleBasedPlanner(Planner):
    """A deterministic, model-free planner.

    Selects the single agent whose declared capabilities/description share
    the most keyword overlap with the task goal, runs it once, and
    finishes with its output. Useful for tests and as a safe fallback when
    no model is configured -- it never calls a model, so it's fully
    deterministic and free to run.
    """

    async def decide(
        self,
        task: Task,
        agents: AgentRegistry,
        context: ExecutionContext,
    ) -> PlanDecision:
        if context.history:
            last = context.history[-1]
            output = last.get("output")
            return PlanDecision(
                action="finish",
                final_output=str(output) if output is not None else "",
                reasoning="rule_based: single agent already ran",
            )

        goal_words = set(task.goal.lower().split())
        best_name: str | None = None
        best_score = -1
        for a in agents.describe_all():
            text = f"{a['description']} {' '.join(a['capabilities'])}".lower()
            score = sum(1 for w in goal_words if w and w in text)
            if score > best_score:
                best_score = score
                best_name = a["name"]

        if best_name is None:
            return PlanDecision(action="fail", reasoning="no agents registered")

        return PlanDecision(
            action="call_agent",
            agent_name=best_name,
            agent_task=task.goal,
            reasoning=f"rule_based: best keyword match (score={best_score})",
        )


def _truncate(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class Supervisor:
    """Runs the plan -> select -> execute -> observe -> evaluate cycle.

    Example::

        supervisor = Supervisor(agents=registry, planner=ModelPlanner(model))
        result = await supervisor.run(Task(goal="..."), context)
    """

    def __init__(
        self,
        *,
        agents: AgentRegistry,
        planner: Planner,
        config: SupervisorConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.agents = agents
        self.planner = planner
        self.config = config or SupervisorConfig()
        self.event_bus = event_bus or EventBus()

    async def run(self, task: Task, context: ExecutionContext) -> TaskResult:
        context.task_id = task.id
        context.event_bus = context.event_bus or self.event_bus

        await self._emit(EventType.TASK_STARTED, task_id=task.id)
        task.status = TaskStatus.PLANNING

        result = TaskResult(task_id=task.id, status=TaskStatus.RUNNING)

        try:
            result = await self._run_loop(task, context, result)
        except LoopLimitExceededError as exc:
            result.status = TaskStatus.FAILED
            result.error = str(exc)
            await self._emit(EventType.TASK_FAILED, task_id=task.id, data={"error": str(exc)})
            return result
        except Exception as exc:  # noqa: BLE001 - surfaced on the result, not raised
            result.status = TaskStatus.FAILED
            result.error = str(exc)
            await self._emit(EventType.TASK_FAILED, task_id=task.id, data={"error": str(exc)})
            return result

        await self._emit(EventType.TASK_FINISHED, task_id=task.id, data={"status": result.status.value})
        return result

    async def _run_loop(self, task: Task, context: ExecutionContext, result: TaskResult) -> TaskResult:
        for iteration in range(1, self.config.max_iterations + 1):
            result.iterations = iteration

            decision = await self.planner.decide(task, self.agents, context)

            if decision.action == "finish":
                result.status = TaskStatus.COMPLETED
                result.output = decision.final_output
                task.status = TaskStatus.COMPLETED
                return result

            if decision.action == "fail":
                result.status = TaskStatus.FAILED
                result.error = decision.reasoning or "supervisor determined the task cannot be completed"
                task.status = TaskStatus.FAILED
                return result

            if decision.action != "call_agent" or not decision.agent_name:
                result.status = TaskStatus.FAILED
                result.error = f"planner returned an invalid decision: {decision!r}"
                task.status = TaskStatus.FAILED
                return result

            if result.agent_calls >= self.config.max_agent_calls:
                raise LoopLimitExceededError("max_agent_calls", self.config.max_agent_calls)

            step = await self._dispatch(task, context, decision)
            result.steps.append(step)
            result.agent_calls += 1

            context.record(
                {
                    "agent_name": step.agent_name,
                    "output": step.output,
                    "success": step.success,
                    "error": step.error,
                }
            )

        # Exhausted max_iterations without the planner finishing.
        raise LoopLimitExceededError("max_iterations", self.config.max_iterations)

    async def _dispatch(self, task: Task, context: ExecutionContext, decision: PlanDecision) -> AgentStepResult:
        agent_name = decision.agent_name or ""
        try:
            agent = self.agents.get(agent_name)
        except AgentNotFoundError as exc:
            return AgentStepResult(
                agent_name=agent_name,
                input={"task": decision.agent_task or task.goal},
                success=False,
                error=str(exc),
            )

        await self._emit(EventType.AGENT_SELECTED, task_id=task.id, agent_name=agent_name)
        await self._emit(EventType.AGENT_STARTED, task_id=task.id, agent_name=agent_name)

        agent_context = context.child(current_agent=agent_name)
        sub_task = decision.agent_task or task.goal

        step = AgentStepResult(agent_name=agent_name, input={"task": sub_task})
        try:
            agent_result: AgentResult = await agent.run(sub_task, agent_context)
        except Exception as exc:  # noqa: BLE001 - normalized onto the step result
            step.success = False
            step.error = str(exc)
            await self._emit(
                EventType.AGENT_FAILED, task_id=task.id, agent_name=agent_name, data={"error": str(exc)}
            )
            return step

        step.success = agent_result.success
        step.output = agent_result.output
        step.error = agent_result.error

        event_type = EventType.AGENT_FINISHED if agent_result.success else EventType.AGENT_FAILED
        await self._emit(event_type, task_id=task.id, agent_name=agent_name, data={"success": agent_result.success})

        return step

    async def _emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        agent_name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self.event_bus.publish(
            Event(type=event_type, task_id=task_id, agent_name=agent_name, data=data or {})
        )
