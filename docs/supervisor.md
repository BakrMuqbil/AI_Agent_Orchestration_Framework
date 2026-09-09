# Supervisor

## Why "supervisor" and not "router"

A router picks one destination and forwards a message. The `Supervisor`
here runs a real cycle:

1. **Understand** the task goal.
2. **Plan** the next step given everything observed so far.
3. **Select** which agent should take that step.
4. **Execute** by dispatching to that agent.
5. **Observe** the agent's result.
6. **Evaluate** whether the goal is satisfied, another agent is needed, or
   the plan needs to change.
7. **Decide**: continue with another agent call, re-plan, or finish.

This repeats -- with a different agent chosen each time if needed -- until
the planner decides the goal is satisfied (`finish`), decides it can't be
done (`fail`), or a safety limit is hit.

```
User
 ↓
Supervisor  ──plan──►  Research Agent
 ↓                          │
 │◄────────result───────────┘
 ↓
Supervisor  ──plan──►  Product Agent
 ↓                          │
 │◄────────result───────────┘
 ↓
Supervisor  ──plan──►  Validator Agent
 ↓                          │
 │◄────────result───────────┘
 ↓
Final Result
```

## Planner: the decision-making policy

The supervisor's *mechanics* (bookkeeping, limits, event emission) are
separate from its *policy* (how to decide the next step). Policy lives in
a `Planner`:

```python
class Planner:
    async def decide(self, task: Task, agents: AgentRegistry, context: ExecutionContext) -> PlanDecision:
        ...
```

Two planners ship with the framework:

- **`ModelPlanner`** -- asks a `ModelProvider` for a structured
  `PlanDecision` (via `generate_structured`), given the task goal, the
  available agents' descriptions/capabilities, and the execution history
  so far. This is the planner used for real, LLM-driven multi-agent
  reasoning.
- **`RuleBasedPlanner`** -- a deterministic, model-free planner that
  selects the single agent whose description/capabilities share the most
  keyword overlap with the goal, runs it once, and finishes. No network
  call, fully deterministic -- useful for tests, demos, and as a safe
  fallback when no model is configured.

Host applications can implement their own `Planner` (e.g. a hand-written
decision tree for a narrow domain) and pass it to `Supervisor(planner=...)`
or `AgentEngine(planner=...)`.

## `PlanDecision`

```python
class PlanDecision(BaseModel):
    action: str  # "call_agent" | "finish" | "fail"
    agent_name: str | None = None
    agent_task: str | None = None
    final_output: str | None = None
    reasoning: str = ""
```

- `call_agent` -- dispatch `agent_task` (or the original task goal, if
  `agent_task` is omitted) to `agent_name`.
- `finish` -- the goal is satisfied; `final_output` becomes
  `TaskResult.output`.
- `fail` -- the goal cannot be accomplished with the available agents;
  `reasoning` becomes `TaskResult.error`.

## Safety limits

The supervisor can never loop forever. `SupervisorConfig` enforces hard
stops:

```python
@dataclass
class SupervisorConfig:
    max_iterations: int = 8       # total planning iterations
    max_agent_calls: int = 8      # total agent dispatches
    max_tool_calls: int = 20      # advisory limit for host-app tool-call bookkeeping
    timeout_seconds: float | None = 120.0
```

When a limit is exceeded, the supervisor stops and returns a
`TaskResult(status=FAILED, error="... exceeded 'max_iterations' limit of 8")`
-- it never raises an unhandled exception for this case, so a host
application always gets a `TaskResult` back.

## Multi-agent example

```python
from ai_agent_framework import AgentEngine, create_provider

provider = create_provider("ollama", model="llama3.1")
engine = AgentEngine(model=provider)  # uses ModelPlanner automatically

engine.register_agent(ResearchAgent())
engine.register_agent(ProductAgent())
engine.register_agent(ValidatorAgent())

result = await engine.run("Analyze this product and prepare a recommendation")

print(result.status)          # TaskStatus.COMPLETED
print(result.output)          # the final answer
print(result.steps)           # every agent call made along the way
print(result.agent_calls)     # how many agent dispatches happened
```

## Observability

Every meaningful step publishes an `Event` to the configured `EventBus`:
`task_started`, `agent_selected`, `agent_started`, `agent_finished` /
`agent_failed`, `task_finished` / `task_failed`. See
[architecture.md](architecture.md) and the event system in
`ai_agent_framework.core.events`.
