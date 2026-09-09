# Orchestration

This document covers the pieces that sit above individual agents:
`AgentEngine` (the top-level facade), execution strategies, and
`ProjectRegistration` (covered in depth in [integrations.md](integrations.md)).

## `AgentEngine`: the entry point

`AgentEngine` composes an `AgentRegistry`, `ToolRegistry`, `ConversationMemory`,
`EventBus`, and a `Supervisor` into one object with an ergonomic surface:

```python
from ai_agent_framework import AgentEngine, create_provider

provider = create_provider("ollama", model="llama3.1")
engine = AgentEngine(model=provider)

engine.register_agent(ProductAgent())
engine.register_agent(ResearchAgent())

result = await engine.run("Analyze this product and prepare a recommendation")
```

`AgentEngine.run(goal, *, session_id=None, user_id=None, input=None, metadata=None, context=None)`:

1. Builds a `Task(goal=goal, input=input, metadata=metadata)`.
2. Builds (or reuses a supplied) `ExecutionContext`, filling in the
   engine's tool registry, permissions, memory, and event bus wherever the
   context doesn't already have them set.
3. Runs `Supervisor.run(task, context)` and returns the `TaskResult`.

If `model` is omitted, `AgentEngine` defaults to a `RuleBasedPlanner`
(no LLM calls at all) -- useful for tests and demos. Passing `model=`
switches to `ModelPlanner` automatically, or you can supply your own
`planner=` explicitly to override either default.

## Execution strategies

For cases where a host application (or a custom `Planner`) already knows
it wants to run several specific agents against the same input, rather
than letting the supervisor decide one agent at a time,
`ai_agent_framework.orchestration.strategies` provides two helpers:

```python
from ai_agent_framework.orchestration.strategies import run_sequential, run_parallel

# One after another; stops early if an agent fails.
results = await run_sequential([agent_a, agent_b], "the task", context)

# All at once, independent of each other.
results = await run_parallel([agent_a, agent_b], "the task", context)
```

Both return a list of `NamedAgentResult(agent_name, result)` and record
each step into `context.history`, the same as the supervisor's own
dispatch does -- so a custom `Planner` can call these directly and still
have later planning iterations see what happened.

## Configuration

Execution limits (`max_iterations`, `max_agent_calls`, `max_tool_calls`,
`timeout_seconds`) are set via `SupervisorConfig`, either directly:

```python
from ai_agent_framework.orchestration.supervisor import SupervisorConfig

engine = AgentEngine(model=provider, config=SupervisorConfig(max_iterations=4))
```

or via environment variables through `FrameworkSettings` (see the
Configuration section in the main README).
