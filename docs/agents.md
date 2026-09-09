# Agents

## What an agent is

An `Agent` is a named, described unit of capability the supervisor can
dispatch a task (or sub-task) to. The framework core never knows what a
concrete agent *does* -- `ResearchAgent`, `ProductAgent`, `ValidatorAgent`
are all just `Agent` subclasses.

## Defining a new agent

```python
from ai_agent_framework import Agent, AgentCapability, AgentResult, Permission
from my_tools import search_web, summarize_text

class ResearchAgent(Agent):
    name = "research_agent"
    description = "Searches the web and summarizes findings"
    capabilities = [
        AgentCapability(name="web_search", description="Search the web for information"),
        AgentCapability(name="summarization", description="Summarize long text"),
    ]
    tools = [search_web, summarize_text]
    permission = Permission.READ

    async def run(self, task: str, context) -> AgentResult:
        # Use context.tool_registry to call tools, context.model (if you
        # attach one) to call the LLM, context.memory for conversation
        # state, etc.
        result = await context.tool_registry.invoke(
            "search_web",
            {"query": task},
            permissions=context.permissions,
            context=context,
        )
        return AgentResult.ok(result.output)
```

## Required attributes

| Attribute | Purpose |
|---|---|
| `name` | Unique identifier used for registration and routing. Required -- raises `ValueError` if empty. |
| `description` | What this agent does, written for a supervisor (human or model) deciding whether to route work to it. |
| `capabilities` | List of `AgentCapability(name, description)` -- specific things this agent can do. Used by the supervisor's planner to reason about routing. |
| `tools` | The tools this agent is allowed to declare/use. The orchestration layer can scope a tool registry to exactly this list via `ToolRegistry.scoped()`. |
| `permission` | The `Permission` level this agent's tool calls run with. |
| `required_model_capability` | Optional free-text hint for host applications choosing which model to run this agent against. Not interpreted by the framework. |

## The `run` method

```python
async def run(self, task: str, context: ExecutionContext) -> AgentResult
```

- `task` -- the natural-language (or structured, if you choose to encode
  it that way) description of what this agent should do. This is often the
  supervisor's decomposition of the overall goal, not necessarily the
  original user request verbatim.
- `context` -- the `ExecutionContext` for this invocation, carrying tools,
  permissions, memory, and history.
- Returns an `AgentResult`: `AgentResult.ok(output, **metadata)` on
  success, `AgentResult.fail(error, **metadata)` on failure. Raising an
  exception is also handled -- the supervisor normalizes it into a failed
  step -- but returning `AgentResult.fail(...)` is preferred since it lets
  you attach a clear error message without an exception's stack trace
  noise.

## Agent lifecycle

1. **Construction**: `MyAgent(model=optional_model_provider)`. The base
   `__init__` validates that `name` is set.
2. **Registration**: `engine.register_agent(my_agent)` or
   `AgentRegistry([my_agent, ...])`. Re-registering the same name
   overwrites the previous instance.
3. **Selection**: the supervisor's planner decides to call this agent by
   name for a given (sub-)task.
4. **Dispatch**: the supervisor builds a child `ExecutionContext`
   (`context.child(current_agent=agent.name)`) and calls `agent.run(...)`.
5. **Result recording**: the `AgentResult` is normalized into an
   `AgentStepResult` and appended to both `TaskResult.steps` and
   `context.history`, so later planning iterations (and other agents, if
   they read context history) can see it.
6. **Unregistration**: `AgentRegistry.unregister(name)` if needed at
   runtime.

## Multi-agent registration

```python
engine.register_agent(ProductAgent())
engine.register_agent(ResearchAgent())
engine.register_agent(AnalyticsAgent())

result = await engine.run("Analyze this product and prepare a recommendation")
```

The supervisor decides which agent(s) the task actually needs -- see
[supervisor.md](supervisor.md).
