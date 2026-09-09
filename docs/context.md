# Execution Context

## What it carries

`ExecutionContext` is threaded through every layer of a task's execution
(supervisor to agent to tool to model). It carries exactly what execution
needs and nothing domain-specific:

```python
@dataclass
class ExecutionContext:
    user_id: str | None = None
    session_id: str = field(default_factory=...)
    task_id: str | None = None
    conversation_id: str | None = None
    current_agent: str | None = None

    tool_registry: ToolRegistry | None = None
    permissions: PermissionSet | None = None
    memory: MemoryStore | None = None
    event_bus: EventBus | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
```

- `user_id` / `session_id` / `task_id` / `conversation_id` -- identity and
  grouping, all opaque to the framework.
- `current_agent` -- which agent is executing right now, set by the
  supervisor when it dispatches.
- `tool_registry` / `permissions` / `memory` / `event_bus` -- the four
  things an agent or tool typically needs, available uniformly rather than
  as separate parameters threaded through every function signature.
- `metadata` -- free-form, host-supplied data. The framework never reads
  business meaning out of this dict; it exists so a host application can
  carry arbitrary extra data through a task's execution without the
  framework needing to know about it.
- `history` -- append-only record of execution steps taken so far. Used by
  planners to reason about what's already happened, and by observability
  tooling to reconstruct a task's timeline.

## `context.record(entry)`

Appends a dict entry to `history`. Called automatically by the supervisor
after each agent dispatch and by the execution strategies
(`run_sequential` / `run_parallel`) -- host code generally doesn't need to
call this directly, but can for custom planners or manual orchestration.

## `context.child(**overrides)`

Creates a copy of the context with specific fields overridden, used when
dispatching to a sub-agent that should share most context but have (for
example) a different `current_agent`:

```python
agent_context = context.child(current_agent="research_agent")
```

Fields not overridden are shared with the parent context object
(`dataclasses.replace` semantics) -- notably `history` is the same list
object unless explicitly overridden, so a child context's recorded steps
are visible to the parent and vice versa. This is intentional: agents
further down the call chain should see what's already happened in the
task.

## What's deliberately not here

The context does not carry business objects (a "current product," a
"current customer"). Anything domain-specific belongs in `metadata` or in
the arguments passed to a specific agent/tool call -- keeping the context
itself free of domain assumptions is what lets the same
`ExecutionContext` type work for a Dalalti agent and a CRM agent without
either one needing fields the other doesn't use.
