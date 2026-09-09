# Architecture

## Overview

The AI Agent Orchestration Framework is organized into independent layers.
Each layer depends only on the layers below it (via small interfaces), and
none of them know anything about a specific business domain (Dalalti,
e-commerce, CRM, ...).

```
AI Agent Orchestration Framework
│
├── core/            Task, ExecutionContext, Event/EventBus, error hierarchy
├── agents/          Agent interface, AgentRegistry
├── tools/           Tool interface, @tool decorator, ToolRegistry
├── models/          ModelProvider interface + Ollama/OpenAI/Gemini providers, factory
├── memory/          MemoryStore interface, ConversationMemory, AgentStateMemory, InMemoryStore
├── rag/             Document, Retriever, EmbeddingProvider interfaces + simple implementations
├── security/        Permission, PermissionSet
├── orchestration/   Supervisor, Planner, AgentEngine, ProjectRegistration, strategies
└── api/             FastAPI app factory + routes
```

## Dependency direction

```
api  ─────────────┐
                   ▼
orchestration ──► agents ──► core
     │        ──► tools  ──► core
     │        ──► models ──► core
     │        ──► memory ──► core
     │        ──► rag    ──► (no core dependency)
     └────────► security ──► (no dependency)
```

`core` has no dependency on anything else in the framework. Everything else
depends on `core` for `ExecutionContext`, `Task`, `Event`, and the error
hierarchy, but core never imports back up into agents/tools/models/etc.
This is what makes each layer independently testable and independently
replaceable.

## The central object: `ExecutionContext`

Every layer receives an `ExecutionContext` rather than reaching for global
state. It carries:

- `user_id` / `session_id` / `task_id` / `conversation_id` -- identity
- `tool_registry` -- which tools are callable right now
- `permissions` -- what's allowed right now
- `memory` -- where conversation/agent state lives
- `event_bus` -- where to publish observability events
- `metadata` -- free-form host-application data the framework never reads
- `history` -- append-only record of what's happened in this task so far

Because everything threads through one object, an agent or tool never needs
a global singleton or an import-time side effect to find its dependencies.

## The execution cycle

```
Input (Task.goal)
      │
      ▼
 Supervisor.run()
      │
      ▼
┌─────────────────────────────────────────────┐
│ loop (bounded by SupervisorConfig limits)     │
│                                               │
│   Planner.decide(task, agents, context)       │
│         │                                     │
│         ├─ "call_agent" ──► dispatch to Agent │
│         │                       │             │
│         │                       ▼             │
│         │                 AgentResult         │
│         │                       │             │
│         │                       ▼             │
│         │               context.record(...)   │
│         │                       │             │
│         │                       ▼             │
│         │              (loop: re-plan)         │
│         │                                     │
│         ├─ "finish" ──► TaskResult(COMPLETED)  │
│         │                                     │
│         └─ "fail" ──► TaskResult(FAILED)       │
└─────────────────────────────────────────────┘
```

The mechanics of the loop (bookkeeping, safety limits, event emission) live
in `Supervisor`. The *policy* of what to do next lives in a `Planner`
(`ModelPlanner` for LLM-driven decisions, `RuleBasedPlanner` for
deterministic/no-LLM decisions). This separation is what makes the
supervisor genuinely swappable -- a host application can supply its own
`Planner` implementation without touching `Supervisor` at all.

## Where a new project's code goes

A project (Dalalti, a CRM, ...) never modifies files under `src/`. It
defines its own `Agent` subclasses and `@tool`-decorated functions,
bundles them in a `ProjectRegistration`, and calls `.apply(engine)`. See
[integrations.md](integrations.md) and `examples/dalalti/`.

## Why not a single mega-class?

The original project's `run_agent()` function combined prompt building,
tool dispatch, memory reads/writes, and response formatting in one place.
That's fine for one agent doing one thing, but it can't be reused: adding a
second agent means duplicating the function, and adding a second model
provider means branching inside it. Splitting into `Agent` / `Tool` /
`ModelProvider` / `Supervisor` as separate, independently testable
interfaces is what makes `from ai_agent_framework import Agent, Supervisor,
Tool` usable in a project this framework has never heard of.
