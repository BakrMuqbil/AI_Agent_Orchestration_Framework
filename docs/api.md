# API

## Overview

The framework exposes a general, provider-independent HTTP API via
FastAPI. It is not tied to any specific host application's domain --
routes operate purely in terms of framework concepts (tasks, agents,
tools).

## Creating the app

```python
from ai_agent_framework import AgentEngine, create_provider
from ai_agent_framework.api.app import create_app

provider = create_provider("ollama", model="llama3.1")
engine = AgentEngine(model=provider)
engine.register_agent(MyAgent())

app = create_app(engine)
```

create_app is a factory, not a single global app object, specifically so
a host application can build its own engine (its own agents, tools,
provider) and get a ready-to-serve API back, rather than the framework
owning one process-wide instance.

The bundled server module does exactly this, wiring settings, provider
creation, engine construction, and app creation together for local
development and as a runnable reference. Run it with uvicorn pointed at
that module's app object.

## Endpoints

All routes are mounted under the /api/v1 prefix.

### POST /api/v1/tasks

Run a task through the supervisor.

Request body:

```json
{
  "goal": "Analyze this product and prepare a recommendation",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id",
  "input": {},
  "metadata": {}
}
```

Response:

```json
{
  "task_id": "...",
  "status": "completed",
  "output": "...",
  "error": null,
  "iterations": 3,
  "agent_calls": 2,
  "steps": [
    {"agent_name": "research_agent", "success": true, "output": "...", "error": null}
  ]
}
```

A 403 response indicates a permission or confirmation error was raised
during execution.

### GET /api/v1/agents

Lists every agent currently registered with the engine: name,
description, capabilities, tools, permission.

### GET /api/v1/tools

Lists every globally-registered tool: name, description, permission,
whether it requires confirmation.

### GET /api/v1/status

Basic liveness/status endpoint: online status plus counts of registered
agents and tools.

## Design notes

- Routes are intentionally minimal and general -- they do not encode
  Dalalti-specific or any other domain-specific paths. A host application
  needing additional domain routes adds its own FastAPI router alongside
  the framework's, using the same engine.
- The engine dependency is overridden by create_app via FastAPI's
  dependency override mechanism, so routes never import a specific engine
  instance directly -- this is what keeps the API layer reusable across
  different host applications' engines.
