# AI Agent Orchestration Framework

A provider-independent, domain-independent framework for building,
registering, and orchestrating AI agents with tools, memory, and RAG.
Built to be embedded in any project that needs AI agents -- an e-commerce
platform, a CRM, an ERP, an inventory system, or anything else -- without
that project's specifics leaking into the framework's core.

```python
from ai_agent_framework import Agent, AgentEngine, AgentResult, tool, Permission

@tool(name="get_time", description="Get the current UTC time", permission=Permission.READ)
async def get_time() -> dict:
    from datetime import datetime, timezone
    return {"time": datetime.now(timezone.utc).isoformat()}

class TimeAgent(Agent):
    name = "time_agent"
    description = "Reports the current time"
    tools = [get_time]

    async def run(self, task: str, context) -> AgentResult:
        result = await context.tool_registry.invoke(
            "get_time", {}, permissions=context.permissions, context=context
        )
        return AgentResult.ok(result.output)

engine = AgentEngine(model=None)  # or a real ModelProvider
engine.register_agent(TimeAgent())

result = await engine.run("what time is it?")
print(result.output)
```

## 1. What this is

A framework for coordinating multiple AI agents against a shared goal,
with a real supervisor (not just a router) deciding which agent handles
each step, a permission system controlling what agents are allowed to do,
and provider-independent access to whichever LLM you configure. See the
architecture doc in the docs folder for the full layer breakdown.

## 2. Architecture

```
AI Agent Orchestration Framework
│
├── core/            Task, ExecutionContext, Event/EventBus, errors
├── agents/          Agent interface, AgentRegistry
├── tools/           Tool interface, @tool decorator, ToolRegistry
├── models/          ModelProvider interface + Ollama/OpenAI/Gemini + factory
├── memory/          MemoryStore, ConversationMemory, AgentStateMemory
├── rag/             Document, Retriever, EmbeddingProvider + defaults
├── security/        Permission, PermissionSet
├── orchestration/   Supervisor, Planner, AgentEngine, ProjectRegistration
└── api/             FastAPI app factory + routes
```

Full detail is in the architecture document under docs.

## 3. Core concepts

- **Task** -- a unit of work with a natural-language goal.
- **ExecutionContext** -- carries identity, tools, permissions, memory,
  and history through every layer of execution. See the context document
  under docs.
- **Event / EventBus** -- provider-independent observability.
- **Errors** -- a single FrameworkError hierarchy host applications can
  catch generically or specifically.

## 4. Agents

Define an Agent subclass with a name, description, capabilities, tools,
and a run method. Multiple agents register into one engine; the
supervisor decides which one(s) handle a given task. Full detail is in
the agents document under docs.

## 5. Supervisor

Runs a real understand -> plan -> select agent -> execute -> observe ->
evaluate -> continue/re-plan/finish cycle, not just single-shot routing.
Backed by a pluggable Planner -- ModelPlanner for LLM-driven decisions,
RuleBasedPlanner for deterministic/no-LLM decisions. Bounded by hard
safety limits (max_iterations, max_agent_calls, timeout). Full detail is
in the supervisor document under docs.

## 6. Tools

Define tools with the @tool decorator: name, description, a Pydantic args
schema, a required permission, and an optional confirmation requirement
for sensitive operations. A ToolRegistry validates arguments, checks
permissions, and executes, returning a uniform ToolResult. Full detail is
in the tools document under docs.

## 7. Model Providers

Ollama, OpenAI (and OpenAI-compatible gateways like OpenRouter), and
Gemini ship as concrete providers behind one ModelProvider interface.
Selected via configuration (AI_PROVIDER, AI_MODEL) -- no provider is a
hard requirement of the framework, and adding a new provider requires no
changes to agents, the supervisor, or anything else. Structured output
(Pydantic-validated JSON) works uniformly across providers. Full detail
is in the models document under docs.

## 8. Memory

Four kinds of state, kept general and provider-independent: conversation
history, per-task execution state, per-agent key/value state, and durable
long-term summaries. Backed by a small MemoryStore interface (in-memory
by default; swappable for Redis, a database, etc). Full detail is in the
memory document under docs.

## 9. Context

ExecutionContext is the single object threaded through supervisor, agent,
and tool calls, carrying exactly what execution needs and nothing
domain-specific. Full detail is in the context document under docs.

## 10. Orchestration

AgentEngine is the top-level facade: register agents and tools, then call
run(goal). Execution strategies (run_sequential, run_parallel) are
available for cases where a host application wants to run several
specific agents directly rather than letting the supervisor choose one at
a time. Full detail is in the orchestration document under docs.

## 11. Permissions

A hierarchical Permission enum (READ < CREATE < UPDATE < DELETE < ADMIN)
plus named permissions for non-hierarchical cases. Tools declare what
they require; a PermissionSet is granted per engine/context; sensitive
tools can additionally require host-application confirmation before
running. Full detail is in the permissions document under docs.

## 12. API

A general FastAPI layer (task execution, agent listing, tool listing,
status) built via a create_app(engine) factory, so any host application's
engine gets a ready HTTP API. Full detail is in the api document under
docs.

## 13. Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install -e ".[dev]"
# or:
pip install -r requirements.txt
```

## 14. Configuration

All framework settings are environment-variable driven via a settings
class built on pydantic-settings, with no secrets or business-specific
values hard-coded anywhere in the framework. Copy the example environment
file and fill in your values before running anything.

Key variables:

```bash
AI_PROVIDER=ollama          # ollama | openai | gemini | <custom>
AI_MODEL=llama3.1
AI_API_KEY=                 # required for openai / gemini
AI_BASE_URL=                # optional override, e.g. for OpenRouter
AI_TEMPERATURE=0.3
AI_MAX_TOKENS=800

MAX_ITERATIONS=8
MAX_AGENT_CALLS=8
MAX_TOOL_CALLS=20
TASK_TIMEOUT_SECONDS=120
```

See the example environment file for the full list, and the config
module inside the framework package for the settings model itself.

## 15. Ollama

Ollama is a first-class provider (local or remote, no API key required),
but never a hard dependency of the framework -- it's just the default in
the example environment file. Point AI_BASE_URL at a remote Ollama server
if you're not running one locally. Requires the Ollama server to be
running separately; this framework does not manage the Ollama process
itself.

## 16. Creating a new agent

```python
from ai_agent_framework import Agent, AgentCapability, AgentResult, Permission

class MyAgent(Agent):
    name = "my_agent"
    description = "What this agent does, for the supervisor's benefit"
    capabilities = [AgentCapability(name="some_capability", description="...")]
    tools = [my_tool]
    permission = Permission.READ

    async def run(self, task: str, context) -> AgentResult:
        # use context.tool_registry, context.memory, context.model, etc.
        return AgentResult.ok("the result")

engine.register_agent(MyAgent())
```

Details are in the agents document under docs.

## 17. Creating a new tool

```python
from pydantic import BaseModel, Field
from ai_agent_framework import tool, Permission

class MyToolArgs(BaseModel):
    value: int = Field(gt=0)

@tool(name="my_tool", description="...", args_schema=MyToolArgs, permission=Permission.READ)
async def my_tool(value: int) -> dict:
    return {"result": value * 2}

engine.register_tool(my_tool)
```

Details are in the tools document under docs.

## 18. Adding a model provider

Subclass ModelProvider, implement generate(), and register it:

```python
from ai_agent_framework.models.base import ModelProvider
from ai_agent_framework.models.factory import register_provider

class MyProvider(ModelProvider):
    provider_name = "my_provider"

    async def generate(self, request):
        ...

register_provider("my_provider", MyProvider)
```

Details are in the models document under docs.

## 19. Integrating with an external project

```python
from ai_agent_framework import AgentEngine, ProjectRegistration

my_project = ProjectRegistration(
    name="my_project",
    agents=[MyAgentA(), MyAgentB()],
    tools=[my_tool_a, my_tool_b],
)

engine = AgentEngine(model=my_model)
my_project.apply(engine)
```

No file inside the framework's core package needs to change for this.
Details are in the integrations document under docs.

## 20. Full example

See the basic example folder for a complete, runnable, dependency-free
walkthrough (two agents, one tool, a full engine run). Details are in the
examples document under docs.

## 21. Dalalti integration example

See the dalalti example folder for an illustrative (non-connected)
example showing how Dalalti-style product/order/customer agents and
tools would register with the framework via a project registration,
without any Dalalti-specific logic inside the framework's core.

## Testing

```bash
pytest
ruff check .
mypy .
```

The test suite uses mocked HTTP transports for model providers and
in-memory fakes for agents/tools, so it never makes a real network call
or requires a running Ollama server or API key. See the development
report for what's covered and how to run everything, including a note on
verifying in the specific environment this project was delivered from.

## Project layout

```
AI_Agent_Orchestration_Framework/
├── src/ai_agent_framework/   the framework package
├── tests/                    the test suite
├── examples/                 basic, dalalti, and legacy-migration examples
├── docs/                     architecture and per-concept documentation
├── server.py                 runnable FastAPI entrypoint (dev/reference)
├── pyproject.toml            packaging + tool configuration
├── requirements.txt
├── .env.example
└── DEVELOPMENT_REPORT.md     old vs new architecture, decisions, migration notes
```

## License

MIT (see pyproject.toml). Adjust to your organization's actual license
before distributing.
# AI_Agent_Orchestration_Framework
