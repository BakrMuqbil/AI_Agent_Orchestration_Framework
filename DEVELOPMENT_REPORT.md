# Development Report: AI Agent Workflow Automation System to AI Agent Orchestration Framework

## 1. Old architecture (audit findings)

The uploaded project ("Techno-Bakr AI Agent" / AI Agent Workflow
Automation System) was a single-agent FastAPI application, not a
framework. Full structure, described by responsibility:

- An application entrypoint wiring the FastAPI app, CORS, and a single
  router mount.
- An API layer with exactly two endpoints: a chat endpoint and a health
  check.
- A settings module using pydantic-settings, but mixing framework-level
  fields (which model provider) with app-level secrets (Trello board and
  list IDs, SMTP host/user/password, provider API keys) in one settings
  class.
- A logging setup module using standard library logging directly, with no
  event abstraction a host application could subscribe to.
- An agent-service module of roughly 150 lines implementing one function
  that did everything: prompt building, a single model call, manual
  tool-call JSON parsing, dispatching exactly one tool per turn, and
  response formatting, all interleaved in one place.
- An LLM-service module making direct HTTP calls to OpenRouter with a
  Gemini fallback path hardcoded inline -- not an abstraction other
  providers could plug into, just two providers' logic interleaved in one
  file.
- A tools module with seven hardcoded functions (send an email over SMTP,
  create an operational ticket via the Trello REST API, prioritize tasks,
  generate a session summary, query a small knowledge base, get the
  current time, calculate a discount), dispatched via a chain of
  if/elif branches on the tool's name string -- no schema, no registry, no
  permission system.
- A memory-service module: a plain dictionary keyed by user id holding a
  list of conversation turns, trimmed by slicing. No interface, so no
  alternate backend was possible without rewriting every call site.
- A vector-service module implementing word-frequency-based "RAG":
  tokenize a query and a set of documents, score by shared-word count,
  and read/write a flat text file as the document source.
- An unrelated, unused schema file (a sentiment-analysis model with no
  connection to anything else in the app -- dead code).
- A prompt-builder utility doing string formatting for the single agent's
  system prompt.
- An empty test file (zero actual test functions).
- A manual, non-pytest smoke script at the repository root that printed
  results to stdout and asserted nothing.
- A flat text file supplying the "RAG" content.
- A requirements file listing fastapi, uvicorn, pydantic, httpx,
  python-dotenv, an unused openai package reference, and requests.
- A Dockerfile and a Vercel deployment config, both specific to deploying
  this one application.

### What existed

- A working single tool-calling loop against OpenRouter, with a Gemini
  fallback path.
- Seven real, useful tool implementations (email, Trello ticketing,
  pricing, time, prioritization, summarization, keyword retrieval).
- A FastAPI app with CORS and structured logging.
- pydantic-settings-based configuration (mixed concerns, but a real
  settings object).
- A minimal but functional word-overlap retrieval mechanism.

### What could be kept

The business logic inside every tool function, the FastAPI-based
approach, the pydantic-settings configuration pattern, and the general
idea of word-overlap retrieval as a dependency-free RAG default -- all of
this was preserved, just repackaged behind proper interfaces (see
section 3).

### What needed refactor

- The single agent function combined five responsibilities (prompt
  building, model call, tool-call parsing, tool dispatch, response
  formatting) with no seams for a second agent, a second model provider,
  or a second tool source.
- Tool dispatch was a chain of string-equality checks -- no schema
  validation, no permission checks, no way for an external project to add
  a tool without editing that function directly.
- The LLM-service module had OpenRouter and Gemini logic interleaved in
  the same file with no shared interface -- adding Ollama would have
  meant a third branch, not a third provider class.
- Memory was a raw dictionary with no abstraction -- unusable outside
  this exact app.
- Configuration mixed framework-level settings (model provider) with
  app-level secrets (Trello IDs, SMTP credentials) in one settings class.

### What needed redesign entirely

- There was no multi-agent concept at all -- one function handled
  everything. This became `Agent` plus `AgentRegistry` plus `Supervisor`.
- There was no permission system -- any tool could always run. This
  became `Permission`/`PermissionSet` plus a confirmation-required flag.
- There was no event or observability abstraction -- only direct logging
  calls inside business logic. This became `Event`/`EventBus`.
- There was no execution-limit or safety mechanism -- nothing prevented
  an unbounded loop if the model kept requesting tool calls. This became
  `SupervisorConfig` with iteration, agent-call, tool-call, and timeout
  limits.

### What was removed and why

- The unrelated sentiment schema -- dead code with no connection to the
  rest of the app; not carried into the new project.
- The root-level manual smoke script -- replaced by a real pytest suite;
  its intent (verify ticket-creation-shaped logic works) is preserved
  conceptually in the legacy migration example, but the script itself
  asserted nothing and was not portable into a test suite.
- The flat text file backing the old retrieval mechanism -- app-specific
  content, not framework material. The retrieval *mechanism* that read it
  (word-overlap scoring) was kept and generalized into the framework's
  simple retriever; the specific file's contents were not.
- Trello/SMTP-specific configuration fields -- removed from the
  framework's settings class since they are host-application secrets, not
  framework configuration. A real migration keeps them in the host
  application's own settings, loaded independently.

## 2. New architecture

```
src/ai_agent_framework/
├── core/            Task, TaskResult, TaskStatus, ExecutionContext,
│                     Event, EventBus, EventType, FrameworkError hierarchy
├── agents/           Agent, AgentCapability, AgentResult, AgentRegistry
├── tools/             Tool, ToolResult, tool decorator, ToolRegistry
├── models/
│   ├── base            ModelProvider, ModelRequest, ModelResponse, ToolCall,
│   │                    generate_structured default implementation
│   ├── factory          create_provider, register_provider
│   └── providers/
│       ├── ollama              OllamaProvider
│       ├── openai_compatible   OpenAICompatibleProvider (base for OpenAI-wire APIs)
│       ├── openai               OpenAIProvider
│       └── gemini                GeminiProvider
├── memory/            MemoryStore protocol, InMemoryStore, ConversationMemory,
│                       AgentStateMemory
├── rag/                Document, DocumentStore, Retriever, EmbeddingProvider
│                        protocols; InMemoryDocumentStore, SimpleRetriever
│                        (keyword-overlap by default, cosine similarity if an
│                        embedding provider is supplied), context builder
├── security/           Permission (hierarchical), PermissionSet
├── orchestration/
│   ├── supervisor        Supervisor, SupervisorConfig, Planner interface,
│   │                      ModelPlanner, RuleBasedPlanner, PlanDecision
│   ├── engine             AgentEngine (top-level facade)
│   ├── project             ProjectRegistration (external-project integration)
│   └── strategies          run_sequential, run_parallel
└── api/
    ├── app                create_app(engine) factory
    ├── routes              task/agent/tool/status endpoints
    └── schemas              request/response models

tests/            125 test functions across 10 files, covering every module above
examples/
├── basic/                  minimal end-to-end example, no LLM required
├── dalalti/                 external-project integration example
└── legacy_techno_bakr/      migrated version of the original app's tools

docs/             12 focused documents covering architecture, agents,
                  supervisor, tools, models, memory, context, orchestration,
                  permissions, api, integrations, and examples

server module     runnable FastAPI entrypoint wiring config, provider, engine, and app together
pyproject.toml    packaging (src layout), dependencies, lint/type/test config
requirements.txt  pip-installable dependency list
.env.example      every configuration variable, no real secrets
```

### Design principle applied throughout

Every module under the framework's source package was checked against one
question: does this need to know Dalalti, or e-commerce, or any specific
business domain exists? Everywhere the answer was yes, that logic was
moved to the examples directory instead of the source package. The three
example subdirectories exist specifically to prove this boundary holds
without leaving the framework itself empty of working demonstrations.

## 3. Files changed

Every file in the original project was either directly ported (logic
preserved, packaging changed) or intentionally left out (see section 1,
"What was removed and why"). Nothing was silently dropped without being
accounted for above.

| Original file | Fate |
|---|---|
| Application entrypoint | Superseded by the new server module plus the API app factory (factory pattern instead of one global app) |
| API routes module | Superseded by the new routes module (general task/agent/tool routes instead of a fixed chat endpoint) |
| Settings module | Superseded by the framework's config module (framework-only settings; app secrets belong in the host application) |
| Logger module | Superseded by the core events module (provider-independent event bus; a host app wires its own logging via an event handler) |
| Agent-service module | Logic split across the agent base class, the supervisor, and the engine |
| LLM-service module | Superseded by the model provider base interface plus the concrete provider modules (one interface, three concrete providers instead of one file with two providers interleaved) |
| Tools module | Ported into the legacy migration example using the tool decorator and permission system |
| Memory-service module | Superseded by the memory base module and the in-memory store implementation |
| Vector-service module | Superseded by the RAG retriever and document-store modules |
| Unrelated sentiment schema | Removed (dead code, unrelated to the app) |
| Prompt-builder utility | Superseded by the model planner's prompt construction inside the supervisor module |
| Empty API test file | Replaced by a real test suite (125 test functions, 10 files) |
| Root-level manual smoke script | Removed (manual script, not a test); intent preserved in the legacy migration example |
| Flat text file backing retrieval | Removed (app-specific content) |
| Requirements file | Replaced with framework dependencies (fastapi, uvicorn, pydantic, pydantic-settings, httpx, pytest, pytest-asyncio, ruff, mypy) |
| Docker and Vercel deployment config | Not carried over -- deployment config is host-application-specific; add your own once you've decided how you're hosting the framework plus your project |

## 4. New files

All 57 Python files under the source, tests, and examples directories,
plus 12 files under the docs directory, the main README, the packaging
file, the requirements file, the example environment file, the git-ignore
file, and the server module are new, replacing the roughly 15-file
original project. See the architecture tree in section 2 for the full
layout.

## 5. Supervisor architecture

The supervisor's run method executes a bounded loop:

1. Calls the planner's decide method with the task, the agent registry,
   and the execution context.
2. If the decision is to call an agent, dispatches to the named agent
   with a child execution context, records the agent's result into both
   the task result's steps and the context's history, and loops back to
   step 1 so the next planning call can see what happened.
3. If the decision is to finish, returns a completed task result with the
   decision's final output.
4. If the decision is to fail, returns a failed task result with the
   decision's reasoning as the error.
5. Any other decision shape, an unknown agent name, or an agent that
   raises is normalized into a failed step or a failed task rather than
   an unhandled exception escaping the run method.
6. The maximum-iterations and maximum-agent-calls limits are checked
   every loop; exceeding either raises a loop-limit error internally,
   which the run method catches and turns into a failed task result
   describing which limit was exceeded.

Decision-making itself is delegated to a planner:

- The model-backed planner asks a configured model provider for a
  structured plan decision, given the task goal, every registered
  agent's description and capabilities, and the execution history so
  far. This is what gives the supervisor genuine multi-step reasoning
  about goal completion.
- The rule-based planner is a deterministic, keyword-overlap-based
  fallback used in tests, in the bundled examples (so they run without
  an LLM), and as a safe default when the engine is constructed without a
  model.

## 6. Agent lifecycle

Construction (validates that a name is set) leads to registration (via
the agent registry or the engine's register-agent method) leads to
selection (a planner names the agent in a plan decision) leads to
dispatch (the supervisor builds a child context and calls the agent's run
method) leads to result recording (the agent's result becomes a step
appended to both the task result and the context's history) and,
optionally, unregistration at runtime.

## 7. Tool system

The tool decorator (name, description, args schema, permission,
confirmation flag, metadata) wraps a sync or async function into a tool
object. The tool's execute method validates arguments against the
Pydantic args schema (or falls back to an open schema if none is given),
runs the function, and returns a uniform result object -- every failure
mode (validation error, runtime exception) is captured as a failed result
rather than an exception escaping. The tool registry's invoke method adds
the permission check and the confirmation-required check on top of the
tool's own execute method, and the registry's definitions method produces
OpenAI-style function definitions for any model provider to consume.

## 8. Model provider system

One abstract model-provider interface (generate, generate-structured)
with three concrete implementations (Ollama, an OpenAI-compatible base
used both directly and by a thin OpenAI-specific subclass, and Gemini),
each translating the framework's provider-agnostic request and response
types to and from its own wire format. A configuration-driven factory
function instantiates a provider by name; a registration function lets a
host application add a custom provider without editing framework code.
The generate-structured method has a working default implementation (ask
for JSON matching a schema, strip Markdown fences, validate) usable by
any provider without per-provider special-casing, which is what the
model-backed planner relies on.

## 9. Memory architecture

A memory-store protocol (get, set, delete, append, get-list, each
namespaced) is the storage interface; an in-memory store is the default,
process-local implementation. A conversation-memory class wraps a memory
store with message-history and durable-summary-event operations, bounded
by configurable history and summary limits. An agent-state-memory class
gives each agent a namespaced key/value scratch space so two agents'
state never collides. Task-scoped state deliberately lives on the
execution context's history and the task result, not in the memory
store, since it is transient by nature and scoped to one task's run.

## 10. RAG architecture

Document, document-store, retriever, and embedding-provider are defined
as protocols. An in-memory document store and a simple retriever are the
bundled defaults: the simple retriever uses cosine similarity over
term-frequency vectors (the same word-overlap approach the original
project used) when no embedding provider is supplied, or real cosine
similarity over embeddings when one is. A context-builder function
formats retrieved chunks into a prompt-ready string. A host application
that needs a real vector database implements the document-store and
retriever protocols against its database of choice; no other framework
code needs to change, since both implementations satisfy the same
protocol.

## 11. Permission system

Permission is a hierarchical enum (read, create, update, delete, admin,
in increasing order); a permission set also supports named,
non-hierarchical grants for capabilities that do not fit a CRUD-shaped
hierarchy. The tool registry's invoke method checks that the caller's
permission set covers the tool's required permission before running it,
raising a permission-denied error on failure, and checks the tool's
confirmation-required flag, raising a confirmation-required error unless
the caller explicitly bypasses it -- giving a host application a clean
catch point to insert a human-confirmation step before a sensitive tool
call proceeds.

## 12. API

An app-factory function builds a FastAPI app with routes mounted under an
API prefix: run a task through the supervisor, list agents, list tools,
and a status endpoint. The engine is injected via FastAPI's dependency
override mechanism rather than imported as a module-level global, so the
same route definitions work for any host application's engine. The
bundled server module is the reference and development entrypoint, wiring
settings, provider creation, engine construction, and app creation
together.

## 13. Testing

125 test functions across 10 files:

| File | Focus |
|---|---|
| Tools test module | Tool definition, schema validation, execution, tool registry, permission set |
| Agents test module | Agent base class, agent registry |
| Memory test module | In-memory store, conversation memory, agent-state memory |
| Model providers test module | Ollama/OpenAI/Gemini providers (mocked HTTP), structured output, provider factory |
| Supervisor test module | Planning cycle, multi-agent execution, loop/safety limits, error handling, events |
| Engine and project test module | Agent engine facade, project registration |
| RAG test module | Document store, retriever, context builder |
| Context test module | Execution context |
| Errors and events test module | Error hierarchy, event bus |
| API test module | FastAPI routes |
| Strategies test module | Sequential and parallel execution helpers |

All model-provider tests use a mocked HTTP transport -- no real network
call is ever made. All agent/supervisor tests use fakes defined in a
shared fixtures module (a fake model provider, an echo agent, a failing
agent, a raising agent) -- no real LLM or API key is required to run the
suite.

### Running the tests in your environment

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy .
```

Important, and stated plainly: the sandbox this project was built in has
no network egress, so a package install could not be run here and the
suite has not been executed against real installed dependencies (fastapi,
pydantic, httpx, and pytest were all unavailable). Every file was
verified in this environment via Python's own compile step (all 57 files
compile with no syntax errors) and a custom AST-based cross-check
confirming every internal import, in both the framework and the test
suite, resolves to a real module and a real exported name (no typos, no
missing exports). This is strong evidence of correctness but is not a
substitute for actually running the test suite -- please run the three
commands above in your own environment before relying on this in
production, and treat anything that fails as a real bug to fix.

### Running the FastAPI app

```bash
cp .env.example .env
uvicorn server:app --reload --port 8000
```

Edit the AI provider, model, and API key values in the environment file
first. This was likewise not executed live in this sandbox for the same
network-access reason; the code path was traced manually and is exercised
indirectly by the API test module, which builds the same app through an
in-process ASGI transport.

## 14. How to run the project

```bash
git clone <this project>
cd AI_Agent_Orchestration_Framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # fill in the AI provider, model, and API key

pytest                  # run the test suite
ruff check .             # lint
mypy .                   # type-check

python examples/basic/run_example.py            # no LLM required
python examples/dalalti/integration_example.py  # no LLM required
uvicorn server:app --reload --port 8000          # run the API
```

## 15. Usage example

```python
from ai_agent_framework import AgentEngine, create_provider

provider = create_provider("ollama", model="llama3.1")
engine = AgentEngine(model=provider)

engine.register_agent(ResearchAgent())
engine.register_agent(ProductAgent())
engine.register_agent(ValidatorAgent())

result = await engine.run("Analyze this product and prepare a recommendation")
print(result.status, result.output)
```

## 16. External-project integration example

```python
from ai_agent_framework import AgentEngine, ProjectRegistration

dalalti = ProjectRegistration(
    name="dalalti",
    agents=[ProductAgent(), OrderAgent(), CustomerAgent()],
    tools=[dalalti_lookup_product, dalalti_create_order, dalalti_lookup_customer],
)

engine = AgentEngine(model=provider)
dalalti.apply(engine)

result = await engine.run("look up product prod_001")
```

The full runnable version lives in the Dalalti example directory. No file
under the framework's source package was touched to make this work.

## 17. Known limitations

- Tests are unverified against real dependencies in this environment. See
  section 13 -- this is the most important limitation to resolve before
  production use. Run the test suite yourself; if anything fails, treat
  it as a real bug to fix, not an artifact of the sandbox.
- The simple retriever's keyword-overlap mode is not a real vector
  search. It is a faithful, generalized carry-over of what the original
  project actually did, exposed behind a swappable retriever protocol so
  upgrading to real embeddings or a real vector database requires no
  changes to agent or supervisor code -- but out of the box it is not
  production-grade semantic search.
- The supervisor's configured timeout is declared but not yet enforced as
  a wall-clock cutoff inside the run loop. The iteration and agent-call
  limits are enforced; wrapping the loop body in an asyncio timeout for
  the wall-clock case is a straightforward follow-up (see section 19)
  that was not completed here.
- The Gemini provider's tool-calling support is implemented but less
  battle-tested than the OpenAI-compatible path, since Gemini's
  function-calling response shape was translated from documentation
  rather than against a live API call (no network access in this
  sandbox).
- No persistent memory-store or document-store backend ships beyond the
  in-memory defaults; Redis or database-backed implementations are
  explicitly left to the host application, per the framework's own
  swappable-interface, minimal-built-in-implementation design.
- No authentication or authorization exists at the API layer itself (the
  permission system governs tool execution, not who can call the HTTP
  API) -- a host application fronting this API in production should add
  its own auth middleware.
- Legacy tool business logic (SMTP sending, the Trello REST call) is
  represented as stubs returning canned data in the legacy migration
  example, matching the instruction not to invent fake functionality for
  the new framework, while making clear in comments where a real
  migration plugs in real credentials and HTTP calls.

## 18. Key architecture decisions

- A planner interface separates supervisor mechanics from supervisor
  policy. This was the single most important decision for satisfying "a
  real supervisor, not a router": the loop, limit, and event-emission
  code in the supervisor never changes regardless of whether decisions
  come from an LLM, a deterministic rule, or a host application's own
  custom planner.
- The execution context is a single object threaded everywhere, rather
  than separate parameters for tools, permissions, memory, and events at
  every layer. This keeps agent and tool signatures stable as the
  framework grows -- adding a new cross-cutting concern means adding one
  field to the context, not changing every agent and tool signature in
  the codebase.
- The model request and response types are provider-agnostic dataclasses,
  with each provider responsible for translating to and from its own wire
  format, rather than exposing OpenAI's shape as the "real" one and
  special-casing Ollama and Gemini around it. The OpenAI-compatible
  provider does use the OpenAI shape as its own wire format, since that is
  genuinely what it speaks, but the Gemini and Ollama providers translate
  fully independently.
- Tool failures never raise past a tool's execute method or the
  registry's invoke method except for permission and confirmation
  errors, which are deliberately exceptions rather than failed results,
  since they represent the caller doing something it categorically is not
  allowed to do, distinct from the tool itself failing.
- The project-registration apply method is deliberately a thin
  composition helper, not a plugin system with lifecycle hooks. The
  requirement was integration without core modification, which a simple
  "register these agents and tools" call satisfies; a heavier plugin
  system would have added complexity the stated goal did not call for.
- Framework configuration intentionally excludes any business-specific
  fields (no Trello IDs, no SMTP credentials) -- this was a direct fix
  for the original project's config mixing framework-level and app-level
  concerns in one settings class, and is necessary for the framework to
  be genuinely reusable across projects with different integrations.

## 19. Suggested next steps

1. Run the verification commands in section 13/14 in a networked
   environment and fix anything that surfaces.
2. Enforce the supervisor's configured timeout as an actual wall-clock
   limit inside the run loop.
3. Add a real embedding-backed retriever implementation as an optional
   extra, so RAG has a production-grade path without forcing every user
   to bring their own from scratch.
4. Add a persistent memory-store implementation (Redis is the natural
   first choice given the interface's shape) as an optional extra.
5. Wire real Dalalti agents and tools against the actual database and
   API layer, replacing the canned-data stubs in the Dalalti integration
   example, once you are ready to actually connect this framework to that
   project.
6. Consider adding authentication middleware to the FastAPI layer before
   exposing it beyond local development.
7. If Gemini tool-calling turns out to need adjustment once tested live,
   that is the most likely provider to need a follow-up fix (see the
   limitation noted in section 17).

## 20. Final architecture review (self-assessment against the stated goals)

| Question | Assessment |
|---|---|
| Reusability -- usable with another project? | Yes: project registration plus the Dalalti example demonstrate registering an external project's agents and tools with zero core changes. |
| Extensibility -- new agent without touching core? | Yes: subclassing the agent base class and registering it never requires editing anything under the framework's source package. |
| Provider independence -- swap Ollama, OpenAI, Gemini easily? | Yes: one environment variable, or a single factory call in code; a registration function adds a fourth provider without framework changes. |
| Tool independence -- host project registers its own tools? | Yes: the tool decorator plus the engine's register-tool method, or a project registration's tools list. |
| Multi-agent -- run several agents? | Yes: the agent registry holds many; the supervisor calls as many as the goal needs, one decision at a time, or the execution strategies allow explicit sequential or parallel fan-out. |
| Supervisor -- real supervisor or just a router? | Real supervisor: a multi-iteration plan, dispatch, observe, evaluate cycle with a swappable planner, not a single-shot lookup table. |
| Security -- permissions and execution limits? | Yes: hierarchical and named permissions, a confirmation flag for sensitive tools, and hard stops on iteration and agent-call counts. The configured timeout is declared but not yet enforced (see section 17, and item 2 in section 19). |
| Maintainability -- is the architecture clear? | Each layer (core, agents, tools, models, memory, rag, security, orchestration, api) has one responsibility and depends only on core plus the layers it composes; no circular imports. |
| Testing -- is core testable without real APIs? | Yes: every model-provider test mocks HTTP; every agent and supervisor test uses in-process fakes. No test in the suite requires a network call, an API key, or a running Ollama server. |
