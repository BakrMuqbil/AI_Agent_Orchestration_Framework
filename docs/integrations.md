# Integrating with an external project

## The problem this solves

A host application (Dalalti, an e-commerce system, a CRM, an ERP, an
inventory system) needs to add AI agents without forking or modifying the
framework's core. The integration layer is the seam that makes this
possible: define your agents and tools, bundle them, register them with an
engine.

## ProjectRegistration

```python
from ai_agent_framework import AgentEngine, ProjectRegistration

dalalti = ProjectRegistration(
    name="dalalti",
    agents=[ProductAgent(), OrderAgent(), CustomerAgent()],
    tools=[create_product_tool, update_order_tool],
    metadata={"domain": "e-commerce", "stack": "Next.js/Prisma/Supabase"},
)

engine = AgentEngine(model=my_model)
dalalti.apply(engine)
```

The apply method registers every agent and tool onto the engine's existing
registries. It does not copy or wrap them -- afterward, the engine's
agent registry and tool registry simply contain those agents/tools
alongside anything else already registered.

## Multiple projects, one engine

Several projects can be applied to the same engine. Agents/tools coexist
as long as their names do not collide -- a name collision overwrites the
earlier registration, matching how the registries behave generally when
you register the same name twice. If a host application wants guaranteed
isolation between projects sharing one engine, namespace names explicitly:

```python
class ProductAgent(Agent):
    name = "dalalti.product_agent"   # not just "product_agent"
```

## What belongs in the external project vs. the framework

| Belongs in the external project | Belongs in the framework |
|---|---|
| Business logic inside tool functions (DB queries, API calls) | Tool interface, schema validation, permission checks |
| Agent decision logic specific to that domain | Agent interface, registration, dispatch |
| Domain-specific configuration (Trello board IDs, SMTP credentials) | Model provider selection, execution limits |
| Which agents exist for that domain | Supervisor's plan/execute/observe cycle |

If you find yourself wanting to add a domain concept (a "Product" type, an
"Order" status enum) inside the framework's core modules, that is a sign
it belongs in the host application's own code instead. See the Dalalti
example under the examples directory for the intended shape.

## Step by step: adding a new project

1. Define your tools with the tool decorator, giving each a clear name,
   description, args schema, and permission.
2. Define your agents as Agent subclasses, declaring the tools each one
   needs.
3. Bundle them into a ProjectRegistration.
4. Call apply(engine) once, wherever your application constructs its
   AgentEngine.
5. Run tasks with engine.run(goal).

No file inside the framework's core package needs to change for any of
this.

## Migrating an existing single-agent application

The legacy migration example under examples shows how to port an
existing project's tools (email, ticketing, pricing, time) onto the
tool/Permission/Agent primitives, preserving behavior while removing the
coupling that made the original code impossible to reuse elsewhere.
