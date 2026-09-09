# Tools

## Defining a tool

```python
from pydantic import BaseModel, Field
from ai_agent_framework import tool, Permission

class CreateProductArgs(BaseModel):
    name: str
    price: float = Field(gt=0)

@tool(
    name="create_product",
    description="Create a product in the catalog",
    args_schema=CreateProductArgs,
    permission=Permission.CREATE,
)
async def create_product(name: str, price: float) -> dict:
    # ... your real logic (DB write, API call, ...) ...
    return {"id": "prod_123", "name": name, "price": price}
```

`@tool(...)` returns a `Tool` instance (not the raw function), ready to
pass to `ToolRegistry.register()` or an `Agent.tools` list. The wrapped
function itself is still callable directly (useful in tests), though
normally you invoke it through the registry so validation and permission
checks apply.

## Schema, validation, and metadata

- `args_schema` -- a Pydantic model describing the tool's parameters.
  Used both to validate incoming arguments (raises `ToolValidationError`
  on failure) and to generate the JSON-schema `parameters` block
  advertised to a model via `Tool.to_definition()`. Omitting it falls
  back to an open `{"type": "object", "additionalProperties": true}`
  schema -- functional, but the model gets no guidance on what arguments
  to send, so supplying a schema is strongly recommended for anything
  exposed to an LLM.
- `permission` -- a `Permission` level (or a named string permission) an
  agent/context must hold to call this tool. See permissions.md.
- `requires_confirmation` -- when `True`, calling the tool raises
  `ConfirmationRequiredError` unless the caller passes
  `allow_confirmation_bypass=True`. Use this for anything destructive or
  hard to undo (deleting data, sending money, sending an email).
- `metadata` -- free-form dict for anything the framework doesn't need to
  interpret (docs links, versioning, owning team, ...).

## `ToolResult`

Every tool call, success or failure, produces a uniform
`ToolResult(tool_name, success, output, error, duration_seconds, metadata)`
-- callers never need to distinguish "tool raised an exception" from "tool
returned an error" from "arguments failed validation"; all three become
`success=False, error="..."` on the result rather than an exception
escaping (`Tool.execute` catches everything internally).

## `ToolRegistry`

```python
from ai_agent_framework import ToolRegistry

registry = ToolRegistry([create_product])
registry.register(another_tool)

definitions = registry.definitions()  # OpenAI-style function definitions,
                                       # ready to pass to ModelRequest(tools=...)

scoped = registry.scoped(["create_product"])  # a registry view restricted
                                               # to just these tool names --
                                               # used to give an agent only
                                               # the tools it declared

result = await registry.invoke(
    "create_product",
    {"name": "Widget", "price": 9.99},
    permissions=context.permissions,
    context=context,
)
```

`invoke()` is the permission-checked entry point:

1. Looks up the tool (`ToolNotFoundError` if missing).
2. Checks `permissions.allows(tool.permission)` (`PermissionDeniedError`
   if not).
3. Checks `tool.requires_confirmation` (`ConfirmationRequiredError` unless
   bypassed).
4. Validates arguments and executes, returning a `ToolResult`.

## Tool discovery

`ToolRegistry.definitions()` is what you hand to a `ModelProvider` so the
model can see which tools exist and choose to call one:

```python
from ai_agent_framework.models.base import ModelRequest, ModelMessage

request = ModelRequest(
    messages=[ModelMessage(role="user", content="Create a $9.99 widget")],
    tools=registry.definitions(),
)
response = await provider.generate(request)
if response.has_tool_calls:
    for call in response.tool_calls:
        result = await registry.invoke(call.name, call.arguments, permissions=context.permissions)
```

## Dynamic / non-decorator tools

For tools built from external specs (an OpenAPI schema, a generated
client) rather than a hand-written function, construct `Tool` directly:

```python
from ai_agent_framework.tools.base import Tool

my_tool = Tool(
    some_callable,
    name="dynamic_tool",
    description="Built from an external spec",
    permission=Permission.READ,
)
```
