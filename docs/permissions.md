# Permissions

## The model

Tools declare the permission they require to run. Agents (or, when a host
application wants per-user/per-session scoping instead, the execution
context) are granted a PermissionSet. Before a tool executes, the
orchestration layer checks that the granted set covers the tool's required
permission.

```python
from ai_agent_framework import Permission, PermissionSet

class Permission(IntEnum):
    READ = 10
    CREATE = 20
    UPDATE = 30
    DELETE = 40
    ADMIN = 100
```

Levels are cumulative and hierarchical: a PermissionSet granted at UPDATE
also satisfies READ and CREATE checks, but not DELETE or ADMIN.

```python
perms = PermissionSet(level=Permission.CREATE)
perms.allows(Permission.READ)     # True
perms.allows(Permission.DELETE)   # False

PermissionSet.admin()       # allows everything
PermissionSet.read_only()   # allows only READ
```

## Named (non-hierarchical) permissions

Some permissions don't fit a hierarchy -- "can send email" isn't more or
less privileged than "can issue refunds," they're just different
capabilities. Use named grants for these:

```python
perms = PermissionSet(level=Permission.READ)
perms.grant_named("send_email")

perms.allows("send_email")     # True
perms.allows("delete_account") # False
```

A tool declares a named requirement the same way it declares a
hierarchical one: permission="send_email" instead of
permission=Permission.CREATE.

## Enforcement point

ToolRegistry.invoke() is where permission checks actually happen:

```python
result = await registry.invoke(
    "delete_product",
    {"product_id": "..."},
    permissions=context.permissions,
)
# raises PermissionDeniedError if context.permissions doesn't cover
# the tool's required permission
```

If an agent calls tool_registry.invoke(...) without passing permissions=,
no permission check happens -- always pass context.permissions so
enforcement is active for real usage. Tests that want to bypass
enforcement can omit it deliberately.

## Confirmation for sensitive operations

Some operations should never happen without an explicit human
confirmation, no matter how strong the caller's permissions are. Mark
these tools requires_confirmation=True:

```python
@tool(
    name="delete_customer_account",
    description="Permanently delete a customer account",
    permission=Permission.DELETE,
    requires_confirmation=True,
)
async def delete_customer_account(customer_id: str) -> dict:
    ...
```

Calling such a tool raises ConfirmationRequiredError(tool_name, arguments)
unless the caller passes allow_confirmation_bypass=True. The intended flow
for a host application:

1. Catch ConfirmationRequiredError.
2. Show the human a confirmation prompt with exc.tool_name and
   exc.arguments.
3. If confirmed, re-invoke with allow_confirmation_bypass=True.
4. If declined, do not re-invoke.

## Errors

| Error | When |
|---|---|
| PermissionDeniedError | Granted permissions don't cover the tool's requirement |
| ConfirmationRequiredError | Tool requires human confirmation and it wasn't given |

Both inherit from FrameworkError, so a host application can catch either
specifically or FrameworkError generally.
