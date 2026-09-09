"""
The ``@tool`` decorator: the ergonomic way to define a :class:`Tool`.

Example::

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
        return {"id": "...", "name": name, "price": price}

If ``args_schema`` is omitted, the framework has no way to validate or
advertise a precise parameter schema for the tool -- it will fall back to
an open schema. Supplying ``args_schema`` is strongly recommended for any
tool exposed to a model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ai_agent_framework.security.permissions import Permission
from ai_agent_framework.tools.base import Tool, ToolFunc


def tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel] | None = None,
    permission: Permission | str = Permission.READ,
    requires_confirmation: bool = False,
    metadata: dict[str, Any] | None = None,
):
    """Decorator that turns a function into a :class:`Tool`.

    The decorated function itself remains callable directly (useful in
    tests); the returned object is a ``Tool`` instance, not the raw
    function, so it can be passed straight to
    :meth:`~ai_agent_framework.tools.registry.ToolRegistry.register`.
    """

    def decorator(func: ToolFunc) -> Tool:
        return Tool(
            func,
            name=name,
            description=description,
            args_schema=args_schema,
            permission=permission,
            requires_confirmation=requires_confirmation,
            metadata=metadata,
        )

    return decorator
