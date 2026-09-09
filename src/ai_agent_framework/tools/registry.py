"""
Tool registry.

Holds the set of tools available to an execution (globally, or scoped to
a particular agent/project via :meth:`ToolRegistry.scoped`), performs
permission checks before dispatching a call, and turns the result into a
:class:`~ai_agent_framework.tools.base.ToolResult`.
"""

from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

from ai_agent_framework.core.errors import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    ToolNotFoundError,
)
from ai_agent_framework.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from ai_agent_framework.core.context import ExecutionContext
    from ai_agent_framework.security.permissions import PermissionSet


class ToolRegistry:
    """Registers tools and mediates permission-checked execution."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        """Register a tool. Re-registering the same name overwrites it."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-style function definitions for every registered tool.

        This is what gets handed to a :class:`~ai_agent_framework.models.base.ModelProvider`
        so the model can decide which tool (if any) to call.
        """
        return [t.to_definition() for t in self._tools.values()]

    def scoped(self, names: Iterable[str]) -> "ToolRegistry":
        """Return a new registry containing only the named tools.

        Used to give an individual agent a restricted view of the full
        tool set (its declared ``tools``) rather than every tool the host
        application has registered globally.
        """
        return ToolRegistry(self.get(name) for name in names)

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        permissions: "PermissionSet | None" = None,
        context: "ExecutionContext | None" = None,
        allow_confirmation_bypass: bool = False,
    ) -> ToolResult:
        """Validate permissions, then execute the named tool.

        Raises:
            ToolNotFoundError: if ``name`` isn't registered.
            PermissionDeniedError: if ``permissions`` doesn't cover the
                tool's required permission.
            ConfirmationRequiredError: if the tool is marked
                ``requires_confirmation`` and ``allow_confirmation_bypass``
                is not set. Host applications catch this, obtain
                confirmation from a human, and re-invoke with
                ``allow_confirmation_bypass=True``.
        """
        t = self.get(name)

        if permissions is not None and not permissions.allows(t.permission):
            raise PermissionDeniedError(name, str(t.permission), permissions)

        if t.requires_confirmation and not allow_confirmation_bypass:
            raise ConfirmationRequiredError(name, arguments)

        return await t.execute(arguments, context=context)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
