"""
Tool interface.

A :class:`Tool` wraps a callable (sync or async) with the metadata needed
to expose it to a model (name, description, JSON schema for arguments),
validate its arguments before execution, enforce the permission required
to call it, and produce a uniform :class:`ToolResult`.

Tools are provider-independent: the same ``Tool`` object can be offered to
an Ollama, OpenAI, or Gemini model, since each model provider is
responsible for translating the tool's schema into whatever format that
provider's function-calling API expects.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ai_agent_framework.core.errors import ToolExecutionError, ToolValidationError
from ai_agent_framework.security.permissions import Permission

if TYPE_CHECKING:
    from ai_agent_framework.core.context import ExecutionContext

ToolFunc = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class ToolResult:
    """The uniform result of executing a tool."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class Tool:
    """A callable tool with metadata, schema validation, and permissions.

    Prefer constructing tools with the :func:`~ai_agent_framework.tools.decorator.tool`
    decorator; this class can also be instantiated directly for cases
    (dynamic tools, tools built from external specs) where the decorator
    is not a good fit.
    """

    def __init__(
        self,
        func: ToolFunc,
        *,
        name: str,
        description: str,
        args_schema: type[BaseModel] | None = None,
        permission: Permission | str = Permission.READ,
        requires_confirmation: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.permission = permission
        self.requires_confirmation = requires_confirmation
        self.metadata = metadata or {}
        self._is_async = inspect.iscoroutinefunction(func)

    # -- schema -----------------------------------------------------------

    def parameters_schema(self) -> dict[str, Any]:
        """Return a JSON-schema-shaped dict describing this tool's arguments.

        Used by model providers to advertise the tool's function-calling
        signature. Falls back to an open object schema when no
        ``args_schema`` was supplied.
        """
        if self.args_schema is not None:
            return self.args_schema.model_json_schema()
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def to_definition(self) -> dict[str, Any]:
        """Return the OpenAI-style function definition for this tool.

        Model providers may adapt this further for provider-specific
        formats, but the OpenAI function-calling shape is the closest
        thing to a lingua franca across Ollama/OpenAI/Gemini-compatible
        APIs, so it's used as the canonical intermediate representation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    # -- validation ---------------------------------------------------------

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate raw arguments against ``args_schema``, if any.

        Returns the validated (and coerced) arguments as a plain dict.
        Raises :class:`ToolValidationError` on failure.
        """
        if self.args_schema is None:
            return arguments
        try:
            validated = self.args_schema(**arguments)
        except ValidationError as exc:
            raise ToolValidationError(self.name, str(exc)) from exc
        return validated.model_dump()

    # -- execution ----------------------------------------------------------

    async def execute(
        self,
        arguments: dict[str, Any],
        context: "ExecutionContext | None" = None,
    ) -> ToolResult:
        """Validate arguments and execute the underlying function.

        Permission checks are performed by the caller (typically the
        orchestration engine or agent runtime), which has access to the
        execution context's granted permissions -- ``Tool.execute`` itself
        only validates and runs.
        """
        started = time.monotonic()
        try:
            validated = self.validate_arguments(arguments)
        except ToolValidationError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                duration_seconds=time.monotonic() - started,
            )

        try:
            if self._accepts_context():
                if self._is_async:
                    output = await self.func(context=context, **validated)
                else:
                    output = self.func(context=context, **validated)
            else:
                if self._is_async:
                    output = await self.func(**validated)
                else:
                    output = self.func(**validated)
        except Exception as exc:  # noqa: BLE001 - normalized into ToolResult
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(ToolExecutionError(self.name, exc)),
                duration_seconds=time.monotonic() - started,
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output,
            duration_seconds=time.monotonic() - started,
        )

    def _accepts_context(self) -> bool:
        try:
            sig = inspect.signature(self.func)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return False
        return "context" in sig.parameters

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Tool(name={self.name!r}, permission={self.permission!r})"
