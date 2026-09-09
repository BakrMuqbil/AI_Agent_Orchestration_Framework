"""
Model provider abstraction.

The framework never talks to a specific LLM API directly from core code.
Every LLM call goes through a :class:`ModelProvider`, so the same agent
and supervisor logic works unmodified against Ollama, OpenAI, Gemini, or
any future provider.

Two shapes of response are supported:

* Free-form text completion, optionally with tool/function calls
  (:class:`ModelResponse`).
* Structured output validated against a Pydantic schema
  (:meth:`ModelProvider.generate_structured`), so callers don't have to
  hand-roll fragile text parsing to get JSON out of a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class ModelMessage:
    """A single chat message in provider-agnostic form."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ModelRequest:
    """A provider-agnostic request to generate a completion."""

    messages: list[ModelMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """A provider-agnostic completion result."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    model: str | None = None
    provider: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ModelProvider:
    """Base interface every model provider implements.

    Subclasses must implement :meth:`generate`. ``generate_structured`` has
    a working default implementation built on top of ``generate`` (it asks
    the model for JSON matching the schema and validates the result), but
    providers with native structured-output support may override it.
    """

    #: Short, stable identifier used in configuration (e.g. ``AI_PROVIDER=ollama``).
    provider_name: str = "base"

    def __init__(self, *, model: str, **_: Any) -> None:
        self.model = model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a completion for the given request.

        Must be implemented by concrete providers. Implementations should
        raise :class:`~ai_agent_framework.core.errors.ModelProviderError`
        on failure rather than letting a provider-specific exception
        (an HTTP error, an SDK exception, ...) escape.
        """
        raise NotImplementedError

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Generate a completion and validate it against a Pydantic schema.

        Default implementation: append an instruction to respond with only
        JSON matching the schema, call :meth:`generate`, strip Markdown
        code fences if present, and validate. Providers with native JSON
        mode / structured output support should override this for
        reliability.
        """
        import json

        schema_json = schema.model_json_schema()
        instruction = ModelMessage(
            role="system",
            content=(
                "Respond with ONLY a single JSON object matching this JSON "
                f"Schema, and nothing else (no prose, no Markdown fences):\n"
                f"{json.dumps(schema_json)}"
            ),
        )
        augmented = ModelRequest(
            messages=[instruction, *request.messages],
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tool_choice="none",
            extra=request.extra,
        )
        response = await self.generate(augmented)
        text = response.content.strip()
        if "```" in text:
            # Strip a leading ```json / ``` fence and trailing ``` fence.
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    text = candidate
                    break

        from ai_agent_framework.core.errors import ModelProviderError

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelProviderError(
                self.provider_name, f"model did not return valid JSON: {exc}"
            ) from exc

        try:
            return schema(**data)
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                self.provider_name, f"model output failed schema validation: {exc}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(model={self.model!r})"
