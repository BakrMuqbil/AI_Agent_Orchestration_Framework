"""
Ollama provider.

Talks to a local (or remote) Ollama server's ``/api/chat`` endpoint.
Ollama is a first-class, well-supported provider in this framework, but
it is never a hard requirement: the framework depends only on the
:class:`~ai_agent_framework.models.base.ModelProvider` interface, and
which provider is active is entirely a matter of configuration
(``AI_PROVIDER=ollama``).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai_agent_framework.core.errors import ModelProviderError
from ai_agent_framework.models.base import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ToolCall,
)


class OllamaProvider(ModelProvider):
    """Provider for a local or remote Ollama server."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **_: Any,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _to_wire_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            wire.append(entry)
        return wire

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire_messages(request.messages),
            "stream": False,
        }
        if request.tools:
            payload["tools"] = request.tools

        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        payload.update(request.extra)

        url = f"{self.base_url}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ModelProviderError(
                self.provider_name,
                f"could not reach Ollama server at {self.base_url}: {exc}",
            ) from exc

        if response.status_code != 200:
            raise ModelProviderError(
                self.provider_name,
                f"HTTP {response.status_code}: {response.text[:500]}",
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelProviderError(self.provider_name, f"invalid JSON response: {exc}") from exc

        if "error" in data:
            raise ModelProviderError(self.provider_name, str(data["error"]))

        message = data.get("message", {})
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

        return ModelResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            raw=data,
            model=data.get("model", self.model),
            provider=self.provider_name,
        )
