"""
Provider for any API that speaks the OpenAI ``/chat/completions`` wire
format -- this covers OpenAI itself, OpenRouter, and many self-hosted or
third-party gateways. Provider-specific subclasses (``OpenAIProvider``)
mostly just fix the base URL and default headers.

This provider makes no assumption about which underlying model is used;
the model name is passed straight through, so it also works for the many
open models OpenRouter/compatible gateways expose.
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


class OpenAICompatibleProvider(ModelProvider):
    """Generic provider for OpenAI-wire-format chat completion APIs."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _to_wire_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            wire.append(entry)
        return wire

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire_messages(request.messages),
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        payload.update(request.extra)

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise ModelProviderError(self.provider_name, f"request failed: {exc}") from exc

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
            err = data["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            raise ModelProviderError(self.provider_name, message)

        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise ModelProviderError(
                self.provider_name, f"unexpected response shape: {data}"
            ) from exc

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=tc["function"]["name"], arguments=args)
            )

        return ModelResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            raw=data,
            model=data.get("model", self.model),
            provider=self.provider_name,
        )
