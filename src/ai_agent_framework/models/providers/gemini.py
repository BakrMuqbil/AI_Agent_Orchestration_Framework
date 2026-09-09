"""
Google Gemini provider.

Talks to the Gemini ``generateContent`` REST API. Gemini's wire format
differs from the OpenAI-style chat completions shape (roles are
"user"/"model" instead of "user"/"assistant", tool definitions use a
different envelope, etc.), so this provider translates the framework's
provider-agnostic :class:`ModelRequest`/``ModelResponse`` to and from
Gemini's native format rather than subclassing the OpenAI-compatible
provider.
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

_ROLE_MAP = {"assistant": "model", "user": "user", "system": "user", "tool": "user"}


class GeminiProvider(ModelProvider):
    """Provider for the Google Gemini API."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        **_: Any,
    ) -> None:
        super().__init__(model=model)
        if not api_key:
            raise ModelProviderError("gemini", "api_key is required for GeminiProvider")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _to_gemini_contents(messages: list[ModelMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        """Split out system instruction and convert the rest to Gemini "contents"."""
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = _ROLE_MAP.get(m.role, "user")
            contents.append({"role": role, "parts": [{"text": m.content}]})
        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        declarations = []
        for t in tools:
            fn = t.get("function", t)
            declarations.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return [{"functionDeclarations": declarations}]

    async def generate(self, request: ModelRequest) -> ModelResponse:
        system_instruction, contents = self._to_gemini_contents(request.messages)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        gemini_tools = self._to_gemini_tools(request.tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config
        payload.update(request.extra)

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
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
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
        except (KeyError, IndexError) as exc:
            raise ModelProviderError(
                self.provider_name, f"unexpected response shape: {data}"
            ) from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for i, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(id=f"gemini-call-{i}", name=fc.get("name", ""), arguments=fc.get("args", {}))
                )

        return ModelResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=data,
            model=self.model,
            provider=self.provider_name,
        )
