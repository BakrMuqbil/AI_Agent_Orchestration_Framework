"""OpenAI provider -- a thin configuration of :class:`OpenAICompatibleProvider`."""

from __future__ import annotations

from typing import Any

from ai_agent_framework.models.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for the OpenAI API.

    Also usable for any OpenAI-wire-compatible gateway (OpenRouter, Azure
    OpenAI's compatible endpoints, self-hosted gateways) by overriding
    ``base_url`` -- OpenRouter in particular is just this provider pointed
    at ``https://openrouter.ai/api/v1``.
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
