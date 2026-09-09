"""
Provider factory.

Turns configuration (``AI_PROVIDER=ollama``, ``AI_MODEL=qwen2.5``, ...)
into a concrete :class:`~ai_agent_framework.models.base.ModelProvider`
instance, without any part of the framework's core logic needing to know
which providers exist. Adding a new provider means adding one entry to
``_PROVIDERS`` -- nothing else in the framework changes.
"""

from __future__ import annotations

from typing import Any, Callable

from ai_agent_framework.core.errors import ConfigurationError
from ai_agent_framework.models.base import ModelProvider

_PROVIDERS: dict[str, Callable[..., ModelProvider]] = {}


def register_provider(name: str, factory: Callable[..., ModelProvider]) -> None:
    """Register a provider factory under a configuration name.

    Host applications can call this to add their own custom provider
    (e.g. an internal LLM gateway) without modifying framework code.
    """
    _PROVIDERS[name] = factory


def _register_builtin_providers() -> None:
    if _PROVIDERS:
        return
    from ai_agent_framework.models.providers.ollama import OllamaProvider
    from ai_agent_framework.models.providers.openai import OpenAIProvider
    from ai_agent_framework.models.providers.gemini import GeminiProvider

    _PROVIDERS["ollama"] = OllamaProvider
    _PROVIDERS["openai"] = OpenAIProvider
    _PROVIDERS["gemini"] = GeminiProvider


def create_provider(provider: str, *, model: str, **kwargs: Any) -> ModelProvider:
    """Instantiate the model provider named ``provider``.

    Args:
        provider: The configuration name of the provider, e.g.
            ``"ollama"``, ``"openai"``, ``"gemini"``, or the name of a
            provider registered via :func:`register_provider`.
        model: The model identifier to use with that provider.
        **kwargs: Forwarded to the provider's constructor
            (``api_key``, ``base_url``, ``timeout``, ...).
    """
    _register_builtin_providers()
    key = provider.strip().lower()
    if key not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS)) or "(none registered)"
        raise ConfigurationError(
            f"Unknown model provider '{provider}'. Available providers: {available}"
        )
    return _PROVIDERS[key](model=model, **kwargs)
