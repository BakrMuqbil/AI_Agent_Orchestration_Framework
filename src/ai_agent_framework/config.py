"""
Centralized configuration.

All framework-level configuration is env-var driven via
``pydantic-settings``, with no secrets or business-specific values hard
coded. This module configures the *framework* (which model provider,
which model, execution limits, logging). It knows nothing about any
specific host application's domain configuration (Trello boards, SMTP
accounts, etc.) -- those belong in the host application's own settings,
not here.

Usage::

    from ai_agent_framework.config import FrameworkSettings

    settings = FrameworkSettings()  # reads from environment / .env
    provider = create_provider(settings.ai_provider, model=settings.ai_model, **settings.provider_kwargs())
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrameworkSettings(BaseSettings):
    """Framework-level settings, populated from environment variables / .env."""

    # -- model provider ----------------------------------------------------
    ai_provider: str = Field(default="ollama", description="ollama | openai | gemini | <custom>")
    ai_model: str = Field(default="llama3.1", description="Model identifier for the selected provider")
    ai_api_key: str | None = Field(default=None, description="API key for providers that require one")
    ai_base_url: str | None = Field(default=None, description="Override base URL, e.g. for OpenRouter or a local gateway")
    ai_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    ai_max_tokens: int | None = Field(default=800, ge=1)
    ai_timeout_seconds: float = Field(default=60.0, gt=0)

    # -- supervisor / execution limits --------------------------------------
    max_iterations: int = Field(default=8, ge=1)
    max_agent_calls: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    task_timeout_seconds: float | None = Field(default=120.0)

    # -- memory --------------------------------------------------------------
    memory_max_history: int = Field(default=20, ge=1)
    memory_max_summary_events: int = Field(default=30, ge=1)

    # -- logging / observability ----------------------------------------------
    log_level: str = Field(default="INFO")
    enable_stdout_event_logging: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    def provider_kwargs(self) -> dict[str, Any]:
        """Constructor keyword arguments for ``create_provider(self.ai_provider, ...)``.

        Only connection-level settings belong here (api key, base url,
        timeout) -- per-request generation settings like temperature and
        max_tokens are request-level, not provider-level, and belong on
        :class:`~ai_agent_framework.models.base.ModelRequest` instead; see
        :meth:`default_generation_kwargs`.
        """
        kwargs: dict[str, Any] = {"timeout": self.ai_timeout_seconds}
        if self.ai_api_key:
            kwargs["api_key"] = self.ai_api_key
        if self.ai_base_url:
            kwargs["base_url"] = self.ai_base_url
        return kwargs

    def default_generation_kwargs(self) -> dict[str, Any]:
        """Default per-request generation settings (temperature, max_tokens)."""
        return {"temperature": self.ai_temperature, "max_tokens": self.ai_max_tokens}
