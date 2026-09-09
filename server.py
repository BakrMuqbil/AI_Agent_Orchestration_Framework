"""
Runnable server entrypoint.

Wires environment configuration to a model provider, builds an
:class:`AgentEngine`, and exposes it as a FastAPI app. This module
contains zero domain-specific agents/tools by default -- it's the
generic "run the framework by itself" entrypoint used for local
development and health-checking. Host applications building a real
product register their own agents (see ``examples/dalalti/``) and call
:func:`ai_agent_framework.api.app.create_app` themselves rather than
importing this module.

Run with::

    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

from ai_agent_framework.api.app import create_app
from ai_agent_framework.config import FrameworkSettings
from ai_agent_framework.core.events import stdout_logging_handler
from ai_agent_framework.models.factory import create_provider
from ai_agent_framework.orchestration.engine import AgentEngine
from ai_agent_framework.orchestration.supervisor import SupervisorConfig

settings = FrameworkSettings()

provider = create_provider(settings.ai_provider, model=settings.ai_model, **settings.provider_kwargs())

engine = AgentEngine(
    model=provider,
    config=SupervisorConfig(
        max_iterations=settings.max_iterations,
        max_agent_calls=settings.max_agent_calls,
        max_tool_calls=settings.max_tool_calls,
        timeout_seconds=settings.task_timeout_seconds,
    ),
)

if settings.enable_stdout_event_logging:
    engine.event_bus.subscribe(None, stdout_logging_handler)

app = create_app(engine)
