"""
FastAPI application factory.

Builds a FastAPI app wired to a given :class:`AgentEngine`. Kept as a
factory function (rather than a single module-level ``app`` object) so
the API is reusable: a host application constructs its own engine
(with its own agents/tools/provider) and gets back a ready-to-serve API,
instead of the framework owning a single global app instance.
"""

from __future__ import annotations

from fastapi import FastAPI

from ai_agent_framework.api.routes import get_engine, router
from ai_agent_framework.orchestration.engine import AgentEngine


def create_app(engine: AgentEngine, *, title: str = "AI Agent Orchestration Framework") -> FastAPI:
    """Create a FastAPI app exposing ``engine`` over HTTP.

    Example::

        engine = AgentEngine(model=create_provider("ollama", model="llama3.1"))
        engine.register_agent(MyAgent())
        app = create_app(engine)
        # uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    app = FastAPI(title=title)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_engine] = lambda: engine
    return app
