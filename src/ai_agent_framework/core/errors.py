"""
Error hierarchy for the AI Agent Orchestration Framework.

All framework-raised exceptions inherit from :class:`FrameworkError` so
host applications can catch a single base type when they don't need
fine-grained handling, while still being able to catch specific
subtypes (permission errors, loop-limit errors, tool errors, etc.)
when they do.
"""

from __future__ import annotations

from typing import Any


class FrameworkError(Exception):
    """Base class for every exception raised by the framework."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class ConfigurationError(FrameworkError):
    """Raised when the framework or a provider is misconfigured."""


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


class AgentError(FrameworkError):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not registered."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            f"No agent registered under the name '{agent_name}'",
            details={"agent_name": agent_name},
        )
        self.agent_name = agent_name


class AgentExecutionError(AgentError):
    """Raised when an agent raises while executing a task."""

    def __init__(self, agent_name: str, cause: BaseException) -> None:
        super().__init__(
            f"Agent '{agent_name}' failed during execution: {cause}",
            details={"agent_name": agent_name, "cause": str(cause)},
        )
        self.agent_name = agent_name
        self.cause = cause


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class ToolError(FrameworkError):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"No tool registered under the name '{tool_name}'",
            details={"tool_name": tool_name},
        )
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation."""

    def __init__(self, tool_name: str, cause: str) -> None:
        super().__init__(
            f"Arguments for tool '{tool_name}' failed validation: {cause}",
            details={"tool_name": tool_name, "cause": cause},
        )
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Raised when a tool raises during execution."""

    def __init__(self, tool_name: str, cause: BaseException) -> None:
        super().__init__(
            f"Tool '{tool_name}' raised during execution: {cause}",
            details={"tool_name": tool_name, "cause": str(cause)},
        )
        self.tool_name = tool_name
        self.cause = cause


# --------------------------------------------------------------------------
# Security / permissions
# --------------------------------------------------------------------------


class PermissionDeniedError(FrameworkError):
    """Raised when an agent lacks the permission required for a tool."""

    def __init__(self, tool_name: str, required: str, granted: Any) -> None:
        super().__init__(
            f"Permission '{required}' is required to call tool '{tool_name}' "
            f"(granted permissions: {granted})",
            details={"tool_name": tool_name, "required": required, "granted": granted},
        )
        self.tool_name = tool_name
        self.required = required


class ConfirmationRequiredError(FrameworkError):
    """Raised when a tool call requires host-application confirmation.

    Host applications catch this to pause execution, surface a
    confirmation prompt to a human, and resume (or cancel) the task.
    """

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        super().__init__(
            f"Tool '{tool_name}' requires confirmation before it can run",
            details={"tool_name": tool_name, "arguments": arguments},
        )
        self.tool_name = tool_name
        self.arguments = arguments


# --------------------------------------------------------------------------
# Orchestration / execution limits
# --------------------------------------------------------------------------


class LoopLimitExceededError(FrameworkError):
    """Raised when the supervisor exceeds a configured safety limit."""

    def __init__(self, limit_name: str, limit_value: int) -> None:
        super().__init__(
            f"Execution stopped: exceeded '{limit_name}' limit of {limit_value}",
            details={"limit_name": limit_name, "limit_value": limit_value},
        )
        self.limit_name = limit_name
        self.limit_value = limit_value


# --------------------------------------------------------------------------
# Model providers
# --------------------------------------------------------------------------


class ModelProviderError(FrameworkError):
    """Raised when a model provider fails to produce a response."""

    def __init__(self, provider_name: str, cause: str) -> None:
        super().__init__(
            f"Model provider '{provider_name}' failed: {cause}",
            details={"provider_name": provider_name, "cause": cause},
        )
        self.provider_name = provider_name
