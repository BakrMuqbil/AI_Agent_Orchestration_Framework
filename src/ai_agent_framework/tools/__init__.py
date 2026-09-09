"""Tool interface, registry, decorator, and permission-aware execution."""

from ai_agent_framework.tools.base import Tool, ToolResult
from ai_agent_framework.tools.decorator import tool
from ai_agent_framework.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolResult", "ToolRegistry", "tool"]
