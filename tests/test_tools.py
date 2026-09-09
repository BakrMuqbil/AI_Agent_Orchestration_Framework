"""Tests for tool definition, validation, permission checks, and execution."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from ai_agent_framework.core.errors import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    ToolNotFoundError,
)
from ai_agent_framework.security.permissions import Permission, PermissionSet
from ai_agent_framework.tools.base import Tool
from ai_agent_framework.tools.decorator import tool
from ai_agent_framework.tools.registry import ToolRegistry


class AddArgs(BaseModel):
    a: int
    b: int = Field(ge=0)


@tool(name="add", description="Add two numbers", args_schema=AddArgs, permission=Permission.READ)
async def add(a: int, b: int) -> dict:
    return {"sum": a + b}


@tool(name="sync_add", description="Sync add", args_schema=AddArgs, permission=Permission.READ)
def sync_add(a: int, b: int) -> dict:
    return {"sum": a + b}


@tool(name="raises", description="Always raises", permission=Permission.READ)
async def raises() -> dict:
    raise ValueError("nope")


@tool(
    name="dangerous",
    description="Requires confirmation",
    permission=Permission.DELETE,
    requires_confirmation=True,
)
async def dangerous() -> dict:
    return {"deleted": True}


class TestToolDefinition:
    def test_decorator_produces_tool_instance(self):
        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.permission == Permission.READ

    def test_to_definition_shape(self):
        definition = add.to_definition()
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "add"
        assert "parameters" in definition["function"]

    def test_parameters_schema_falls_back_when_no_args_schema(self):
        t = Tool(lambda: None, name="x", description="x")
        schema = t.parameters_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True


class TestToolValidation:
    def test_valid_arguments_pass(self):
        validated = add.validate_arguments({"a": 1, "b": 2})
        assert validated == {"a": 1, "b": 2}

    def test_invalid_arguments_raise(self):
        from ai_agent_framework.core.errors import ToolValidationError

        with pytest.raises(ToolValidationError):
            add.validate_arguments({"a": 1, "b": -5})  # b must be >= 0

    def test_missing_required_argument_raises(self):
        from ai_agent_framework.core.errors import ToolValidationError

        with pytest.raises(ToolValidationError):
            add.validate_arguments({"a": 1})


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_async_tool_executes(self):
        result = await add.execute({"a": 2, "b": 3})
        assert result.success is True
        assert result.output == {"sum": 5}

    @pytest.mark.asyncio
    async def test_sync_tool_executes(self):
        result = await sync_add.execute({"a": 4, "b": 6})
        assert result.success is True
        assert result.output == {"sum": 10}

    @pytest.mark.asyncio
    async def test_validation_failure_returns_failed_result_not_exception(self):
        result = await add.execute({"a": 1, "b": -1})
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_tool_exception_is_captured_in_result(self):
        result = await raises.execute({})
        assert result.success is False
        assert "nope" in result.error


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(add)
        assert registry.get("add") is add

    def test_get_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("missing")

    def test_definitions_lists_all_tools(self):
        registry = ToolRegistry([add, sync_add])
        defs = registry.definitions()
        names = {d["function"]["name"] for d in defs}
        assert names == {"add", "sync_add"}

    def test_scoped_returns_subset(self):
        registry = ToolRegistry([add, sync_add, dangerous])
        scoped = registry.scoped(["add"])
        assert len(scoped) == 1
        assert scoped.has("add")
        assert not scoped.has("sync_add")

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        registry = ToolRegistry([add])
        result = await registry.invoke("add", {"a": 1, "b": 1}, permissions=PermissionSet.admin())
        assert result.success is True
        assert result.output == {"sum": 2}

    @pytest.mark.asyncio
    async def test_invoke_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            await registry.invoke("nope", {}, permissions=PermissionSet.admin())

    @pytest.mark.asyncio
    async def test_invoke_denies_insufficient_permission(self):
        registry = ToolRegistry([dangerous])
        with pytest.raises(PermissionDeniedError):
            await registry.invoke(
                "dangerous", {}, permissions=PermissionSet.read_only(), allow_confirmation_bypass=True
            )

    @pytest.mark.asyncio
    async def test_invoke_allows_sufficient_permission(self):
        registry = ToolRegistry([add])
        result = await registry.invoke(
            "add", {"a": 1, "b": 1}, permissions=PermissionSet(level=Permission.READ)
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_invoke_requires_confirmation(self):
        registry = ToolRegistry([dangerous])
        with pytest.raises(ConfirmationRequiredError):
            await registry.invoke("dangerous", {}, permissions=PermissionSet.admin())

    @pytest.mark.asyncio
    async def test_invoke_confirmation_bypass_allows_execution(self):
        registry = ToolRegistry([dangerous])
        result = await registry.invoke(
            "dangerous", {}, permissions=PermissionSet.admin(), allow_confirmation_bypass=True
        )
        assert result.success is True
        assert result.output == {"deleted": True}


class TestPermissionSet:
    def test_hierarchical_permission_allows_lower_levels(self):
        perms = PermissionSet(level=Permission.ADMIN)
        assert perms.allows(Permission.READ)
        assert perms.allows(Permission.DELETE)

    def test_read_only_denies_create(self):
        perms = PermissionSet.read_only()
        assert perms.allows(Permission.READ)
        assert not perms.allows(Permission.CREATE)

    def test_named_permission(self):
        perms = PermissionSet(level=Permission.READ)
        perms.grant_named("send_email")
        assert perms.allows("send_email")
        assert not perms.allows("delete_account")
