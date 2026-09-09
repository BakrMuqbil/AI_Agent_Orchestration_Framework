"""
Dalalti integration example.

Demonstrates how an external project (Dalalti, an e-commerce platform)
registers its own domain-specific agents and tools with the framework
using :class:`~ai_agent_framework.orchestration.project.ProjectRegistration`,
without any Dalalti-specific code living inside the framework's core.

This is illustrative only -- it does not connect to a real Dalalti
database or API. ``ProductAgent``, ``OrderAgent``, and ``CustomerAgent``
here are stand-ins showing the *shape* of integration: define tools with
``@tool``, define agents that declare and use those tools, bundle them
into a ``ProjectRegistration``, and apply it to an ``AgentEngine``.

Run with::

    python examples/dalalti/integration_example.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ai_agent_framework import (
    Agent,
    AgentCapability,
    AgentEngine,
    AgentResult,
    Permission,
    ProjectRegistration,
    tool,
)

# ---------------------------------------------------------------------------
# Dalalti-specific tools. These would call real Dalalti services
# (Prisma/Supabase queries, internal APIs, ...) in production; here they
# return canned data so the example is runnable standalone.
# ---------------------------------------------------------------------------


class LookupProductArgs(BaseModel):
    product_id: str = Field(min_length=1)


@tool(
    name="dalalti_lookup_product",
    description="Look up a Dalalti product by id",
    args_schema=LookupProductArgs,
    permission=Permission.READ,
)
async def dalalti_lookup_product(product_id: str) -> dict:
    return {"product_id": product_id, "name": "Sample Product", "price": 250.0, "stock": 12}


class CreateOrderArgs(BaseModel):
    product_id: str
    quantity: int = Field(gt=0)
    customer_id: str


@tool(
    name="dalalti_create_order",
    description="Create a new order in Dalalti",
    args_schema=CreateOrderArgs,
    permission=Permission.CREATE,
    requires_confirmation=True,  # a real order-creating call should be confirmed by the host app
)
async def dalalti_create_order(product_id: str, quantity: int, customer_id: str) -> dict:
    return {"order_id": "ord_12345", "product_id": product_id, "quantity": quantity, "customer_id": customer_id}


class LookupCustomerArgs(BaseModel):
    customer_id: str = Field(min_length=1)


@tool(
    name="dalalti_lookup_customer",
    description="Look up a Dalalti customer by id",
    args_schema=LookupCustomerArgs,
    permission=Permission.READ,
)
async def dalalti_lookup_customer(customer_id: str) -> dict:
    return {"customer_id": customer_id, "name": "Sample Customer", "tier": "gold"}


# ---------------------------------------------------------------------------
# Dalalti-specific agents.
# ---------------------------------------------------------------------------


class ProductAgent(Agent):
    name = "dalalti.product_agent"
    description = "Looks up and reasons about Dalalti products"
    capabilities = [AgentCapability(name="product_lookup", description="Find product details")]
    tools = [dalalti_lookup_product]
    permission = Permission.READ

    async def run(self, task: str, context) -> AgentResult:
        result = await context.tool_registry.invoke(
            "dalalti_lookup_product",
            {"product_id": "prod_001"},
            permissions=context.permissions,
            context=context,
        )
        return AgentResult.ok(result.output)


class OrderAgent(Agent):
    name = "dalalti.order_agent"
    description = "Creates and manages Dalalti orders"
    capabilities = [AgentCapability(name="order_management", description="Create orders")]
    tools = [dalalti_create_order]
    permission = Permission.CREATE

    async def run(self, task: str, context) -> AgentResult:
        return AgentResult.ok("Order agent ready (call dalalti_create_order via the tool registry).")


class CustomerAgent(Agent):
    name = "dalalti.customer_agent"
    description = "Looks up Dalalti customer information"
    capabilities = [AgentCapability(name="customer_lookup", description="Find customer details")]
    tools = [dalalti_lookup_customer]
    permission = Permission.READ

    async def run(self, task: str, context) -> AgentResult:
        result = await context.tool_registry.invoke(
            "dalalti_lookup_customer",
            {"customer_id": "cust_001"},
            permissions=context.permissions,
            context=context,
        )
        return AgentResult.ok(result.output)


dalalti_project = ProjectRegistration(
    name="dalalti",
    agents=[ProductAgent(), OrderAgent(), CustomerAgent()],
    tools=[dalalti_lookup_product, dalalti_create_order, dalalti_lookup_customer],
    metadata={"domain": "e-commerce", "stack": "Next.js/Prisma/Supabase"},
)


async def main() -> None:
    engine = AgentEngine(model=None)  # swap in a real provider for LLM-driven routing
    dalalti_project.apply(engine)

    result = await engine.run("look up product prod_001")
    print("status:", result.status)
    print("output:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
