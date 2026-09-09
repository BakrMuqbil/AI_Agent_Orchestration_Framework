"""
Basic end-to-end example: register two generic agents and a tool, run a
task through the supervisor, and print the result.

This example intentionally uses a deterministic, model-free
``RuleBasedPlanner`` (via ``AgentEngine(model=None)``) so it runs without
any API key or local Ollama server -- useful for a first smoke test.
Swap in a real model provider (see the commented-out block below) once
you have one configured.

Run with::

    python examples/basic/run_example.py
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
    tool,
)


# ---------------------------------------------------------------------------
# A tool: pure business logic, no framework coupling beyond the decorator.
# ---------------------------------------------------------------------------


class DiscountArgs(BaseModel):
    price: float = Field(gt=0)
    discount_percent: float = Field(ge=0, le=100)


@tool(
    name="calculate_discount",
    description="Calculate the final price after applying a percentage discount",
    args_schema=DiscountArgs,
    permission=Permission.READ,
)
async def calculate_discount(price: float, discount_percent: float) -> dict:
    final_price = price - (price * (discount_percent / 100))
    return {"final_price": round(final_price, 2)}


# ---------------------------------------------------------------------------
# Two generic agents.
# ---------------------------------------------------------------------------


class PricingAgent(Agent):
    name = "pricing_agent"
    description = "Calculates prices, discounts, and totals"
    capabilities = [AgentCapability(name="pricing", description="Discount and price calculations")]
    tools = [calculate_discount]
    permission = Permission.READ

    async def run(self, task: str, context) -> AgentResult:
        # A real agent would use context.tool_registry to invoke tools
        # (typically chosen by a model). Here we call it directly to keep
        # the example runnable without any model configured.
        result = await context.tool_registry.invoke(
            "calculate_discount",
            {"price": 1500.0, "discount_percent": 5.0},
            permissions=context.permissions,
            context=context,
        )
        return AgentResult.ok(result.output)


class GreeterAgent(Agent):
    name = "greeter_agent"
    description = "Greets the user and explains what happened"
    capabilities = [AgentCapability(name="greeting", description="Friendly responses")]

    async def run(self, task: str, context) -> AgentResult:
        return AgentResult.ok("Hello! I can help with pricing questions.")


async def main() -> None:
    engine = AgentEngine(model=None)  # RuleBasedPlanner: no LLM needed for this demo
    engine.register_agent(PricingAgent())
    engine.register_agent(GreeterAgent())
    engine.register_tool(calculate_discount)

    result = await engine.run("calculate discount pricing for the enterprise package")
    print("status:", result.status)
    print("output:", result.output)
    print("steps:", [(s.agent_name, s.success, s.output) for s in result.steps])

    # --- To use a real model-driven supervisor instead, replace the
    #     AgentEngine construction above with something like:
    #
    # from ai_agent_framework import create_provider
    # provider = create_provider("ollama", model="llama3.1")
    # engine = AgentEngine(model=provider)


if __name__ == "__main__":
    asyncio.run(main())
