"""
Legacy migration example.

Shows how the *original* "Techno-Bakr AI Agent" project's tools
(send_email, create_operational_task/Trello, prioritize_tasks,
generate_session_summary, query_knowledge, get_current_time,
calculate_discount) map onto the new framework's ``@tool`` /
``Permission`` / ``Agent`` primitives, preserving functionality while
removing the coupling that made the original code impossible to reuse
in another project.

This is an illustration, not a drop-in replacement -- a real migration
would also wire real SMTP/Trello credentials via host-application config
(kept out of the framework itself, since it's Techno-Bakr-specific) and
register these tools/agent with a real model provider instead of the
deterministic demo planner used here.

Run with::

    python examples/legacy_techno_bakr/migrated_agent.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

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
# Tools, ported from app/services/tools.py. Business logic is unchanged;
# only the packaging (decorator, schema, permission) is new.
# ---------------------------------------------------------------------------


class DiscountArgs(BaseModel):
    price: float = Field(gt=0)
    discount_percent: float = Field(ge=0, le=100)


@tool(
    name="calculate_discount",
    description="Calculate the final price after a percentage discount",
    args_schema=DiscountArgs,
    permission=Permission.READ,
)
async def calculate_discount(price: float, discount_percent: float) -> dict:
    return {"final_price": price - (price * (discount_percent / 100))}


@tool(
    name="get_current_time",
    description="Get the current UTC time",
    permission=Permission.READ,
)
async def get_current_time() -> dict:
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "formatted": now.strftime("%Y-%m-%d %H:%M UTC")}


class SendEmailArgs(BaseModel):
    recipient: str
    subject: str
    body: str


@tool(
    name="send_email",
    description="Send an email (host application supplies real SMTP wiring)",
    args_schema=SendEmailArgs,
    permission=Permission.CREATE,
    requires_confirmation=True,
)
async def send_email(recipient: str, subject: str, body: str) -> dict:
    # The original project sent real SMTP mail here. That is host-application
    # infrastructure, not framework logic -- a real migration plugs in an
    # SMTP client the same way this stub does, but reading credentials
    # from the host app's own settings rather than the framework's.
    return {"status": "queued", "to": recipient, "subject": subject}


class CreateTicketArgs(BaseModel):
    title: str
    description: str
    priority: str = Field(default="Medium", pattern="^(High|Medium|Low)$")


@tool(
    name="create_operational_task",
    description="Create a ticket in an external task tracker (e.g. Trello)",
    args_schema=CreateTicketArgs,
    permission=Permission.CREATE,
    requires_confirmation=True,
)
async def create_operational_task(title: str, description: str, priority: str = "Medium") -> dict:
    # Original project called the Trello REST API here.
    return {"status": "created", "title": title, "priority": priority}


# ---------------------------------------------------------------------------
# A single agent bundling the legacy tools, replacing the monolithic
# run_agent() function from the original app/services/agent_service.py.
# ---------------------------------------------------------------------------


class OperationsAgent(Agent):
    name = "operations_agent"
    description = "Handles operational tasks: email, tickets, time, pricing"
    capabilities = [
        AgentCapability(name="email", description="Send emails"),
        AgentCapability(name="ticketing", description="Create operational tickets"),
        AgentCapability(name="pricing", description="Calculate discounts"),
    ]
    tools = [calculate_discount, get_current_time, send_email, create_operational_task]
    permission = Permission.CREATE

    async def run(self, task: str, context) -> AgentResult:
        result = await context.tool_registry.invoke(
            "get_current_time",
            {},
            permissions=context.permissions,
            context=context,
        )
        return AgentResult.ok(result.output)


async def main() -> None:
    engine = AgentEngine(model=None)
    engine.register_agent(OperationsAgent())
    for t in (calculate_discount, get_current_time, send_email, create_operational_task):
        engine.register_tool(t)

    result = await engine.run("what time is it")
    print("status:", result.status)
    print("output:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
