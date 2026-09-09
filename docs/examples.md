# Examples

Three worked examples ship under the examples directory, each runnable
standalone.

## Basic

A minimal, dependency-free end-to-end walkthrough: two generic agents
(pricing and greeting), one tool, and a full engine run using the
deterministic rule-based planner (no LLM or API key needed). This is the
best starting point to confirm your install works and to see the
register-agents-then-run shape in its simplest form. It prints the task
status, final output, and every step the supervisor took.

## Dalalti integration

Shows how an external project registers its own domain-specific agents
and tools using a project registration, without any Dalalti-specific
code living inside the framework. Includes three illustrative agents
(product lookup, order creation, customer lookup) and matching tools, one
of which is marked as requiring confirmation to demonstrate the
permission and confirmation flow on a write operation.

This example does not connect to a real Dalalti database or API -- the
tools return canned data. It exists to show integration shape, not to be
a working Dalalti connector.

## Legacy migration

Ports the original Techno-Bakr AI Agent project's tools (send email,
create an operational ticket, calculate a discount, get the current time)
onto the framework's tool, permission, and agent primitives. Demonstrates
that existing functionality is preserved through the migration, not
discarded, and shows what changes (packaging, schema, permission,
confirmation) and what doesn't (the actual business logic inside each
tool).

## Using a real model provider in any example

Every example defaults to an engine constructed with no model, which uses
the deterministic rule-based planner and requires no external service. To
see model-driven, multi-step supervision instead, construct a real
provider and pass it in when building the engine:

```python
from ai_agent_framework import create_provider, AgentEngine

provider = create_provider("ollama", model="llama3.1")
engine = AgentEngine(model=provider)
```

This switches the engine to the model-backed planner automatically -- no
other code in the examples needs to change.
