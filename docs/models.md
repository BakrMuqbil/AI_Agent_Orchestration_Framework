# Model Providers

## Why an abstraction

Core framework code (agents, the supervisor's `ModelPlanner`) never talks
to a specific LLM API directly. Everything goes through the
`ModelProvider` interface, so the same agent/supervisor logic works
unmodified whether the configured provider is Ollama, OpenAI, Gemini, or
a custom one a host application registers.

```
Model Provider (interface)
│
├── OllamaProvider              -- local/self-hosted, no API key required
├── OpenAICompatibleProvider    -- base for anything speaking the OpenAI
│                                  chat-completions wire format
│     └── OpenAIProvider        -- OpenAI itself, or OpenRouter/other
│                                  compatible gateways via base_url
├── GeminiProvider               -- Google Gemini's native REST API
└── <your custom provider>
```

## Choosing a provider via configuration

```bash
AI_PROVIDER=ollama
AI_MODEL=qwen2.5
```

or

```bash
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
AI_API_KEY=sk-...
```

or

```bash
AI_PROVIDER=gemini
AI_MODEL=gemini-1.5-flash
AI_API_KEY=...
```

```python
from ai_agent_framework.config import FrameworkSettings
from ai_agent_framework.models.factory import create_provider

settings = FrameworkSettings()
provider = create_provider(settings.ai_provider, model=settings.ai_model, **settings.provider_kwargs())
```

## The `ModelProvider` interface

```python
class ModelProvider:
    provider_name: str

    async def generate(self, request: ModelRequest) -> ModelResponse: ...
    async def generate_structured(self, request: ModelRequest, schema: type[SchemaT]) -> SchemaT: ...
```

`ModelRequest` and `ModelResponse` are provider-agnostic dataclasses:

```python
@dataclass
class ModelRequest:
    messages: list[ModelMessage]
    tools: list[dict] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str = "auto"

@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    model: str | None = None
    provider: str | None = None
```

Every provider translates this shape to and from its own wire format
internally (OpenAI-style `chat/completions` JSON, Ollama's `/api/chat`,
Gemini's `generateContent`). Callers never see the differences.

## Structured output

```python
from pydantic import BaseModel

class Recommendation(BaseModel):
    action: str
    confidence: float

result = await provider.generate_structured(request, Recommendation)
# result is a validated Recommendation instance, not a raw string
```

The base implementation works with any provider: it appends a system
instruction asking for JSON matching the schema, calls `generate()`,
strips Markdown code fences if present, parses JSON, and validates against
the schema -- raising `ModelProviderError` if the model's output isn't
valid JSON or doesn't satisfy the schema. This is what `ModelPlanner` uses
to get a `PlanDecision` back from any configured provider without brittle
text parsing.

## Adding a new provider

1. Subclass `ModelProvider`, set `provider_name`, implement `generate()`.
   Raise `ModelProviderError` (not a provider-specific/HTTP exception) on
   failure.
2. Register it: `register_provider("my_provider", MyProvider)`, or add it
   to `_PROVIDERS` in `ai_agent_framework/models/factory.py` if it's a
   first-class addition to the framework itself.
3. Nothing else in the framework changes -- `Supervisor`, `AgentEngine`,
   and every `Agent` work with it unmodified.

## Ollama

Ollama is treated as a first-class provider, but never a hard
requirement -- the framework depends only on the `ModelProvider`
interface. `OllamaProvider(model="llama3.1", base_url="http://localhost:11434")`
talks to a local or remote Ollama server's `/api/chat` endpoint and needs
no API key.
