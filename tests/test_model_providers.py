"""
Tests for model providers.

All HTTP calls are mocked via ``httpx.MockTransport`` so these tests
never touch the network, regardless of which provider (Ollama, OpenAI,
Gemini) is being exercised.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from ai_agent_framework.core.errors import ConfigurationError, ModelProviderError
from ai_agent_framework.models.base import ModelMessage, ModelRequest
from ai_agent_framework.models.factory import create_provider
from ai_agent_framework.models.providers.gemini import GeminiProvider
from ai_agent_framework.models.providers.ollama import OllamaProvider
from ai_agent_framework.models.providers.openai import OpenAIProvider


def _mock_client_factory(handler):
    def _factory(*args, **kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), *args, **kwargs)

    return _factory


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-test",
                    "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
                },
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        response = await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))
        assert response.content == "hello there"
        assert response.has_tool_calls is False

    @pytest.mark.asyncio
    async def test_generate_parses_tool_calls(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-test",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "get_time",
                                            "arguments": json.dumps({"tz": "UTC"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        response = await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))
        assert response.has_tool_calls is True
        assert response.tool_calls[0].name == "get_time"
        assert response.tool_calls[0].arguments == {"tz": "UTC"}

    @pytest.mark.asyncio
    async def test_http_error_status_raises_model_provider_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal error")

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        with pytest.raises(ModelProviderError):
            await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))

    @pytest.mark.asyncio
    async def test_api_error_field_raises(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": {"message": "bad request"}})

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        with pytest.raises(ModelProviderError):
            await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"model": "llama3.1", "message": {"role": "assistant", "content": "hi from ollama"}},
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OllamaProvider(model="llama3.1")
        response = await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))
        assert response.content == "hi from ollama"
        assert response.provider == "ollama"

    @pytest.mark.asyncio
    async def test_connection_error_raises_model_provider_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OllamaProvider(model="llama3.1")
        with pytest.raises(ModelProviderError):
            await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))


class TestGeminiProvider:
    def test_requires_api_key(self):
        with pytest.raises(ModelProviderError):
            GeminiProvider(model="gemini-test", api_key=None)

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "hi from gemini"}]}}]},
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = GeminiProvider(model="gemini-test", api_key="fake-key")
        response = await provider.generate(ModelRequest(messages=[ModelMessage(role="user", content="hi")]))
        assert response.content == "hi from gemini"


class TestStructuredOutput:
    class Answer(BaseModel):
        answer: str
        confidence: float

    @pytest.mark.asyncio
    async def test_generate_structured_parses_json(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-test",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"answer": "42", "confidence": 0.9}),
                            }
                        }
                    ],
                },
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        result = await provider.generate_structured(
            ModelRequest(messages=[ModelMessage(role="user", content="what is the answer?")]),
            self.Answer,
        )
        assert result.answer == "42"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_generate_structured_strips_markdown_fences(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            fenced = "```json\n" + json.dumps({"answer": "43", "confidence": 0.5}) + "\n```"
            return httpx.Response(
                200,
                json={"model": "gpt-test", "choices": [{"message": {"role": "assistant", "content": fenced}}]},
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        result = await provider.generate_structured(
            ModelRequest(messages=[ModelMessage(role="user", content="q")]), self.Answer
        )
        assert result.answer == "43"

    @pytest.mark.asyncio
    async def test_generate_structured_invalid_json_raises(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"model": "gpt-test", "choices": [{"message": {"role": "assistant", "content": "not json"}}]},
            )

        monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
        provider = OpenAIProvider(model="gpt-test", api_key="fake-key")
        with pytest.raises(ModelProviderError):
            await provider.generate_structured(
                ModelRequest(messages=[ModelMessage(role="user", content="q")]), self.Answer
            )


class TestProviderFactory:
    def test_create_ollama_provider(self):
        provider = create_provider("ollama", model="llama3.1")
        assert isinstance(provider, OllamaProvider)

    def test_create_openai_provider(self):
        provider = create_provider("openai", model="gpt-test", api_key="k")
        assert isinstance(provider, OpenAIProvider)

    def test_create_gemini_provider(self):
        provider = create_provider("gemini", model="gemini-test", api_key="k")
        assert isinstance(provider, GeminiProvider)

    def test_unknown_provider_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            create_provider("not_a_real_provider", model="x")

    def test_provider_name_is_case_insensitive(self):
        provider = create_provider("OLLAMA", model="llama3.1")
        assert isinstance(provider, OllamaProvider)

    def test_custom_provider_registration(self):
        from ai_agent_framework.models.factory import register_provider
        from ai_agent_framework.models.base import ModelProvider

        class CustomProvider(ModelProvider):
            provider_name = "custom"

            async def generate(self, request):  # pragma: no cover - not exercised
                raise NotImplementedError

        register_provider("custom", CustomProvider)
        provider = create_provider("custom", model="x")
        assert isinstance(provider, CustomProvider)
