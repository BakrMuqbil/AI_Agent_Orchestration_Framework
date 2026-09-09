"""Tests for RAG: document store, retriever, context builder."""

from __future__ import annotations

import pytest

from ai_agent_framework.rag.base import Document
from ai_agent_framework.rag.context_builder import build_rag_context
from ai_agent_framework.rag.document_store import InMemoryDocumentStore
from ai_agent_framework.rag.retriever import SimpleRetriever


class TestInMemoryDocumentStore:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        store = InMemoryDocumentStore()
        doc = Document(id="1", content="hello world")
        await store.add(doc)
        assert await store.get("1") == doc

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        store = InMemoryDocumentStore()
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_all_returns_every_document(self):
        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="a"))
        await store.add(Document(id="2", content="b"))
        docs = await store.all()
        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="a"))
        await store.delete("1")
        assert await store.get("1") is None


class TestSimpleRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_ranks_by_keyword_overlap(self):
        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="solar panel installation guide"))
        await store.add(Document(id="2", content="unrelated cooking recipe"))
        retriever = SimpleRetriever(store)

        results = await retriever.retrieve("solar panel installation", top_k=2)
        assert results[0].document.id == "1"

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_for_empty_store(self):
        retriever = SimpleRetriever(InMemoryDocumentStore())
        results = await retriever.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_excludes_zero_score_documents(self):
        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="completely unrelated content xyz"))
        retriever = SimpleRetriever(store)
        results = await retriever.retrieve("solar panel battery inverter", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        store = InMemoryDocumentStore()
        for i in range(5):
            await store.add(Document(id=str(i), content="solar panel battery inverter sizing"))
        retriever = SimpleRetriever(store)
        results = await retriever.retrieve("solar panel", top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_embedding_provider(self):
        class FakeEmbeddingProvider:
            async def embed(self, text: str) -> list[float]:
                # Deterministic fake embedding: presence of "solar" -> [1, 0], else [0, 1]
                return [1.0, 0.0] if "solar" in text.lower() else [0.0, 1.0]

        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="solar energy systems"))
        await store.add(Document(id="2", content="totally different topic"))
        retriever = SimpleRetriever(store, embedding_provider=FakeEmbeddingProvider())

        results = await retriever.retrieve("solar power", top_k=2)
        assert results[0].document.id == "1"


class TestContextBuilder:
    def test_empty_chunks_returns_empty_string(self):
        assert build_rag_context([]) == ""

    @pytest.mark.asyncio
    async def test_builds_context_from_chunks(self):
        store = InMemoryDocumentStore()
        await store.add(Document(id="1", content="solar panel guide"))
        retriever = SimpleRetriever(store)
        chunks = await retriever.retrieve("solar panel")
        context = build_rag_context(chunks)
        assert "solar panel guide" in context
        assert context.startswith("Relevant context:")
