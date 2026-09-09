"""
RAG interfaces.

Deliberately small and provider-independent. The framework does not ship
a vector database; it defines the seams (``DocumentStore``,
``EmbeddingProvider``, ``Retriever``) that a host application plugs a
real one into (Chroma, pgvector, Pinecone, ...). A basic in-memory,
keyword-overlap implementation is provided so the framework is usable
out of the box without any external dependency, matching what the
original project actually had (word-frequency similarity, not real
embeddings) rather than pretending to be a production vector database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Document:
    """A single retrievable document / chunk."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A document returned by a retriever, with its relevance score."""

    document: Document
    score: float


class EmbeddingProvider(Protocol):
    """Interface for turning text into a vector representation.

    Host applications plug in a real embedding model (OpenAI embeddings,
    a local sentence-transformer, ...) by implementing this. The bundled
    default (:class:`~ai_agent_framework.rag.retriever.SimpleRetriever`'s
    fallback) does not require one -- it uses keyword overlap -- so RAG
    works without any embedding provider configured.
    """

    async def embed(self, text: str) -> list[float]: ...


class DocumentStore(Protocol):
    """Interface for storing and listing documents available to retrieval."""

    async def add(self, document: Document) -> None: ...

    async def get(self, doc_id: str) -> Document | None: ...

    async def all(self) -> list[Document]: ...

    async def delete(self, doc_id: str) -> None: ...


class Retriever(Protocol):
    """Interface for retrieving the documents most relevant to a query."""

    async def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]: ...
