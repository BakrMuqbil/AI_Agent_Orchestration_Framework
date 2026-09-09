"""In-memory :class:`~ai_agent_framework.rag.base.DocumentStore` implementation."""

from __future__ import annotations

from ai_agent_framework.rag.base import Document


class InMemoryDocumentStore:
    """A simple process-local document store.

    Suitable for development and small knowledge bases. Host applications
    needing persistence or a real vector index implement the
    ``DocumentStore``/``Retriever`` protocols against their vector
    database of choice.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    async def add(self, document: Document) -> None:
        self._documents[document.id] = document

    async def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    async def all(self) -> list[Document]:
        return list(self._documents.values())

    async def delete(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)
