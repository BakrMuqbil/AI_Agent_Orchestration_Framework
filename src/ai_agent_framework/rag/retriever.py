"""
A simple, dependency-free retriever.

Uses word-overlap scoring against documents in a
:class:`~ai_agent_framework.rag.base.DocumentStore`. This mirrors what
the original project actually implemented (a basic word-frequency
similarity, not a real vector index) but is exposed behind the
:class:`~ai_agent_framework.rag.base.Retriever` protocol, so a host
application can drop in a real embedding-backed retriever (Chroma,
pgvector, ...) without changing any agent or supervisor code -- both
implementations satisfy the same interface.

If an :class:`~ai_agent_framework.rag.base.EmbeddingProvider` is
supplied, cosine similarity over embeddings is used instead of keyword
overlap.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ai_agent_framework.rag.base import Document, DocumentStore, EmbeddingProvider, RetrievedChunk

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _keyword_overlap_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    shared = set(query_counts) & set(doc_counts)
    if not shared:
        return 0.0
    dot = sum(query_counts[w] * doc_counts[w] for w in shared)
    query_norm = math.sqrt(sum(v * v for v in query_counts.values()))
    doc_norm = math.sqrt(sum(v * v for v in doc_counts.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class SimpleRetriever:
    """Retrieves the top-k most relevant documents from a document store."""

    def __init__(self, store: DocumentStore, *, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    async def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        documents = await self.store.all()
        if not documents:
            return []

        if self.embedding_provider is not None:
            scored = await self._score_with_embeddings(query, documents)
        else:
            scored = self._score_with_keywords(query, documents)

        scored.sort(key=lambda c: c.score, reverse=True)
        return [c for c in scored if c.score > 0][:top_k]

    def _score_with_keywords(self, query: str, documents: list[Document]) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        results = []
        for doc in documents:
            score = _keyword_overlap_score(query_tokens, _tokenize(doc.content))
            results.append(RetrievedChunk(document=doc, score=score))
        return results

    async def _score_with_embeddings(self, query: str, documents: list[Document]) -> list[RetrievedChunk]:
        assert self.embedding_provider is not None
        query_vec = await self.embedding_provider.embed(query)
        results = []
        for doc in documents:
            doc_vec = await self.embedding_provider.embed(doc.content)
            score = _cosine_similarity(query_vec, doc_vec)
            results.append(RetrievedChunk(document=doc, score=score))
        return results
