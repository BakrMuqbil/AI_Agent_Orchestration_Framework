"""
Context builder: turns retrieved chunks into a grounded prompt block.

Kept separate from the retriever so a host application can customize how
retrieved context is formatted and injected without touching retrieval
logic.
"""

from __future__ import annotations

from ai_agent_framework.rag.base import RetrievedChunk


def build_rag_context(chunks: list[RetrievedChunk], *, header: str = "Relevant context:") -> str:
    """Render retrieved chunks into a single context string for a prompt.

    Returns an empty string if ``chunks`` is empty, so callers can safely
    concatenate the result without conditionally checking for emptiness.
    """
    if not chunks:
        return ""
    body = "\n\n".join(c.document.content for c in chunks)
    return f"{header}\n{body}"
