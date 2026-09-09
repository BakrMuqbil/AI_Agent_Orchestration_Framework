"""
Default in-process :class:`~ai_agent_framework.memory.base.MemoryStore`
implementation.

Suitable for development, tests, and single-process deployments. Host
applications that need persistence across restarts or multiple processes
should implement ``MemoryStore`` against Redis, a database, etc. -- the
interface is small by design specifically to make that easy.
"""

from __future__ import annotations

import copy
from typing import Any


class InMemoryStore:
    """A simple process-local, in-memory :class:`MemoryStore`."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lists: dict[str, dict[str, list[Any]]] = {}

    async def get(self, namespace: str, key: str) -> Any:
        return self._data.get(namespace, {}).get(key)

    async def set(self, namespace: str, key: str, value: Any) -> None:
        self._data.setdefault(namespace, {})[key] = copy.deepcopy(value)

    async def delete(self, namespace: str, key: str) -> None:
        self._data.get(namespace, {}).pop(key, None)
        self._lists.get(namespace, {}).pop(key, None)

    async def append(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        max_length: int | None = None,
    ) -> None:
        bucket = self._lists.setdefault(namespace, {}).setdefault(key, [])
        bucket.append(copy.deepcopy(value))
        if max_length is not None and len(bucket) > max_length:
            del bucket[: len(bucket) - max_length]

    async def get_list(self, namespace: str, key: str) -> list[Any]:
        return list(self._lists.get(namespace, {}).get(key, []))
