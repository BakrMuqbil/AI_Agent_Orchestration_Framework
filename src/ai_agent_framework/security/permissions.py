"""
Permission system.

Tools declare the permission they require to run. Agents (or the
execution context, when a host application wants to scope permissions
per user/session rather than per agent) are granted a :class:`PermissionSet`.
Before a tool executes, the orchestration layer checks that the granted
set covers the tool's required permission; if not, execution stops with
:class:`~ai_agent_framework.core.errors.PermissionDeniedError`.

Sensitive operations can additionally be marked ``requires_confirmation``
on the :class:`~ai_agent_framework.tools.base.Tool`, in which case the
host application is expected to catch
:class:`~ai_agent_framework.core.errors.ConfirmationRequiredError` and
decide whether to resume execution.
"""

from __future__ import annotations

from enum import IntEnum


class Permission(IntEnum):
    """Standard permission levels, ordered from least to most privileged.

    Levels are cumulative: granting ``UPDATE`` implies ``READ`` and
    ``CREATE`` are also allowed, and so on up to ``ADMIN``. Host
    applications that need non-hierarchical, named permissions (e.g.
    "can_refund" vs "can_ship") should model those as separate tool
    metadata / capabilities rather than forcing them into this enum —
    this hierarchy covers the common CRUD-shaped case.
    """

    READ = 10
    CREATE = 20
    UPDATE = 30
    DELETE = 40
    ADMIN = 100


class PermissionSet:
    """The set of permissions granted to an agent or execution context.

    Supports both the hierarchical :class:`Permission` levels and
    arbitrary named grants (for permissions that don't fit a hierarchy,
    e.g. ``"send_email"``).
    """

    def __init__(
        self,
        level: Permission = Permission.READ,
        *,
        named: set[str] | None = None,
    ) -> None:
        self.level = level
        self.named = named or set()

    def allows(self, required: Permission | str) -> bool:
        """Return whether this permission set satisfies ``required``."""
        if isinstance(required, Permission):
            return self.level >= required
        return required in self.named or self.level >= Permission.ADMIN

    def grant_named(self, name: str) -> "PermissionSet":
        self.named.add(name)
        return self

    @classmethod
    def admin(cls) -> "PermissionSet":
        """A permission set that allows everything."""
        return cls(level=Permission.ADMIN)

    @classmethod
    def read_only(cls) -> "PermissionSet":
        return cls(level=Permission.READ)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"PermissionSet(level={self.level.name}, named={sorted(self.named)})"
