"""RBAC — Role-Based Access Control for Phase 7 enterprise features.

Model:
  User  →  has one or more Roles
  Role  →  has a set of Permissions
  Permission  →  "{action}:{resource}"  e.g. "read:documents", "execute:agents"

AccessControl.can(user, action, resource) is the single check point used
throughout the platform.  Wildcards are supported: permission "read:*" grants
read on every resource; "*:documents" grants all actions on documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class Resource(str, Enum):
    DOCUMENTS = "documents"
    SEARCH = "search"
    AGENTS = "agents"
    COPILOT = "copilot"
    EVALUATION = "evaluation"
    AUDIT = "audit"
    USERS = "users"
    ALL = "*"


@dataclass(frozen=True)
class Permission:
    """A single action+resource pair.

    Examples:
        Permission("read", "documents")
        Permission.of("execute:agents")
    """

    action: str
    resource: str

    def __str__(self) -> str:
        return f"{self.action}:{self.resource}"

    @classmethod
    def of(cls, spec: str) -> Permission:
        """Parse "action:resource" string."""
        parts = spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid permission spec '{spec}' — expected 'action:resource'")
        return cls(action=parts[0], resource=parts[1])

    def matches(self, action: str, resource: str) -> bool:
        """True if this permission covers the given action+resource pair.

        Wildcards: "*" in either field matches anything.
        """
        action_ok = self.action in (action, "*")
        resource_ok = self.resource in (resource, "*")
        return action_ok and resource_ok


@dataclass
class Role:
    """A named collection of permissions."""

    name: str
    permissions: set[Permission] = field(default_factory=set)

    def grant(self, *specs: str) -> Role:
        """Add permissions from "action:resource" strings.  Returns self for chaining."""
        for spec in specs:
            self.permissions.add(Permission.of(spec))
        return self

    def revoke(self, spec: str) -> Role:
        self.permissions.discard(Permission.of(spec))
        return self

    def can(self, action: str, resource: str) -> bool:
        return any(p.matches(action, resource) for p in self.permissions)


# ── Built-in platform roles ───────────────────────────────────────────────────

def _make_admin_role() -> Role:
    return Role("admin").grant("*:*")

def _make_analyst_role() -> Role:
    return (
        Role("analyst")
        .grant("read:documents", "read:search", "read:copilot")
        .grant("execute:agents", "execute:copilot", "execute:evaluation")
        .grant("write:evaluation")
    )

def _make_viewer_role() -> Role:
    return (
        Role("viewer")
        .grant("read:documents", "read:search", "read:copilot")
    )

def _make_auditor_role() -> Role:
    return (
        Role("auditor")
        .grant("read:documents", "read:audit")
        .grant("read:search", "read:evaluation")
    )


BUILTIN_ROLES: dict[str, Role] = {
    "admin": _make_admin_role(),
    "analyst": _make_analyst_role(),
    "viewer": _make_viewer_role(),
    "auditor": _make_auditor_role(),
}


@dataclass
class User:
    """A platform user with one or more assigned roles."""

    user_id: str
    roles: list[Role] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def assign_role(self, role: Role) -> None:
        if role not in self.roles:
            self.roles.append(role)

    def remove_role(self, role_name: str) -> None:
        self.roles = [r for r in self.roles if r.name != role_name]

    def role_names(self) -> list[str]:
        return [r.name for r in self.roles]


class AccessControl:
    """Central access control check point.

    Usage:
        ac = AccessControl()
        user = User("alice", roles=[BUILTIN_ROLES["analyst"]])
        ac.can(user, "execute", "agents")   # True
        ac.can(user, "delete", "documents") # False
        ac.require(user, "delete", "documents")  # raises PermissionDeniedError
    """

    def can(self, user: User, action: str, resource: str) -> bool:
        """Return True if *user* holds a role that grants action on resource."""
        return any(role.can(action, resource) for role in user.roles)

    def require(self, user: User, action: str, resource: str) -> None:
        """Raise PermissionDeniedError if the user lacks the required permission."""
        if not self.can(user, action, resource):
            raise PermissionDeniedError(
                f"User '{user.user_id}' does not have '{action}:{resource}' permission."
            )

    def permissions_for(self, user: User) -> set[Permission]:
        """Return the merged set of all permissions the user holds."""
        result: set[Permission] = set()
        for role in user.roles:
            result |= role.permissions
        return result


class PermissionDeniedError(Exception):
    """Raised when a user attempts an action they are not authorised for."""
