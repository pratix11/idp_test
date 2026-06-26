"""Tests for Phase 7 RBAC (Task 48).

Tests cover:
- Permission.of() parsing and __str__
- Permission.matches() including wildcards
- Role.grant() / revoke() / can()
- Built-in roles have correct capabilities
- User.assign_role() / remove_role() / role_names()
- AccessControl.can() with single and multiple roles
- AccessControl.require() raises PermissionDeniedError
- AccessControl.permissions_for() union of all roles
- Admin wildcard grants everything
- Viewer is read-only
"""

from __future__ import annotations

import pytest

from property_intel.enterprise.rbac import (
    BUILTIN_ROLES,
    AccessControl,
    Permission,
    PermissionDeniedError,
    Role,
    User,
)


# ── Permission ────────────────────────────────────────────────────────────────


def test_permission_of_parses_spec() -> None:
    p = Permission.of("read:documents")
    assert p.action == "read"
    assert p.resource == "documents"


def test_permission_str() -> None:
    assert str(Permission("execute", "agents")) == "execute:agents"


def test_permission_of_invalid_raises() -> None:
    with pytest.raises(ValueError):
        Permission.of("nodcolon")


def test_permission_matches_exact() -> None:
    p = Permission.of("read:documents")
    assert p.matches("read", "documents")
    assert not p.matches("write", "documents")
    assert not p.matches("read", "agents")


def test_permission_matches_action_wildcard() -> None:
    p = Permission.of("*:documents")
    assert p.matches("read", "documents")
    assert p.matches("delete", "documents")
    assert not p.matches("read", "agents")


def test_permission_matches_resource_wildcard() -> None:
    p = Permission.of("read:*")
    assert p.matches("read", "documents")
    assert p.matches("read", "agents")
    assert not p.matches("write", "documents")


def test_permission_matches_full_wildcard() -> None:
    p = Permission.of("*:*")
    assert p.matches("delete", "users")
    assert p.matches("execute", "evaluation")


# ── Role ──────────────────────────────────────────────────────────────────────


def test_role_grant_and_can() -> None:
    r = Role("analyst").grant("read:documents", "execute:agents")
    assert r.can("read", "documents")
    assert r.can("execute", "agents")
    assert not r.can("delete", "documents")


def test_role_revoke() -> None:
    r = Role("test").grant("read:documents", "write:documents")
    r.revoke("write:documents")
    assert r.can("read", "documents")
    assert not r.can("write", "documents")


def test_role_grant_returns_self_for_chaining() -> None:
    r = Role("test")
    result = r.grant("read:search")
    assert result is r


# ── Built-in roles ────────────────────────────────────────────────────────────


def test_admin_can_do_everything() -> None:
    admin = BUILTIN_ROLES["admin"]
    for action in ["read", "write", "delete", "execute", "admin"]:
        for resource in ["documents", "search", "agents", "audit", "users"]:
            assert admin.can(action, resource), f"admin should have {action}:{resource}"


def test_viewer_can_only_read() -> None:
    viewer = BUILTIN_ROLES["viewer"]
    assert viewer.can("read", "documents")
    assert viewer.can("read", "search")
    assert not viewer.can("write", "documents")
    assert not viewer.can("delete", "documents")
    assert not viewer.can("execute", "agents")


def test_analyst_can_execute_agents() -> None:
    analyst = BUILTIN_ROLES["analyst"]
    assert analyst.can("execute", "agents")
    assert analyst.can("execute", "copilot")
    assert analyst.can("read", "documents")
    assert not analyst.can("delete", "documents")


def test_auditor_can_read_audit_not_write() -> None:
    auditor = BUILTIN_ROLES["auditor"]
    assert auditor.can("read", "audit")
    assert auditor.can("read", "documents")
    assert not auditor.can("write", "documents")
    assert not auditor.can("execute", "agents")


# ── User ──────────────────────────────────────────────────────────────────────


def test_user_assign_role() -> None:
    user = User("alice")
    user.assign_role(BUILTIN_ROLES["analyst"])
    assert "analyst" in user.role_names()


def test_user_remove_role() -> None:
    user = User("bob", roles=[BUILTIN_ROLES["viewer"], BUILTIN_ROLES["analyst"]])
    user.remove_role("viewer")
    assert "viewer" not in user.role_names()
    assert "analyst" in user.role_names()


def test_user_no_duplicate_roles() -> None:
    user = User("carol")
    role = BUILTIN_ROLES["viewer"]
    user.assign_role(role)
    user.assign_role(role)
    assert user.role_names().count("viewer") == 1


# ── AccessControl ─────────────────────────────────────────────────────────────


def test_access_control_can_true() -> None:
    ac = AccessControl()
    user = User("alice", roles=[BUILTIN_ROLES["analyst"]])
    assert ac.can(user, "execute", "agents")


def test_access_control_can_false() -> None:
    ac = AccessControl()
    user = User("alice", roles=[BUILTIN_ROLES["viewer"]])
    assert not ac.can(user, "delete", "documents")


def test_access_control_can_multiple_roles() -> None:
    ac = AccessControl()
    user = User("dave", roles=[BUILTIN_ROLES["viewer"], BUILTIN_ROLES["analyst"]])
    assert ac.can(user, "execute", "agents")  # from analyst
    assert ac.can(user, "read", "documents")  # from viewer


def test_access_control_user_no_roles_denied() -> None:
    ac = AccessControl()
    user = User("nobody")
    assert not ac.can(user, "read", "documents")


def test_access_control_require_passes() -> None:
    ac = AccessControl()
    user = User("alice", roles=[BUILTIN_ROLES["admin"]])
    ac.require(user, "delete", "documents")  # should not raise


def test_access_control_require_raises() -> None:
    ac = AccessControl()
    user = User("guest", roles=[BUILTIN_ROLES["viewer"]])
    with pytest.raises(PermissionDeniedError, match="execute:agents"):
        ac.require(user, "execute", "agents")


def test_access_control_permissions_for_union() -> None:
    ac = AccessControl()
    user = User("multi", roles=[BUILTIN_ROLES["viewer"], BUILTIN_ROLES["auditor"]])
    perms = ac.permissions_for(user)
    perm_strings = {str(p) for p in perms}
    assert "read:documents" in perm_strings
    assert "read:audit" in perm_strings
