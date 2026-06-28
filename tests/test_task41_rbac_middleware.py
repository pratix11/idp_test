"""Tests for Task 41: RBAC middleware on FastAPI endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from property_intel.api.app import _get_copilot, _get_search, app, get_current_user
from property_intel.db.session import get_session
from property_intel.copilot.context_builder import Citation
from property_intel.copilot.service import CopilotAnswer
from property_intel.enterprise.rbac import BUILTIN_ROLES, User


# ── fixtures ───────────────────────────────────────────────────────────────────

def _mock_copilot() -> MagicMock:
    mock = MagicMock()
    result = CopilotAnswer(
        answer="Answer.",
        citations=[Citation(index=1, chunk_id=1, document_id=1, section_title=None, content_snippet="x")],
    )
    mock.ask.return_value = result
    mock.summarize.return_value = result
    mock.compare.return_value = result
    return mock


def _mock_search() -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = MagicMock(items=[])
    return mock


def _mock_session() -> MagicMock:
    """Session mock that satisfies the health check's SELECT 1."""
    mock = MagicMock()
    mock.execute.return_value = None
    return mock


@pytest.fixture()
def base_client() -> Iterator[TestClient]:
    """Client with mocked services but NO user override — tests RBAC headers."""
    app.dependency_overrides[_get_copilot] = lambda: _mock_copilot()
    app.dependency_overrides[_get_search] = lambda: _mock_search()
    app.dependency_overrides[get_session] = lambda: _mock_session()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ── /api/v1/me ────────────────────────────────────────────────────────────────

def test_me_analyst(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/me", headers={"X-User-Role": "analyst"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "analyst"
    assert "execute:agents" in data["permissions"]
    assert "execute:copilot" in data["permissions"]
    assert "read:search" in data["permissions"]


def test_me_viewer(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/me", headers={"X-User-Role": "viewer"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "viewer"
    assert "execute:copilot" not in data["permissions"]
    assert "execute:agents" not in data["permissions"]


def test_me_unknown_role_returns_401(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/me", headers={"X-User-Role": "hacker"})
    assert resp.status_code == 401


def test_me_default_role_is_viewer(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/me")  # no header → defaults to viewer
    assert resp.status_code == 200
    assert resp.json()["role"] == "viewer"


# ── /api/v1/search ────────────────────────────────────────────────────────────

def test_search_allowed_for_viewer(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/search?q=test", headers={"X-User-Role": "viewer"})
    assert resp.status_code == 200


def test_search_allowed_for_analyst(base_client: TestClient) -> None:
    resp = base_client.get("/api/v1/search?q=test", headers={"X-User-Role": "analyst"})
    assert resp.status_code == 200


# ── /api/v1/ask — permission enforcement ──────────────────────────────────────

def test_ask_allowed_for_analyst(base_client: TestClient) -> None:
    resp = base_client.post(
        "/api/v1/ask",
        json={"question": "test"},
        headers={"X-User-Role": "analyst"},
    )
    assert resp.status_code == 200


def test_ask_blocked_for_viewer(base_client: TestClient) -> None:
    resp = base_client.post(
        "/api/v1/ask",
        json={"question": "test"},
        headers={"X-User-Role": "viewer"},
    )
    assert resp.status_code == 403


def test_ask_allowed_for_admin(base_client: TestClient) -> None:
    resp = base_client.post(
        "/api/v1/ask",
        json={"question": "test"},
        headers={"X-User-Role": "admin"},
    )
    assert resp.status_code == 200


# ── /api/v1/agent — permission enforcement ────────────────────────────────────

def test_agent_blocked_for_viewer(base_client: TestClient) -> None:
    resp = base_client.post(
        "/api/v1/agent",
        json={"task": "summarize MahaRERA"},
        headers={"X-User-Role": "viewer"},
    )
    assert resp.status_code == 403


def test_agent_blocked_for_auditor(base_client: TestClient) -> None:
    resp = base_client.post(
        "/api/v1/agent",
        json={"task": "summarize MahaRERA"},
        headers={"X-User-Role": "auditor"},
    )
    assert resp.status_code == 403


# ── /health is always public ───────────────────────────────────────────────────

def test_health_requires_no_auth(base_client: TestClient) -> None:
    resp = base_client.get("/health")
    assert resp.status_code == 200


# ── get_current_user unit tests ────────────────────────────────────────────────

def test_get_current_user_analyst() -> None:
    user = get_current_user("analyst")
    assert user.role_names() == ["analyst"]


def test_get_current_user_default_is_viewer() -> None:
    user = get_current_user("viewer")
    assert user.role_names() == ["viewer"]


def test_get_current_user_admin() -> None:
    user = get_current_user("admin")
    from property_intel.enterprise.rbac import AccessControl
    ac = AccessControl()
    assert ac.can(user, "execute", "agents")
    assert ac.can(user, "admin", "users")
