"""Tests for Task 40: POST /api/v1/agent endpoint (Phase 5 LangGraph agents)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from property_intel.api.app import _get_agent_router, app, get_current_user
from property_intel.enterprise.rbac import BUILTIN_ROLES, User


def _admin_user() -> User:
    return User(user_id="test-admin", roles=[BUILTIN_ROLES["admin"]])


def _mock_router(answer: str = "Agent answer.", agent_name: str = "document_analyst") -> MagicMock:
    mock = MagicMock()
    mock._classify.return_value = agent_name
    mock.route.return_value = answer
    return mock


@pytest.fixture()
def client() -> Iterator[TestClient]:
    mock_router = _mock_router()
    app.dependency_overrides[_get_agent_router] = lambda: mock_router
    app.dependency_overrides[get_current_user] = _admin_user
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def comparison_client() -> Iterator[TestClient]:
    mock_router = _mock_router(
        answer="Comparison result.", agent_name="comparison"
    )
    app.dependency_overrides[_get_agent_router] = lambda: mock_router
    app.dependency_overrides[get_current_user] = _admin_user
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ── POST /api/v1/agent ─────────────────────────────────────────────────────────

def test_agent_returns_answer(client: TestClient) -> None:
    resp = client.post("/api/v1/agent", json={"task": "Summarize the MahaRERA act."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Agent answer."
    assert data["agent"] == "document_analyst"


def test_agent_comparison_routing(comparison_client: TestClient) -> None:
    resp = comparison_client.post(
        "/api/v1/agent", json={"task": "Compare MahaRERA vs MHADA rules."}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Comparison result."
    assert data["agent"] == "comparison"


def test_agent_empty_task_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/agent", json={"task": ""})
    assert resp.status_code == 422


def test_agent_missing_task_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/agent", json={})
    assert resp.status_code == 422


def test_agent_calls_route_with_task(client: TestClient) -> None:
    from property_intel.api.app import _get_agent_router

    mock_router = _mock_router()
    app.dependency_overrides[_get_agent_router] = lambda: mock_router
    app.dependency_overrides[get_current_user] = _admin_user
    try:
        with TestClient(app) as c:
            c.post("/api/v1/agent", json={"task": "What are MHADA regulations?"})
        mock_router.route.assert_called_once_with("What are MHADA regulations?")
    finally:
        app.dependency_overrides.clear()
