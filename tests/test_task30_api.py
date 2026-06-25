"""Tests for Task 30: FastAPI copilot endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from property_intel.api.app import app, _get_copilot
from property_intel.copilot.context_builder import Citation
from property_intel.copilot.service import CopilotAnswer


def _citation() -> Citation:
    return Citation(
        index=1,
        chunk_id=1,
        document_id=10,
        section_title="Sec 4",
        content_snippet="snippet",
    )


def _mock_copilot(
    answer: str = "Test answer.",
    stream_chunks: list[str] | None = None,
) -> MagicMock:
    mock = MagicMock()
    result = CopilotAnswer(answer=answer, citations=[_citation()])
    mock.ask.return_value = result
    mock.summarize.return_value = result
    mock.compare.return_value = result
    mock.stream_ask.return_value = iter(stream_chunks or ["Hello", " world"])
    mock.stream_summarize.return_value = iter(stream_chunks or ["Sum", "mary"])
    mock.stream_compare.return_value = iter(stream_chunks or ["Diff"])
    return mock


@pytest.fixture()
def client() -> Iterator[TestClient]:
    mock_copilot = _mock_copilot()
    app.dependency_overrides[_get_copilot] = lambda: mock_copilot
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── POST /api/v1/ask ──────────────────────────────────────────────────────────

def test_ask_returns_answer(client: TestClient) -> None:
    resp = client.post("/api/v1/ask", json={"question": "What is Section 4?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer."


def test_ask_returns_citations(client: TestClient) -> None:
    resp = client.post("/api/v1/ask", json={"question": "What is Section 4?"})
    citations = resp.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["index"] == 1
    assert citations[0]["document_id"] == 10


def test_ask_empty_question_returns_422(client: TestClient) -> None:
    resp = client.post("/api/v1/ask", json={"question": ""})
    assert resp.status_code == 422


# ── POST /api/v1/summarize ────────────────────────────────────────────────────

def test_summarize_returns_200(client: TestClient) -> None:
    resp = client.post("/api/v1/summarize", json={"query": "builder rules"})
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_summarize_empty_query_returns_422(client: TestClient) -> None:
    resp = client.post("/api/v1/summarize", json={"query": ""})
    assert resp.status_code == 422


# ── POST /api/v1/compare ─────────────────────────────────────────────────────

def test_compare_returns_200(client: TestClient) -> None:
    resp = client.post("/api/v1/compare", json={"query_a": "2020 rules", "query_b": "2023 rules"})
    assert resp.status_code == 200


def test_compare_missing_query_b_returns_422(client: TestClient) -> None:
    resp = client.post("/api/v1/compare", json={"query_a": "only a"})
    assert resp.status_code == 422


# ── POST /api/v1/ask/stream ───────────────────────────────────────────────────

def test_ask_stream_returns_event_stream(client: TestClient) -> None:
    resp = client.post("/api/v1/ask/stream", json={"question": "What is RAG?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_ask_stream_contains_sse_chunks(client: TestClient) -> None:
    resp = client.post("/api/v1/ask/stream", json={"question": "RAG?"})
    body = resp.text
    assert "data:" in body
    assert "[DONE]" in body


# ── POST /api/v1/summarize/stream ─────────────────────────────────────────────

def test_summarize_stream_returns_event_stream(client: TestClient) -> None:
    resp = client.post("/api/v1/summarize/stream", json={"query": "topic"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


# ── POST /api/v1/compare/stream ───────────────────────────────────────────────

def test_compare_stream_returns_event_stream(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/compare/stream",
        json={"query_a": "2020", "query_b": "2023"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
