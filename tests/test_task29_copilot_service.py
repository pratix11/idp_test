"""Tests for Task 29: CopilotService facade."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from property_intel.copilot.context_builder import BuiltContext, Citation
from property_intel.copilot.service import CopilotAnswer, CopilotService
from property_intel.retrieval.models import ScoredChunk


def _citation(index: int = 1) -> Citation:
    return Citation(
        index=index,
        chunk_id=index,
        document_id=10,
        section_title="Section X",
        content_snippet="snippet",
    )


def _chunk(chunk_id: int = 1) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=10,
        chunk_index=0,
        content="sample content",
        section_title="Section X",
        score=0.9,
    )


def _make_service(
    chunks: list[ScoredChunk] | None = None,
    answer: str = "The answer is X.",
) -> CopilotService:
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = chunks or [_chunk()]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = answer
    mock_llm.stream_complete.return_value = iter(["chunk1", " chunk2"])

    mock_ctx_builder = MagicMock()
    mock_ctx_builder.build.return_value = BuiltContext(
        context="[1] Section: X\nsample content",
        citations=[_citation()],
        token_count=20,
    )

    return CopilotService(
        retrieval=mock_retrieval,
        llm=mock_llm,
        context_builder=mock_ctx_builder,
    )


# ── ask() ──────────────────────────────────────────────────────────────────────

def test_ask_returns_answer_and_citations() -> None:
    svc = _make_service(answer="Builder must register.")
    result = svc.ask("What must builders do?")
    assert result.answer == "Builder must register."
    assert len(result.citations) == 1


def test_ask_blank_question_returns_empty() -> None:
    svc = _make_service()
    result = svc.ask("   ")
    assert result.answer == ""
    assert result.citations == []


def test_ask_calls_retrieval_with_question() -> None:
    svc = _make_service()
    svc.ask("builder rules?")
    svc._retrieval.search.assert_called_once()
    call_args = svc._retrieval.search.call_args
    assert call_args[0][0] == "builder rules?" or call_args[1].get("query") == "builder rules?" or "builder rules?" in str(call_args)


def test_ask_calls_llm_complete() -> None:
    svc = _make_service()
    svc.ask("question?")
    svc._llm.complete.assert_called_once()


# ── stream_ask() ───────────────────────────────────────────────────────────────

def test_stream_ask_yields_chunks() -> None:
    svc = _make_service()
    chunks = list(svc.stream_ask("question?"))
    assert chunks == ["chunk1", " chunk2"]


def test_stream_ask_blank_yields_nothing() -> None:
    svc = _make_service()
    result = list(svc.stream_ask("   "))
    assert result == []


# ── summarize() ────────────────────────────────────────────────────────────────

def test_summarize_returns_answer_and_citations() -> None:
    svc = _make_service(answer="Summary here.")
    result = svc.summarize("builder registration")
    assert result.answer == "Summary here."
    assert len(result.citations) == 1


def test_summarize_blank_query_returns_empty() -> None:
    svc = _make_service()
    result = svc.summarize("")
    assert result.answer == ""
    assert result.citations == []


def test_stream_summarize_yields_chunks() -> None:
    svc = _make_service()
    result = list(svc.stream_summarize("topic"))
    assert result == ["chunk1", " chunk2"]


# ── compare() ──────────────────────────────────────────────────────────────────

def test_compare_returns_answer() -> None:
    svc = _make_service(answer="A differs from B in X.")
    result = svc.compare("query A", "query B")
    assert result.answer == "A differs from B in X."


def test_compare_blank_query_returns_empty() -> None:
    svc = _make_service()
    result = svc.compare("", "query b")
    assert result.answer == ""


def test_compare_merges_citations_from_both_contexts() -> None:
    svc = _make_service()
    # Two separate context build calls → two citation lists merged
    ctx_a = BuiltContext("[1] text a", [_citation(1)], 10)
    ctx_b = BuiltContext("[1] text b", [_citation(2)], 10)
    svc._ctx_builder.build.side_effect = [ctx_a, ctx_b]

    result = svc.compare("query a", "query b")
    assert len(result.citations) == 2


def test_stream_compare_yields_chunks() -> None:
    svc = _make_service()
    result = list(svc.stream_compare("a", "b"))
    assert result == ["chunk1", " chunk2"]
