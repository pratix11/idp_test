"""Tests for Task 27: ContextBuilder."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from property_intel.copilot.context_builder import BuiltContext, Citation, ContextBuilder
from property_intel.retrieval.models import ScoredChunk


def _chunk(
    chunk_id: int = 1,
    document_id: int = 10,
    content: str = "some content",
    section_title: str | None = "Section A",
    score: float = 0.9,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content=content,
        section_title=section_title,
        score=score,
    )


# ── empty input ────────────────────────────────────────────────────────────────

def test_build_empty_list_returns_empty_context() -> None:
    result = ContextBuilder().build([])
    assert result.context == ""
    assert result.citations == []
    assert result.token_count == 0


# ── formatting ─────────────────────────────────────────────────────────────────

def test_build_single_chunk_contains_citation_marker() -> None:
    chunk = _chunk(content="Builder must register.", section_title="Registration")
    result = ContextBuilder().build([chunk])
    assert "[1] Section: Registration" in result.context
    assert "Builder must register." in result.context


def test_build_null_section_uses_dash() -> None:
    chunk = _chunk(section_title=None)
    result = ContextBuilder().build([chunk])
    assert "[1] Section: —" in result.context


def test_build_multiple_chunks_are_numbered_sequentially() -> None:
    chunks = [_chunk(chunk_id=i, content=f"text {i}") for i in range(1, 4)]
    result = ContextBuilder().build(chunks)
    assert "[1]" in result.context
    assert "[2]" in result.context
    assert "[3]" in result.context


def test_build_chunks_separated_by_double_newline() -> None:
    chunks = [_chunk(chunk_id=1, content="first"), _chunk(chunk_id=2, content="second")]
    result = ContextBuilder().build(chunks)
    assert "\n\n" in result.context


# ── citations ─────────────────────────────────────────────────────────────────

def test_build_returns_citation_per_chunk() -> None:
    chunks = [_chunk(chunk_id=i) for i in range(1, 4)]
    result = ContextBuilder().build(chunks)
    assert len(result.citations) == 3


def test_build_citation_index_matches_context_marker() -> None:
    chunk = _chunk(chunk_id=42, document_id=7, section_title="Fees")
    result = ContextBuilder().build([chunk])
    assert result.citations[0].index == 1
    assert result.citations[0].chunk_id == 42
    assert result.citations[0].document_id == 7
    assert result.citations[0].section_title == "Fees"


def test_build_citation_snippet_capped_at_200_chars() -> None:
    long_content = "x" * 300
    chunk = _chunk(content=long_content)
    result = ContextBuilder().build([chunk])
    assert len(result.citations[0].content_snippet) == 200


# ── token budget ──────────────────────────────────────────────────────────────

def test_build_respects_token_budget() -> None:
    # Each chunk token count mocked to 100; budget = 150 → only first fits
    chunks = [_chunk(chunk_id=i, content=f"chunk {i}") for i in range(1, 5)]

    with patch("property_intel.copilot.context_builder._count_tokens", return_value=100):
        result = ContextBuilder(max_context_tokens=150).build(chunks)

    assert len(result.citations) == 1
    assert "[2]" not in result.context


def test_build_token_count_matches_included_chunks() -> None:
    chunks = [_chunk(chunk_id=i) for i in range(1, 3)]

    with patch("property_intel.copilot.context_builder._count_tokens", return_value=50):
        result = ContextBuilder(max_context_tokens=200).build(chunks)

    assert result.token_count == 100  # 2 chunks × 50 tokens each


# ── document title in context ──────────────────────────────────────────────────

def test_build_includes_document_title_in_header() -> None:
    chunk = ScoredChunk(
        chunk_id=1,
        document_id=10,
        chunk_index=0,
        content="Builder must register.",
        section_title="Registration",
        score=0.9,
        document_title="Real_Estate_Act_2016",
    )
    result = ContextBuilder().build([chunk])
    assert "Document: Real_Estate_Act_2016" in result.context
    assert "Section: Registration" in result.context
    assert result.citations[0].document_title == "Real_Estate_Act_2016"


def test_build_no_document_title_uses_section_only_format() -> None:
    chunk = _chunk(section_title="Fees")
    result = ContextBuilder().build([chunk])
    assert "[1] Section: Fees" in result.context
    assert "Document:" not in result.context
    assert result.citations[0].document_title is None
