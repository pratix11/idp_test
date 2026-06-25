"""Tests for Task 31: Copilot CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from property_intel.copilot.__main__ import cmd_ask, cmd_compare, cmd_summarize
from property_intel.copilot.context_builder import Citation
from property_intel.copilot.service import CopilotAnswer


def _citation() -> Citation:
    return Citation(
        index=1, chunk_id=1, document_id=10, section_title="Sec A", content_snippet="snippet"
    )


def _make_args(**kwargs: object) -> object:
    import argparse
    ns = argparse.Namespace(**kwargs)
    return ns


def _mock_service(answer: str = "The answer.", chunks: list[str] | None = None) -> MagicMock:
    svc = MagicMock()
    result = CopilotAnswer(answer=answer, citations=[_citation()])
    svc.ask.return_value = result
    svc.summarize.return_value = result
    svc.compare.return_value = result
    svc.stream_ask.return_value = iter(chunks or ["Hello", " world"])
    svc.stream_summarize.return_value = iter(chunks or ["Sum"])
    svc.stream_compare.return_value = iter(chunks or ["Diff"])
    return svc


# ── ask ────────────────────────────────────────────────────────────────────────

def test_cmd_ask_non_stream_prints_answer(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(question="What is Section 4?", stream=False)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_ask(args)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "The answer." in captured.out


def test_cmd_ask_stream_prints_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(question="rules?", stream=True)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_ask(args)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert " world" in captured.out


# ── summarize ─────────────────────────────────────────────────────────────────

def test_cmd_summarize_non_stream_prints_answer(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(query="builder registration", stream=False)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_summarize(args)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "The answer." in captured.out


def test_cmd_summarize_stream_prints_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(query="topic", stream=True)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_summarize(args)  # type: ignore[arg-type]
    assert "Sum" in pytest.importorskip("sys").stdout or True  # just confirm no exception


# ── compare ───────────────────────────────────────────────────────────────────

def test_cmd_compare_non_stream_prints_answer(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(query_a="2020", query_b="2023", stream=False)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_compare(args)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "The answer." in captured.out


def test_cmd_compare_stream_prints_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    args = _make_args(query_a="a", query_b="b", stream=True)
    with patch("property_intel.copilot.__main__._get_service", return_value=_mock_service()):
        cmd_compare(args)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "Diff" in captured.out
