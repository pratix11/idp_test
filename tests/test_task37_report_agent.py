"""Tests for Phase 5 Report Agent (Task 38).

Tests cover:
- Instantiation and from_retrieval factory
- run() returns a string
- generate() builds correct task with all sections mentioned
- generate() with scope constraint includes scope in task
- Tool-calling path
- Report sections present in mock output
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from property_intel.agents.report import ReportAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_retrieval() -> MagicMock:
    chunk = MagicMock()
    chunk.content = "MahaRERA registration is mandatory for all projects above 500 sqm."
    chunk.section_title = "Section 3"
    svc = MagicMock()
    svc.search.return_value = [chunk]
    return svc


def _llm_direct(answer: str = "## Executive Summary\nThis is the report.") -> MagicMock:
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(content=answer)
    llm.bind_tools.return_value = bound
    return llm


def _llm_tool_then_answer(answer: str) -> MagicMock:
    llm = MagicMock()
    bound = MagicMock()
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "summarize_topic", "args": {"topic": "registration"}, "id": "t1", "type": "tool_call"}],
    )
    bound.invoke.side_effect = [tool_msg, AIMessage(content=answer)]
    llm.bind_tools.return_value = bound
    return llm


_FULL_REPORT = """\
## Executive Summary
MahaRERA mandates builder registration.

## Background
The Real Estate (Regulation and Development) Act 2016 established MahaRERA.

## Key Findings
- All projects above 500 sqm must register.
- Penalty: Rs. 10 lakh per day of default.

## Applicable Regulations
Section 3, Section 59.

## Risk Assessment
High risk for unregistered projects.

## Recommendations
Register immediately and maintain escrow account.
"""


# ── Instantiation ─────────────────────────────────────────────────────────────


def test_report_agent_instantiates() -> None:
    agent = ReportAgent.from_retrieval(_mock_retrieval(), _llm_direct())
    assert isinstance(agent, ReportAgent)


def test_from_retrieval_binds_three_tools() -> None:
    llm = _llm_direct()
    ReportAgent.from_retrieval(_mock_retrieval(), llm)
    tools = llm.bind_tools.call_args[0][0]
    names = {t.name for t in tools}
    assert names == {"retrieve_documents", "summarize_topic", "extract_entities"}


# ── run() ─────────────────────────────────────────────────────────────────────


def test_run_returns_string() -> None:
    agent = ReportAgent.from_retrieval(_mock_retrieval(), _llm_direct())
    assert isinstance(agent.run("Generate report on builder obligations."), str)


# ── generate() ───────────────────────────────────────────────────────────────


def test_generate_returns_string() -> None:
    agent = ReportAgent.from_retrieval(_mock_retrieval(), _llm_direct(_FULL_REPORT))
    result = agent.generate("MahaRERA builder obligations")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_includes_topic_in_task() -> None:
    llm = _llm_direct("Report done.")
    agent = ReportAgent.from_retrieval(_mock_retrieval(), llm)
    agent.generate("MahaRERA registration requirements")
    call_messages = llm.bind_tools.return_value.invoke.call_args[0][0]
    combined = " ".join(str(m.content) for m in call_messages)
    assert "MahaRERA registration requirements" in combined


def test_generate_with_scope_includes_scope() -> None:
    llm = _llm_direct("Scoped report.")
    agent = ReportAgent.from_retrieval(_mock_retrieval(), llm)
    agent.generate("Penalty provisions", scope="residential projects only")
    call_messages = llm.bind_tools.return_value.invoke.call_args[0][0]
    combined = " ".join(str(m.content) for m in call_messages)
    assert "residential projects only" in combined


def test_generate_section_headers_in_output() -> None:
    agent = ReportAgent.from_retrieval(_mock_retrieval(), _llm_direct(_FULL_REPORT))
    result = agent.generate("builder report")
    assert "Executive Summary" in result
    assert "Key Findings" in result
    assert "Recommendations" in result


# ── Tool call path ────────────────────────────────────────────────────────────


def test_run_with_tool_call_invokes_llm_twice() -> None:
    llm = _llm_tool_then_answer(_FULL_REPORT)
    agent = ReportAgent.from_retrieval(_mock_retrieval(), llm)
    agent.run("Draft a compliance report.")
    assert llm.bind_tools.return_value.invoke.call_count == 2
