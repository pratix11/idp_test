"""Tests for Phase 5 Compliance Agent (Task 36).

Tests cover:
- Instantiation and from_retrieval factory
- run() returns a verdict string
- validate() builds correct task and delegates
- Tool-calling path (retrieve + extract)
- Verdict keywords in output (COMPLIANT / NON-COMPLIANT / NEEDS REVIEW)
- Empty retrieval handled gracefully
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from property_intel.agents.compliance import ComplianceAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_retrieval() -> MagicMock:
    chunk = MagicMock()
    chunk.content = "Builders must register before launch per Section 3."
    chunk.section_title = "Section 3"
    svc = MagicMock()
    svc.search.return_value = [chunk]
    return svc


def _llm_direct(answer: str) -> MagicMock:
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
        tool_calls=[{
            "name": "retrieve_documents",
            "args": {"query": "registration requirement", "limit": 5},
            "id": "tc1",
            "type": "tool_call",
        }],
    )
    bound.invoke.side_effect = [tool_msg, AIMessage(content=answer)]
    llm.bind_tools.return_value = bound
    return llm


# ── Instantiation ─────────────────────────────────────────────────────────────


def test_compliance_agent_instantiates() -> None:
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), _llm_direct("COMPLIANT"))
    assert isinstance(agent, ComplianceAgent)


def test_from_retrieval_binds_two_tools() -> None:
    llm = _llm_direct("ok")
    ComplianceAgent.from_retrieval(_mock_retrieval(), llm)
    tools = llm.bind_tools.call_args[0][0]
    names = {t.name for t in tools}
    assert "retrieve_documents" in names
    assert "extract_entities" in names


# ── run() and validate() ──────────────────────────────────────────────────────


def test_run_returns_string() -> None:
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), _llm_direct("COMPLIANT."))
    assert isinstance(agent.run("Builder launched without registration."), str)


def test_run_with_compliant_verdict() -> None:
    verdict = "COMPLIANT: Builder has registered on time as per Section 3."
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), _llm_direct(verdict))
    result = agent.run("Builder registered on day 25.")
    assert "COMPLIANT" in result


def test_run_with_non_compliant_verdict() -> None:
    verdict = "NON-COMPLIANT: Registration deadline missed by 10 days."
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), _llm_direct(verdict))
    result = agent.run("Builder launched on day 45 without registration.")
    assert "NON-COMPLIANT" in result


def test_validate_includes_situation_in_task() -> None:
    llm = _llm_direct("NEEDS REVIEW")
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), llm)
    situation = "Builder collected advance payment without escrow."
    agent.validate(situation)
    call_messages = llm.bind_tools.return_value.invoke.call_args[0][0]
    combined = " ".join(str(m.content) for m in call_messages if hasattr(m, "content"))
    assert "escrow" in combined.lower() or "advance" in combined.lower()


def test_run_with_tool_call_loops() -> None:
    llm = _llm_tool_then_answer("NON-COMPLIANT: missing registration.")
    agent = ComplianceAgent.from_retrieval(_mock_retrieval(), llm)
    result = agent.run("Check builder XYZ registration status.")
    assert llm.bind_tools.return_value.invoke.call_count == 2
    assert isinstance(result, str)


def test_run_empty_retrieval_returns_string() -> None:
    empty = MagicMock()
    empty.search.return_value = []
    agent = ComplianceAgent.from_retrieval(empty, _llm_direct("NEEDS REVIEW: no data."))
    result = agent.run("Validate situation.")
    assert isinstance(result, str)
