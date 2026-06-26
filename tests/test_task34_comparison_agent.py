"""Tests for Phase 5 Regulation Comparison Agent (Task 35).

Tests cover:
- Agent instantiation and from_retrieval factory
- run() returns text result
- compare() builds correct task string and delegates to run()
- Tool-calling path executes retrieve_documents
- No-tool path works
- Failure handling: empty retrieval still returns a string
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from property_intel.agents.comparison import ComparisonAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_retrieval(content: str = "Rule text") -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.section_title = "Section A"
    svc = MagicMock()
    svc.search.return_value = [chunk]
    return svc


def _mock_llm_direct(answer: str = "Comparison result.") -> MagicMock:
    llm = MagicMock()
    response = AIMessage(content=answer)
    bound = MagicMock()
    bound.invoke.return_value = response
    llm.bind_tools.return_value = bound
    return llm


def _mock_llm_tool_then_answer(answer: str = "Differences found.") -> MagicMock:
    llm = MagicMock()
    bound = MagicMock()
    tool_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "retrieve_documents",
            "args": {"query": "builder registration rules", "limit": 5},
            "id": "c001",
            "type": "tool_call",
        }],
    )
    final_msg = AIMessage(content=answer)
    bound.invoke.side_effect = [tool_msg, final_msg]
    llm.bind_tools.return_value = bound
    return llm


# ── Instantiation ─────────────────────────────────────────────────────────────


def test_comparison_agent_instantiates() -> None:
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), _mock_llm_direct())
    assert agent is not None
    assert isinstance(agent, ComparisonAgent)


def test_from_retrieval_binds_one_tool() -> None:
    llm = _mock_llm_direct()
    ComparisonAgent.from_retrieval(_mock_retrieval(), llm)
    tools_list = llm.bind_tools.call_args[0][0]
    assert len(tools_list) == 1
    assert tools_list[0].name == "retrieve_documents"


# ── run() ─────────────────────────────────────────────────────────────────────


def test_run_returns_string() -> None:
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), _mock_llm_direct())
    result = agent.run("Compare 2016 rules with 2019 amendments.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_run_no_tool_call_returns_direct_answer() -> None:
    expected = "The 2019 rules add stricter penalties."
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), _mock_llm_direct(expected))
    result = agent.run("Compare regulations.")
    assert expected in result


def test_run_with_tool_call_invokes_llm_twice() -> None:
    llm = _mock_llm_tool_then_answer("Key differences: ...")
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), llm)
    result = agent.run("Compare Section 3 vs Section 5.")
    assert llm.bind_tools.return_value.invoke.call_count == 2
    assert isinstance(result, str)


# ── compare() convenience method ─────────────────────────────────────────────


def test_compare_builds_task_from_subjects() -> None:
    llm = _mock_llm_direct("Comparison done.")
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), llm)
    result = agent.compare("MahaRERA 2016", "MahaRERA 2019")
    # Should invoke LLM with task containing both subjects
    call_messages = llm.bind_tools.return_value.invoke.call_args[0][0]
    combined = " ".join(str(m.content) for m in call_messages if hasattr(m, "content"))
    assert "MahaRERA 2016" in combined
    assert "MahaRERA 2019" in combined


def test_compare_returns_string() -> None:
    agent = ComparisonAgent.from_retrieval(_mock_retrieval(), _mock_llm_direct())
    result = agent.compare("subject A", "subject B")
    assert isinstance(result, str)


# ── Failure handling ──────────────────────────────────────────────────────────


def test_run_empty_retrieval_returns_string() -> None:
    empty_retrieval = MagicMock()
    empty_retrieval.search.return_value = []
    agent = ComparisonAgent.from_retrieval(empty_retrieval, _mock_llm_direct("No data found."))
    result = agent.run("Compare missing topics.")
    assert isinstance(result, str)
