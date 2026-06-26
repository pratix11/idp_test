"""Tests for Phase 5 Research Agent (Task 37).

Tests cover:
- Instantiation and from_retrieval factory (3 tools: retrieve, summarize, extract)
- run() returns a string
- research() builds correct task and delegates
- Multiple tool calls simulate multi-document research loop
- Failure handling for empty corpus
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from property_intel.agents.research import ResearchAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_retrieval(content: str = "Regulatory text.") -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.section_title = "Sec 1"
    svc = MagicMock()
    svc.search.return_value = [chunk]
    return svc


def _llm_direct(answer: str = "Research brief.") -> MagicMock:
    llm = MagicMock()
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(content=answer)
    llm.bind_tools.return_value = bound
    return llm


def _llm_two_tools_then_answer(answer: str) -> MagicMock:
    """Simulate LLM making two tool calls before giving final answer."""
    llm = MagicMock()
    bound = MagicMock()
    call1 = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve_documents", "args": {"query": "builder registration", "limit": 5}, "id": "t1", "type": "tool_call"}],
    )
    call2 = AIMessage(
        content="",
        tool_calls=[{"name": "summarize_topic", "args": {"topic": "penalties"}, "id": "t2", "type": "tool_call"}],
    )
    final = AIMessage(content=answer)
    bound.invoke.side_effect = [call1, call2, final]
    llm.bind_tools.return_value = bound
    return llm


# ── Instantiation ─────────────────────────────────────────────────────────────


def test_research_agent_instantiates() -> None:
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), _llm_direct())
    assert isinstance(agent, ResearchAgent)


def test_from_retrieval_binds_three_tools() -> None:
    llm = _llm_direct()
    ResearchAgent.from_retrieval(_mock_retrieval(), llm)
    tools = llm.bind_tools.call_args[0][0]
    names = {t.name for t in tools}
    assert names == {"retrieve_documents", "summarize_topic", "extract_entities"}


# ── run() ─────────────────────────────────────────────────────────────────────


def test_run_returns_string() -> None:
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), _llm_direct("Brief."))
    assert isinstance(agent.run("What are MahaRERA builder obligations?"), str)


def test_run_with_direct_answer() -> None:
    brief = "Key Findings: Builders must register."
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), _llm_direct(brief))
    result = agent.run("Research question")
    assert "Builders must register" in result


# ── research() ────────────────────────────────────────────────────────────────


def test_research_method_includes_question_in_messages() -> None:
    llm = _llm_direct("Found it.")
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), llm)
    agent.research("What are the escrow requirements?")
    call_messages = llm.bind_tools.return_value.invoke.call_args[0][0]
    combined = " ".join(str(m.content) for m in call_messages)
    assert "escrow" in combined.lower()


def test_research_returns_string() -> None:
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), _llm_direct())
    result = agent.research("Comprehensive question")
    assert isinstance(result, str)


# ── Multi-tool loop ────────────────────────────────────────────────────────────


def test_multi_tool_loop_invokes_llm_three_times() -> None:
    brief = "Research Brief: findings from multiple sources."
    llm = _llm_two_tools_then_answer(brief)
    agent = ResearchAgent.from_retrieval(_mock_retrieval(), llm)
    result = agent.run("Research builder obligations and penalties.")
    assert llm.bind_tools.return_value.invoke.call_count == 3
    assert isinstance(result, str)


# ── Failure handling ──────────────────────────────────────────────────────────


def test_empty_corpus_returns_string() -> None:
    empty = MagicMock()
    empty.search.return_value = []
    agent = ResearchAgent.from_retrieval(empty, _llm_direct("No data available."))
    result = agent.run("Any question")
    assert isinstance(result, str)
