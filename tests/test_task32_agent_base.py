"""Tests for Phase 5 agent base infrastructure (Task 32/33).

Tests cover:
- AgentState structure and typing
- BaseAgent._should_continue edge logic
- Tool factory functions produce callable tools
- AgentRouter keyword classification
- AgentRouter dispatches to the registered agent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from property_intel.agents.base import AgentState, BaseAgent
from property_intel.agents.router import AgentRouter
from property_intel.agents.tools import (
    make_extract_tool,
    make_retrieve_tool,
    make_summarize_tool,
)


# ── AgentState ──────────────────────────────────────────────────────────────


def test_agent_state_fields_exist() -> None:
    state: AgentState = {
        "messages": [],
        "task": "test task",
        "result": None,
        "metadata": {},
    }
    assert state["task"] == "test task"
    assert state["result"] is None
    assert state["metadata"] == {}
    assert state["messages"] == []


def test_agent_state_accepts_metadata() -> None:
    state: AgentState = {
        "messages": [],
        "task": "check compliance",
        "result": "done",
        "metadata": {"doc_id": "abc123", "score": 0.95},
    }
    assert state["metadata"]["doc_id"] == "abc123"


# ── BaseAgent._should_continue ───────────────────────────────────────────────


def test_should_continue_returns_end_when_no_tool_calls() -> None:
    ai_message = AIMessage(content="Here is my final answer.")
    state: AgentState = {
        "messages": [ai_message],
        "task": "t",
        "result": None,
        "metadata": {},
    }
    result = BaseAgent._should_continue(state)
    assert result == "END" or result == "__end__" or "end" in str(result).lower()


def test_should_continue_returns_tools_when_tool_calls_present() -> None:
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve_documents", "args": {"query": "q"}, "id": "1", "type": "tool_call"}],
    )
    state: AgentState = {
        "messages": [ai_message],
        "task": "t",
        "result": None,
        "metadata": {},
    }
    result = BaseAgent._should_continue(state)
    assert result == "tools"


# ── Tool factories ───────────────────────────────────────────────────────────


def _mock_retrieval(chunks: list[tuple[str, str]]) -> MagicMock:
    """Build a mock RetrievalService that returns fake ScoredChunks."""
    mock = MagicMock()
    scored_chunks = []
    for content, section in chunks:
        chunk = MagicMock()
        chunk.content = content
        chunk.section_title = section
        scored_chunks.append(chunk)
    mock.search.return_value = scored_chunks
    return mock


def test_retrieve_tool_returns_formatted_chunks() -> None:
    retrieval = _mock_retrieval([
        ("Builder must register within 30 days.", "Section 3"),
        ("Fee is Rs. 10,000.", "Section 4"),
    ])
    tool_fn = make_retrieve_tool(retrieval)
    result = tool_fn.invoke({"query": "registration rules", "limit": 2})
    assert "[1]" in result
    assert "Section 3" in result
    assert "Builder must register" in result


def test_retrieve_tool_empty_returns_message() -> None:
    retrieval = _mock_retrieval([])
    tool_fn = make_retrieve_tool(retrieval)
    result = tool_fn.invoke({"query": "nonexistent", "limit": 5})
    assert "No relevant documents found" in result


def test_summarize_tool_returns_context() -> None:
    retrieval = _mock_retrieval([("Rule A applies to all builders.", "Sec 1")])
    tool_fn = make_summarize_tool(retrieval)
    result = tool_fn.invoke({"topic": "builder rules"})
    assert "Rule A applies" in result


def test_extract_tool_returns_passages() -> None:
    retrieval = _mock_retrieval([("Penalty: Rs. 50,000 per day.", "Sec 7")])
    tool_fn = make_extract_tool(retrieval)
    result = tool_fn.invoke({"query": "penalties"})
    assert "Penalty" in result


def test_tools_have_langchain_name_attribute() -> None:
    retrieval = _mock_retrieval([])
    for factory in [make_retrieve_tool, make_summarize_tool, make_extract_tool]:
        t = factory(retrieval)
        assert hasattr(t, "name")
        assert isinstance(t.name, str)


# ── AgentRouter ──────────────────────────────────────────────────────────────


def test_router_classify_comparison() -> None:
    router = AgentRouter({})
    assert router._classify("compare two documents") == "comparison"


def test_router_classify_compliance() -> None:
    router = AgentRouter({})
    assert router._classify("check compliance with the new regulation") == "compliance"


def test_router_classify_report() -> None:
    router = AgentRouter({})
    assert router._classify("generate report for this document") == "report"


def test_router_classify_research() -> None:
    router = AgentRouter({})
    assert router._classify("research across multiple documents") == "research"


def test_router_classify_fallback_to_document_analyst() -> None:
    router = AgentRouter({})
    assert router._classify("what is the deadline?") == "document_analyst"


def test_router_dispatches_to_agent() -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "analysis result"
    router = AgentRouter({"document_analyst": mock_agent})
    result = router.route("summarize the registration rules")
    mock_agent.run.assert_called_once()
    assert result == "analysis result"


def test_router_raises_when_agent_not_registered() -> None:
    router = AgentRouter({})
    with pytest.raises(KeyError, match="document_analyst"):
        router.route("summarize something")
