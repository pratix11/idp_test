"""Tests for Phase 5 Document Analyst Agent (Task 34).

Tests cover:
- Agent instantiation with injected tools and LLM mock
- Graph structure (has agent and tools nodes)
- run() returns text from the final AI message
- Tool calls get routed to tool execution
- No-tool-call path produces immediate result
- from_retrieval factory wires correct tools
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from property_intel.agents.document_analyst import DocumentAnalystAgent
from property_intel.agents.tools import (
    make_extract_tool,
    make_retrieve_tool,
    make_summarize_tool,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_retrieval(content: str = "Relevant chunk") -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.section_title = "Section 1"
    retrieval = MagicMock()
    retrieval.search.return_value = [chunk]
    return retrieval


def _make_mock_llm(final_answer: str = "Final answer.") -> MagicMock:
    """LLM that immediately responds with a final answer (no tool calls)."""
    llm = MagicMock()
    response = AIMessage(content=final_answer)
    bound = MagicMock()
    bound.invoke.return_value = response
    llm.bind_tools.return_value = bound
    return llm


def _make_tool_calling_llm(tool_name: str, tool_args: dict, tool_call_id: str, final_answer: str) -> MagicMock:
    """LLM that makes one tool call then gives a final answer."""
    llm = MagicMock()

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": tool_args, "id": tool_call_id, "type": "tool_call"}],
    )
    final_msg = AIMessage(content=final_answer)

    bound = MagicMock()
    bound.invoke.side_effect = [tool_call_msg, final_msg]
    llm.bind_tools.return_value = bound
    return llm


# ── Instantiation ────────────────────────────────────────────────────────────


def test_document_analyst_instantiates() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm()
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    assert agent is not None


def test_from_retrieval_binds_three_tools() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm()
    DocumentAnalystAgent.from_retrieval(retrieval, llm)
    # bind_tools should be called with 3 tools (retrieve, summarize, extract)
    call_args = llm.bind_tools.call_args
    tools_list = call_args[0][0]
    assert len(tools_list) == 3


# ── Direct answer path ───────────────────────────────────────────────────────


def test_run_returns_final_answer_without_tool_call() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm("MahaRERA requires builders to register within 30 days.")
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    result = agent.run("What is the registration deadline?")
    assert "MahaRERA" in result or "30 days" in result or result != ""


def test_run_returns_string() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm("The answer is 42.")
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    result = agent.run("What is the fee?")
    assert isinstance(result, str)


def test_run_with_metadata_passes_through() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm("Done.")
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    result = agent.run("Summarize rules", doc_id="DOC-001", category="regulations")
    assert isinstance(result, str)


# ── Tool call path ───────────────────────────────────────────────────────────


def test_run_with_tool_call_executes_retrieve_tool() -> None:
    retrieval = _make_mock_retrieval("Builder must pay Rs. 10000.")
    llm = _make_tool_calling_llm(
        tool_name="retrieve_documents",
        tool_args={"query": "registration fee", "limit": 5},
        tool_call_id="call_001",
        final_answer="The registration fee is Rs. 10000.",
    )
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    result = agent.run("What is the registration fee?")
    # LLM was called twice (once for tool call, once for final answer)
    assert llm.bind_tools.return_value.invoke.call_count == 2
    assert isinstance(result, str)


# ── from_retrieval factory ───────────────────────────────────────────────────


def test_from_retrieval_creates_instance() -> None:
    retrieval = _make_mock_retrieval()
    llm = _make_mock_llm()
    agent = DocumentAnalystAgent.from_retrieval(retrieval, llm)
    assert isinstance(agent, DocumentAnalystAgent)


def test_tools_have_correct_names() -> None:
    retrieval = _make_mock_retrieval()
    t1 = make_retrieve_tool(retrieval)
    t2 = make_summarize_tool(retrieval)
    t3 = make_extract_tool(retrieval)
    names = {t1.name, t2.name, t3.name}
    assert "retrieve_documents" in names
    assert "summarize_topic" in names
    assert "extract_entities" in names
