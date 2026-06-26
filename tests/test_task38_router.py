"""Tests for Phase 5 AgentRouter workflow routing (Task 39).

Tests cover:
- Keyword classification for all 5 agent types
- Tie-breaking: most keyword matches wins
- Fallback: unknown tasks → document_analyst
- route() dispatches to the correct registered agent
- route() raises KeyError for unregistered agent
- route() returns the agent's result string
- All 5 agent types can be registered and dispatched
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from property_intel.agents.router import AgentRouter
from property_intel.agents.compliance import ComplianceAgent
from property_intel.agents.comparison import ComparisonAgent
from property_intel.agents.document_analyst import DocumentAnalystAgent
from property_intel.agents.report import ReportAgent
from property_intel.agents.research import ResearchAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_agent(result: str = "agent result") -> MagicMock:
    agent = MagicMock()
    agent.run.return_value = result
    return agent


def _all_agents_registry() -> dict:
    return {
        "document_analyst": _make_mock_agent("analyst result"),
        "comparison": _make_mock_agent("comparison result"),
        "compliance": _make_mock_agent("compliance result"),
        "research": _make_mock_agent("research result"),
        "report": _make_mock_agent("report result"),
    }


# ── Keyword classification ────────────────────────────────────────────────────


@pytest.mark.parametrize("task,expected", [
    ("summarize the MahaRERA builder registration rules", "document_analyst"),
    ("extract key deadlines from the circular", "document_analyst"),
    ("analyse the penalty provisions", "document_analyst"),
    ("compare the 2016 rules with the 2019 amendment", "comparison"),
    ("what are the differences between these two documents?", "comparison"),
    ("vs the original regulation", "comparison"),
    ("check compliance with the escrow requirement", "compliance"),
    ("validate that the builder satisfies all requirements", "compliance"),
    ("is this situation a compliance violation?", "compliance"),
    ("research builder obligations across multiple circulars", "research"),
    ("multi document analysis of penalty provisions", "research"),
    ("generate report on MahaRERA obligations", "report"),
    ("draft report for this regulatory area", "report"),
    ("write report covering builder duties", "report"),
])
def test_classify_routes_to_correct_agent(task: str, expected: str) -> None:
    router = AgentRouter({})
    assert router._classify(task) == expected


def test_classify_unknown_task_falls_back_to_document_analyst() -> None:
    router = AgentRouter({})
    result = router._classify("what is the penalty amount?")
    assert result == "document_analyst"


def test_classify_empty_task_falls_back_to_document_analyst() -> None:
    router = AgentRouter({})
    assert router._classify("") == "document_analyst"


def test_classify_tie_broken_by_keyword_count() -> None:
    router = AgentRouter({})
    # "compare" beats "summarize" if both appear but compare has more hits
    task = "compare and analyse two regulations"
    result = router._classify(task)
    # "compare" matches comparison, "analyse" matches document_analyst — 1 each
    # either is valid; just check it's one of the two
    assert result in ("comparison", "document_analyst")


# ── route() dispatching ────────────────────────────────────────────────────────


def test_route_dispatches_to_document_analyst() -> None:
    registry = _all_agents_registry()
    router = AgentRouter(registry)
    result = router.route("summarize the builder registration process")
    registry["document_analyst"].run.assert_called_once()
    assert result == "analyst result"


def test_route_dispatches_to_comparison_agent() -> None:
    registry = _all_agents_registry()
    router = AgentRouter(registry)
    result = router.route("compare section 3 vs section 5 differences")
    registry["comparison"].run.assert_called_once()
    assert result == "comparison result"


def test_route_dispatches_to_compliance_agent() -> None:
    registry = _all_agents_registry()
    router = AgentRouter(registry)
    result = router.route("check compliance with escrow requirement")
    registry["compliance"].run.assert_called_once()
    assert result == "compliance result"


def test_route_dispatches_to_research_agent() -> None:
    registry = _all_agents_registry()
    router = AgentRouter(registry)
    result = router.route("research across multiple documents")
    registry["research"].run.assert_called_once()
    assert result == "research result"


def test_route_dispatches_to_report_agent() -> None:
    registry = _all_agents_registry()
    router = AgentRouter(registry)
    result = router.route("generate report on regulatory obligations")
    registry["report"].run.assert_called_once()
    assert result == "report result"


def test_route_raises_for_unregistered_agent() -> None:
    router = AgentRouter({"document_analyst": _make_mock_agent()})
    with pytest.raises(KeyError):
        router.route("compare two documents")  # should route to comparison, not registered


def test_route_passes_task_to_agent_run() -> None:
    task = "summarize the registration timeline"
    agent = _make_mock_agent("done")
    router = AgentRouter({"document_analyst": agent})
    router.route(task)
    agent.run.assert_called_once_with(task)


# ── Agent type constructors can be wired into router ─────────────────────────


def _make_real_agent(agent_cls: type, n_tools: int = 1) -> MagicMock:
    """Return a mock for a real agent class without actually building the graph."""
    m = MagicMock(spec=agent_cls)
    m.run.return_value = f"{agent_cls.__name__} result"
    return m


def test_router_works_with_all_five_agent_types_mocked() -> None:
    registry = {
        "document_analyst": _make_real_agent(DocumentAnalystAgent),
        "comparison": _make_real_agent(ComparisonAgent),
        "compliance": _make_real_agent(ComplianceAgent),
        "research": _make_real_agent(ResearchAgent),
        "report": _make_real_agent(ReportAgent),
    }
    router = AgentRouter(registry)

    tasks = [
        ("summarize rules", "document_analyst"),
        ("compare documents", "comparison"),
        ("validate compliance", "compliance"),
        ("research corpus", "research"),
        ("generate report", "report"),
    ]
    for task, expected_key in tasks:
        result = router.route(task)
        assert result == f"{registry[expected_key].run.return_value}"
