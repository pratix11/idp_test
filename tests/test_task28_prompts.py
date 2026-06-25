"""Tests for Task 28: Prompt templates."""

from __future__ import annotations

from property_intel.copilot.context_builder import BuiltContext
from property_intel.copilot.prompts import (
    build_compare_prompt,
    build_qa_prompt,
    build_summarize_prompt,
)


def _ctx(context: str = "some context") -> BuiltContext:
    return BuiltContext(context=context, citations=[], token_count=10)


# ── qa prompt ─────────────────────────────────────────────────────────────────

def test_qa_prompt_has_system_and_user() -> None:
    messages = build_qa_prompt("What is Section 4?", _ctx())
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


def test_qa_prompt_user_contains_question() -> None:
    messages = build_qa_prompt("What is Section 4?", _ctx())
    assert "What is Section 4?" in str(messages[-1]["content"])


def test_qa_prompt_user_contains_context() -> None:
    messages = build_qa_prompt("q?", _ctx(context="[1] Section: A\nsome text"))
    assert "[1] Section: A" in str(messages[-1]["content"])


def test_qa_prompt_system_contains_citation_instruction() -> None:
    messages = build_qa_prompt("q?", _ctx())
    system = str(messages[0]["content"])
    assert "[N]" in system


def test_qa_prompt_system_instructs_to_use_only_provided_documents() -> None:
    messages = build_qa_prompt("q?", _ctx())
    system = str(messages[0]["content"])
    assert "ONLY" in system


# ── summarize prompt ──────────────────────────────────────────────────────────

def test_summarize_prompt_has_system_and_user() -> None:
    messages = build_summarize_prompt(_ctx())
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


def test_summarize_prompt_user_contains_context() -> None:
    messages = build_summarize_prompt(_ctx(context="[1] the text"))
    assert "[1] the text" in str(messages[-1]["content"])


def test_summarize_prompt_system_mentions_summary() -> None:
    messages = build_summarize_prompt(_ctx())
    assert "summary" in str(messages[0]["content"]).lower()


# ── compare prompt ────────────────────────────────────────────────────────────

def test_compare_prompt_has_system_and_user() -> None:
    messages = build_compare_prompt(_ctx("set a"), _ctx("set b"))
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


def test_compare_prompt_includes_both_contexts() -> None:
    messages = build_compare_prompt(_ctx("context alpha"), _ctx("context beta"))
    user = str(messages[-1]["content"])
    assert "context alpha" in user
    assert "context beta" in user


def test_compare_prompt_labels_sets() -> None:
    messages = build_compare_prompt(_ctx("a"), _ctx("b"))
    user = str(messages[-1]["content"])
    assert "Set A" in user
    assert "Set B" in user
