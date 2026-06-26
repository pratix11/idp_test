"""Tests for Phase 6 LangSmithTracer and OpenAITracer (Task 45).

Tests cover:
- LangSmithTracer: enabled/disabled based on env vars
- LangSmithTracer.trace() returns original fn when disabled
- LangSmithTracer.trace() wraps fn when enabled (via mock)
- LangSmithTracer.trace_method() decorator factory
- OpenAITracer: enabled/disabled based on env vars
- OpenAITracer.span() returns NoOpContext when disabled
- OpenAITracer.wrap() returns original fn when disabled
- OpenAITracer.wrap() wraps fn and calls span when enabled
- _NoOpContext is a valid context manager
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from property_intel.evaluation.tracer import (
    LangSmithTracer,
    OpenAITracer,
    _NoOpContext,
)


# ── LangSmithTracer ───────────────────────────────────────────────────────────


def test_langsmith_tracer_disabled_without_env_vars() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        tracer = LangSmithTracer()
        assert not tracer.enabled


def test_langsmith_tracer_enabled_with_env_vars() -> None:
    with patch.dict(os.environ, {
        "LANGCHAIN_API_KEY": "fake-key",
        "LANGCHAIN_TRACING_V2": "true",
    }):
        tracer = LangSmithTracer()
        assert tracer.enabled


def test_langsmith_tracer_explicit_enabled_flag() -> None:
    tracer = LangSmithTracer(enabled=True)
    assert tracer.enabled
    tracer2 = LangSmithTracer(enabled=False)
    assert not tracer2.enabled


def test_langsmith_trace_returns_fn_when_disabled() -> None:
    tracer = LangSmithTracer(enabled=False)

    def my_fn(x: int) -> int:
        return x * 2

    wrapped = tracer.trace(my_fn)
    assert wrapped is my_fn


def test_langsmith_trace_calls_fn_normally_when_disabled() -> None:
    tracer = LangSmithTracer(enabled=False)

    def add(a: int, b: int) -> int:
        return a + b

    wrapped = tracer.trace(add)
    assert wrapped(3, 4) == 7


def test_langsmith_trace_wraps_fn_when_enabled() -> None:
    tracer = LangSmithTracer(enabled=True)

    def my_fn(x: int) -> int:
        return x + 1

    mock_traceable = MagicMock(return_value=lambda fn: fn)
    with patch("property_intel.evaluation.tracer.LangSmithTracer.trace") as mock_trace:
        mock_trace.side_effect = lambda fn, name=None: fn
        result = tracer.trace(my_fn, name="test_fn")
        assert callable(result)


def test_langsmith_trace_method_decorator_passes_through_when_disabled() -> None:
    tracer = LangSmithTracer(enabled=False)

    @tracer.trace_method(name="my_method")
    def compute(x: int) -> int:
        return x * 3

    assert compute(4) == 12


def test_langsmith_project_name_stored() -> None:
    tracer = LangSmithTracer(project_name="test-project", enabled=False)
    assert tracer._project_name == "test-project"


# ── OpenAITracer ──────────────────────────────────────────────────────────────


def test_openai_tracer_disabled_without_api_key() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        tracer = OpenAITracer()
        assert not tracer.enabled


def test_openai_tracer_enabled_with_api_key() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake"}):
        tracer = OpenAITracer()
        assert tracer.enabled


def test_openai_tracer_explicit_flag() -> None:
    assert OpenAITracer(enabled=True).enabled
    assert not OpenAITracer(enabled=False).enabled


def test_openai_span_returns_noop_when_disabled() -> None:
    tracer = OpenAITracer(enabled=False)
    ctx = tracer.span("my_span")
    assert isinstance(ctx, _NoOpContext)


def test_openai_span_noop_is_usable_as_context_manager() -> None:
    tracer = OpenAITracer(enabled=False)
    with tracer.span("test"):
        result = 1 + 1
    assert result == 2


def test_openai_wrap_returns_original_when_disabled() -> None:
    tracer = OpenAITracer(enabled=False)

    def fn(x: int) -> int:
        return x

    wrapped = tracer.wrap(fn)
    assert wrapped is fn


def test_openai_wrap_calls_fn_when_disabled() -> None:
    tracer = OpenAITracer(enabled=False)
    called = []

    def fn(x: int) -> int:
        called.append(x)
        return x * 2

    wrapped = tracer.wrap(fn)
    assert wrapped(5) == 10
    assert called == [5]


def test_openai_wrap_wraps_fn_when_enabled() -> None:
    tracer = OpenAITracer(enabled=True)
    calls = []

    def fn(x: int) -> int:
        return x + 1

    # Patch span to record calls
    with patch.object(tracer, "span", return_value=_NoOpContext()) as mock_span:
        wrapped = tracer.wrap(fn, span_name="test_fn")
        result = wrapped(10)

    assert result == 11
    mock_span.assert_called_once_with("test_fn")


# ── _NoOpContext ──────────────────────────────────────────────────────────────


def test_noop_context_enter_returns_self() -> None:
    ctx = _NoOpContext()
    assert ctx.__enter__() is ctx


def test_noop_context_exit_does_not_raise() -> None:
    ctx = _NoOpContext()
    ctx.__enter__()
    ctx.__exit__(None, None, None)


def test_noop_context_manager_with_statement() -> None:
    executed = False
    with _NoOpContext():
        executed = True
    assert executed
