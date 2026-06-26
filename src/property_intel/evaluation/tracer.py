"""Tracer — Phase 6 LangSmith and OpenAI observability wrappers.

LangSmithTracer:
  Wraps a callable with LangSmith's @traceable decorator so every invocation
  is recorded in the LangSmith dashboard (inputs, outputs, latency, tokens).
  Requires LANGCHAIN_API_KEY env var and LANGCHAIN_TRACING_V2=true.

OpenAITracer:
  Wraps calls with the openai SDK's trace context manager so spans appear in
  OpenAI's platform dashboard.  Requires OPENAI_API_KEY.

Both tracers are no-ops when the relevant env vars are absent — safe to use
in any environment without crashing.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class LangSmithTracer:
    """Wrap callables with LangSmith tracing.

    Usage:
        tracer = LangSmithTracer(project_name="property-intel-eval")
        traced_ask = tracer.trace(copilot_service.ask, name="copilot_ask")
        result = traced_ask("What is the registration fee?")
    """

    def __init__(
        self,
        project_name: str = "property-intel",
        *,
        enabled: bool | None = None,
    ) -> None:
        self._project_name = project_name
        if enabled is None:
            self._enabled = (
                bool(os.getenv("LANGCHAIN_API_KEY"))
                and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
            )
        else:
            self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace(self, fn: F, *, name: str | None = None) -> F:
        """Return *fn* wrapped with LangSmith tracing.

        If tracing is disabled (no API key), returns *fn* unchanged.
        """
        if not self._enabled:
            return fn

        try:
            from langsmith import traceable

            run_name = name or fn.__name__
            traced = traceable(name=run_name, project_name=self._project_name)(fn)
            return traced  # type: ignore[return-value]
        except Exception:
            return fn

    def trace_method(self, name: str | None = None) -> Callable[[F], F]:
        """Decorator factory for wrapping methods at definition time."""

        def decorator(fn: F) -> F:
            return self.trace(fn, name=name)

        return decorator


class OpenAITracer:
    """Wrap OpenAI API calls with the SDK's built-in trace context manager.

    Usage:
        tracer = OpenAITracer()
        with tracer.span("copilot_ask"):
            result = openai_client.chat.completions.create(...)
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        if enabled is None:
            self._enabled = bool(os.getenv("OPENAI_API_KEY"))
        else:
            self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def span(self, name: str) -> Any:
        """Context manager that records a named span in OpenAI's trace view.

        Falls back to a no-op context manager when tracing is disabled.
        """
        if not self._enabled:
            return _NoOpContext()

        try:
            import openai

            if hasattr(openai, "trace"):
                return openai.trace(name)  # type: ignore[attr-defined]
        except Exception:
            pass
        return _NoOpContext()

    def wrap(self, fn: F, *, span_name: str | None = None) -> F:
        """Return *fn* wrapped so each call is recorded as an OpenAI trace span."""
        if not self._enabled:
            return fn

        name = span_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.span(name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


class _NoOpContext:
    """No-op context manager used when tracing is disabled."""

    def __enter__(self) -> _NoOpContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass
