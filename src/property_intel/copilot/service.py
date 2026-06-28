"""CopilotService — main facade for Phase 4 AI Copilot.

Wires together RetrievalService (Phase 3), ContextBuilder, prompt templates,
and LLMClient into three user-facing operations: ask, summarize, compare.

All three operations have a streaming variant that yields text chunks
for progressive delivery to the UI.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langsmith import traceable
from sqlalchemy.orm import Session

from property_intel.copilot.context_builder import Citation, ContextBuilder
from property_intel.copilot.llm_client import LLMClient
from property_intel.copilot.prompts import (
    build_compare_prompt,
    build_qa_prompt,
    build_summarize_prompt,
)
from property_intel.retrieval.service import RetrievalService


@dataclass(frozen=True)
class CopilotAnswer:
    answer: str
    citations: list[Citation]


class CopilotService:
    """Single entry point for all Phase 4 AI Copilot operations.

    Usage (non-streaming):
        svc = CopilotService.from_settings(session)
        result = svc.ask("What are the builder registration rules?")
        print(result.answer)
        for c in result.citations:
            print(c.index, c.section_title)

    Usage (streaming):
        for chunk in svc.stream_ask("..."):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        context_builder: ContextBuilder,
        retrieval_limit: int = 10,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._ctx_builder = context_builder
        self._retrieval_limit = retrieval_limit

    @classmethod
    def from_settings(cls, session: Session) -> CopilotService:
        return cls(
            retrieval=RetrievalService.from_settings(session),
            llm=LLMClient.from_settings(),
            context_builder=ContextBuilder(),
        )

    # ── ask ────────────────────────────────────────────────────────────────────

    @traceable(run_type="chain", name="copilot.ask")
    def ask(self, question: str) -> CopilotAnswer:
        """Answer a question using retrieved context."""
        if not question.strip():
            return CopilotAnswer(answer="", citations=[])

        chunks = self._retrieval.search(question, limit=self._retrieval_limit)
        ctx = self._ctx_builder.build(chunks)
        messages = build_qa_prompt(question, ctx)
        answer = self._llm.complete(messages)
        return CopilotAnswer(answer=answer, citations=ctx.citations)

    def stream_ask(self, question: str) -> Iterator[str]:
        """Stream answer chunks for a question."""
        if not question.strip():
            return

        chunks = self._retrieval.search(question, limit=self._retrieval_limit)
        ctx = self._ctx_builder.build(chunks)
        messages = build_qa_prompt(question, ctx)
        yield from self._llm.stream_complete(messages)

    # ── summarize ──────────────────────────────────────────────────────────────

    @traceable(run_type="chain", name="copilot.summarize")
    def summarize(self, query: str) -> CopilotAnswer:
        """Summarise documents relevant to the query."""
        if not query.strip():
            return CopilotAnswer(answer="", citations=[])

        chunks = self._retrieval.search(query, limit=self._retrieval_limit)
        ctx = self._ctx_builder.build(chunks)
        messages = build_summarize_prompt(ctx)
        answer = self._llm.complete(messages)
        return CopilotAnswer(answer=answer, citations=ctx.citations)

    def stream_summarize(self, query: str) -> Iterator[str]:
        """Stream summarization chunks."""
        if not query.strip():
            return

        chunks = self._retrieval.search(query, limit=self._retrieval_limit)
        ctx = self._ctx_builder.build(chunks)
        messages = build_summarize_prompt(ctx)
        yield from self._llm.stream_complete(messages)

    # ── compare ────────────────────────────────────────────────────────────────

    @traceable(run_type="chain", name="copilot.compare")
    def compare(self, query_a: str, query_b: str) -> CopilotAnswer:
        """Compare documents matching query_a against those matching query_b."""
        if not query_a.strip() or not query_b.strip():
            return CopilotAnswer(answer="", citations=[])

        chunks_a = self._retrieval.search(query_a, limit=self._retrieval_limit // 2 or 5)
        chunks_b = self._retrieval.search(query_b, limit=self._retrieval_limit // 2 or 5)
        ctx_a = self._ctx_builder.build(chunks_a)
        ctx_b = self._ctx_builder.build(chunks_b)
        messages = build_compare_prompt(ctx_a, ctx_b)
        answer = self._llm.complete(messages)
        return CopilotAnswer(answer=answer, citations=ctx_a.citations + ctx_b.citations)

    def stream_compare(self, query_a: str, query_b: str) -> Iterator[str]:
        """Stream comparison chunks."""
        if not query_a.strip() or not query_b.strip():
            return

        chunks_a = self._retrieval.search(query_a, limit=self._retrieval_limit // 2 or 5)
        chunks_b = self._retrieval.search(query_b, limit=self._retrieval_limit // 2 or 5)
        ctx_a = self._ctx_builder.build(chunks_a)
        ctx_b = self._ctx_builder.build(chunks_b)
        messages = build_compare_prompt(ctx_a, ctx_b)
        yield from self._llm.stream_complete(messages)
