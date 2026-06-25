"""FastAPI application for the Phase 4 AI Copilot.

Endpoints:
  GET  /health                  — liveness check
  POST /api/v1/ask              — question answering (non-streaming)
  POST /api/v1/summarize        — document summarisation (non-streaming)
  POST /api/v1/compare          — document comparison (non-streaming)
  POST /api/v1/ask/stream       — question answering (SSE streaming)
  POST /api/v1/summarize/stream — summarisation (SSE streaming)
  POST /api/v1/compare/stream   — comparison (SSE streaming)

Run with:
    uvicorn property_intel.api.app:app --reload
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from property_intel.api.schemas import (
    AskRequest,
    CitationOut,
    CompareRequest,
    CopilotResponse,
    HealthResponse,
    SummarizeRequest,
)
from property_intel.copilot.service import CopilotService
from property_intel.db.session import get_session

app = FastAPI(
    title="Property Intelligence AI Copilot",
    description="RAG-powered Q&A over Indian property and regulatory documents.",
    version="4.0.0",
)


def _get_copilot(session: Session = Depends(get_session)) -> CopilotService:
    return CopilotService.from_settings(session)


def _to_response(result: object) -> CopilotResponse:
    from property_intel.copilot.service import CopilotAnswer
    assert isinstance(result, CopilotAnswer)
    return CopilotResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                index=c.index,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                section_title=c.section_title,
                content_snippet=c.content_snippet,
            )
            for c in result.citations
        ],
    )


async def _stream_response(chunks: object) -> AsyncGenerator[str, None]:
    from collections.abc import Iterator
    assert isinstance(chunks, Iterator)
    for chunk in chunks:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"


# ── health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


# ── non-streaming endpoints ────────────────────────────────────────────────────

@app.post("/api/v1/ask", response_model=CopilotResponse)
def ask(
    body: AskRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> CopilotResponse:
    return _to_response(copilot.ask(body.question))


@app.post("/api/v1/summarize", response_model=CopilotResponse)
def summarize(
    body: SummarizeRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> CopilotResponse:
    return _to_response(copilot.summarize(body.query))


@app.post("/api/v1/compare", response_model=CopilotResponse)
def compare(
    body: CompareRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> CopilotResponse:
    return _to_response(copilot.compare(body.query_a, body.query_b))


# ── streaming endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/ask/stream")
def ask_stream(
    body: AskRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_ask(body.question)),
        media_type="text/event-stream",
    )


@app.post("/api/v1/summarize/stream")
def summarize_stream(
    body: SummarizeRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_summarize(body.query)),
        media_type="text/event-stream",
    )


@app.post("/api/v1/compare/stream")
def compare_stream(
    body: CompareRequest,
    copilot: CopilotService = Depends(_get_copilot),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_compare(body.query_a, body.query_b)),
        media_type="text/event-stream",
    )
