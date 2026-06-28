"""FastAPI application for the Phase 4 AI Copilot.

Endpoints:
  GET  /health                  — liveness check
  POST /api/v1/ask              — question answering (non-streaming)
  POST /api/v1/summarize        — document summarisation (non-streaming)
  POST /api/v1/compare          — document comparison (non-streaming)
  GET  /api/v1/search           — BM25 / full-text document search
  POST /api/v1/ask/stream       — question answering (SSE streaming)
  POST /api/v1/summarize/stream — summarisation (SSE streaming)
  POST /api/v1/compare/stream   — comparison (SSE streaming)

Run with:
    uvicorn property_intel.api.app:app --reload
"""

from __future__ import annotations

import json
import traceback

# Load .env into os.environ so third-party SDKs (LangSmith, etc.) can read it
from dotenv import load_dotenv
load_dotenv()
from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from property_intel.api.schemas import (
    AskRequest,
    CitationOut,
    CompareRequest,
    CopilotResponse,
    HealthResponse,
    SearchResponse,
    SearchResultOut,
    SummarizeRequest,
)
from property_intel.config import get_settings
from property_intel.copilot.service import CopilotService
from property_intel.db.session import get_session
from property_intel.retrieval.vector_store import QdrantStore
from property_intel.search.schema import SearchQuery
from property_intel.search.service import SearchService

app = FastAPI(
    title="Property Intelligence AI Copilot",
    description="RAG-powered Q&A over Indian property and regulatory documents.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": traceback.format_exc()},
    )


# ── singleton services ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _shared_qdrant() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port)


# ── FastAPI dependencies ───────────────────────────────────────────────────────

def _get_copilot(session: Session = Depends(get_session)) -> CopilotService:
    from property_intel.copilot.context_builder import ContextBuilder
    from property_intel.copilot.llm_client import LLMClient
    from property_intel.retrieval.service import RetrievalService

    retrieval = RetrievalService.from_settings(session)
    return CopilotService(
        retrieval=retrieval,
        llm=LLMClient.from_settings(),
        context_builder=ContextBuilder(),
    )


def _get_search(session: Session = Depends(get_session)) -> SearchService:
    return SearchService(session)


# ── helpers ────────────────────────────────────────────────────────────────────

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
                document_title=c.document_title,
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


# ── search ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/search", response_model=SearchResponse)
def search(
    q: str = "",
    mode: str = "bm25",
    limit: int = 10,
    search_svc: SearchService = Depends(_get_search),
) -> SearchResponse:
    if not q.strip():
        return SearchResponse(results=[])
    query = SearchQuery(text=q, page_size=limit)
    mode_arg = mode if mode in ("fulltext", "bm25", "metadata") else "bm25"
    page = search_svc.search(query, mode=mode_arg)  # type: ignore[arg-type]
    return SearchResponse(
        results=[
            SearchResultOut(
                document_id=item.document_id,
                title=item.title,
                snippet=item.snippet,
                score=item.score,
                category=item.category,
            )
            for item in page.items
        ]
    )


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
