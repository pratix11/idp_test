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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from property_intel.api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    CitationOut,
    CompareRequest,
    CopilotResponse,
    HealthResponse,
    MeResponse,
    SearchResponse,
    SearchResultOut,
    SummarizeRequest,
)
from property_intel.enterprise.rbac import (
    BUILTIN_ROLES,
    AccessControl,
    PermissionDeniedError,
    User,
)
from property_intel.config import get_settings
from property_intel.copilot.service import CopilotService
from property_intel.db.session import get_engine, get_session, init_db
from property_intel.search.schema import SearchQuery
from property_intel.search.service import SearchService

app = FastAPI(
    title="Property Intelligence AI Copilot",
    description="RAG-powered Q&A over Indian property and regulatory documents.",
    version="4.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    import logging
    log = logging.getLogger("property_intel.startup")
    engine = get_engine()
    init_db(engine)
    _ensure_data_ready(engine)
    log.info("Startup complete")


def _ensure_data_ready(engine: object) -> None:
    """Restore documents and chunks if the DB is empty (e.g. after test teardown)."""
    import hashlib
    import logging
    from datetime import date
    from pathlib import Path
    from sqlalchemy import Engine, text

    from property_intel.db.models import ChunkModel, DocumentModel
    from property_intel.db.session import get_session_factory

    log = logging.getLogger("property_intel.startup")
    assert isinstance(engine, Engine)
    session = get_session_factory(engine)()
    try:
        doc_count = session.query(DocumentModel).count()
        if doc_count == 0:
            log.warning("documents table empty — restoring from data/processed/")
            processed = Path("data/processed")
            if not processed.exists():
                log.warning("data/processed not found, skipping restore")
                return
            valid_cats = {"acts", "circulars", "regulations", "rules", "orders", "reports"}
            inserted = 0
            for md in sorted(processed.rglob("*.md")):
                parts = md.relative_to(processed).parts
                cat = parts[0].lower() if parts else "acts"
                if cat not in valid_cats:
                    cat = "acts"
                content = md.read_text(encoding="utf-8", errors="replace")
                h = hashlib.sha256(content.encode()).hexdigest()
                if session.query(DocumentModel).filter_by(content_hash=h).first():
                    continue
                session.add(DocumentModel(
                    title=md.stem.replace("_", " ").replace("-", " "),
                    source="mahaRERA",
                    category=cat,
                    document_type=cat,
                    date=date(2020, 1, 1),
                    pages=1,
                    file_path=str(md.with_suffix(".pdf")).replace("processed", "raw"),
                    markdown_path=str(md),
                    content=content,
                    content_hash=h,
                    state="completed",
                ))
                inserted += 1
            session.commit()
            log.warning(f"Restored {inserted} documents from markdown")

        chunk_count = session.query(ChunkModel).count()
        if chunk_count == 0:
            log.warning("document_chunks empty — re-indexing into Qdrant")
            from property_intel.retrieval.service import RetrievalService
            svc = RetrievalService.from_settings(session)
            counts = svc.index_documents(reindex=True)
            log.warning(f"Re-indexed {counts['documents']} docs -> {counts['chunks']} chunks")
    finally:
        session.close()

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


def _get_agent_router(session: Session = Depends(get_session)) -> object:
    from langchain_openai import ChatOpenAI

    from property_intel.agents.comparison import ComparisonAgent
    from property_intel.agents.compliance import ComplianceAgent
    from property_intel.agents.document_analyst import DocumentAnalystAgent
    from property_intel.agents.report import ReportAgent
    from property_intel.agents.research import ResearchAgent
    from property_intel.agents.router import AgentRouter
    from property_intel.retrieval.service import RetrievalService

    settings = get_settings()
    retrieval = RetrievalService.from_settings(session)
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        temperature=0.0,
    )
    agents = {
        "document_analyst": DocumentAnalystAgent.from_retrieval(retrieval, llm),
        "comparison": ComparisonAgent.from_retrieval(retrieval, llm),
        "compliance": ComplianceAgent.from_retrieval(retrieval, llm),
        "research": ResearchAgent.from_retrieval(retrieval, llm),
        "report": ReportAgent.from_retrieval(retrieval, llm),
    }
    return AgentRouter(agents=agents)


# ── RBAC dependencies ─────────────────────────────────────────────────────────

_ac = AccessControl()


def get_current_user(x_user_role: str = Header(default="viewer")) -> User:
    """Read X-User-Role header and return a User with the matching built-in role.

    In production this would decode a JWT; here a plain header is enough for
    local testing.  Unknown role names return 401.
    """
    role = BUILTIN_ROLES.get(x_user_role.lower())
    if role is None:
        raise HTTPException(
            status_code=401,
            detail=f"Unknown role '{x_user_role}'. Valid: {list(BUILTIN_ROLES)}",
        )
    return User(user_id=f"api-{x_user_role}", roles=[role])


from collections.abc import Callable as _Callable


def require_permission(action: str, resource: str) -> _Callable[..., User]:
    """FastAPI dependency factory that enforces a single action:resource check."""
    def _check(user: User = Depends(get_current_user)) -> User:
        try:
            _ac.require(user, action, resource)
        except PermissionDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return user
    return _check


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


# ── identity ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)) -> MeResponse:
    """Return the current user's role and permissions (read from X-User-Role header)."""
    perms = sorted(str(p) for p in _ac.permissions_for(user))
    return MeResponse(user_id=user.user_id, role=user.role_names()[0], permissions=perms)


# ── search ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/search", response_model=SearchResponse)
def search(
    q: str = "",
    mode: str = "bm25",
    limit: int = 10,
    search_svc: SearchService = Depends(_get_search),
    _user: User = Depends(require_permission("read", "search")),
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
    _user: User = Depends(require_permission("execute", "copilot")),
) -> CopilotResponse:
    return _to_response(copilot.ask(body.question))


@app.post("/api/v1/summarize", response_model=CopilotResponse)
def summarize(
    body: SummarizeRequest,
    copilot: CopilotService = Depends(_get_copilot),
    _user: User = Depends(require_permission("execute", "copilot")),
) -> CopilotResponse:
    return _to_response(copilot.summarize(body.query))


@app.post("/api/v1/compare", response_model=CopilotResponse)
def compare(
    body: CompareRequest,
    copilot: CopilotService = Depends(_get_copilot),
    _user: User = Depends(require_permission("execute", "copilot")),
) -> CopilotResponse:
    return _to_response(copilot.compare(body.query_a, body.query_b))


# ── streaming endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/ask/stream")
def ask_stream(
    body: AskRequest,
    copilot: CopilotService = Depends(_get_copilot),
    _user: User = Depends(require_permission("execute", "copilot")),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_ask(body.question)),
        media_type="text/event-stream",
    )


@app.post("/api/v1/summarize/stream")
def summarize_stream(
    body: SummarizeRequest,
    copilot: CopilotService = Depends(_get_copilot),
    _user: User = Depends(require_permission("execute", "copilot")),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_summarize(body.query)),
        media_type="text/event-stream",
    )


@app.post("/api/v1/compare/stream")
def compare_stream(
    body: CompareRequest,
    copilot: CopilotService = Depends(_get_copilot),
    _user: User = Depends(require_permission("execute", "copilot")),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_response(copilot.stream_compare(body.query_a, body.query_b)),
        media_type="text/event-stream",
    )


# ── agent endpoint (Phase 5) ───────────────────────────────────────────────────

@app.post("/api/v1/agent", response_model=AgentResponse)
def run_agent(
    body: AgentRequest,
    router: object = Depends(_get_agent_router),
    _user: User = Depends(require_permission("execute", "agents")),
) -> AgentResponse:
    from property_intel.agents.router import AgentRouter

    r: AgentRouter = router  # type: ignore[assignment]
    agent_name = r._classify(body.task)
    answer = r.route(body.task)
    return AgentResponse(answer=answer, agent=agent_name)
