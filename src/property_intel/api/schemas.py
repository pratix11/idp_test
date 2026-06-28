"""Request / response schemas for the FastAPI copilot endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")


class SummarizeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Topic or keywords to summarise.")


class CompareRequest(BaseModel):
    query_a: str = Field(..., min_length=1, description="First topic or document set.")
    query_b: str = Field(..., min_length=1, description="Second topic or document set.")


class CitationOut(BaseModel):
    index: int
    chunk_id: int
    document_id: int
    document_title: str | None = None
    section_title: str | None
    content_snippet: str


class CopilotResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class HealthResponse(BaseModel):
    status: str = "ok"


class SearchResultOut(BaseModel):
    document_id: int
    title: str
    snippet: str | None = None
    score: float | None = None
    category: str


class SearchResponse(BaseModel):
    results: list[SearchResultOut]


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, description="The task or question for the agent.")


class AgentResponse(BaseModel):
    answer: str
    agent: str = Field(..., description="Name of the agent that handled the task.")


class MeResponse(BaseModel):
    user_id: str
    role: str
    permissions: list[str]
