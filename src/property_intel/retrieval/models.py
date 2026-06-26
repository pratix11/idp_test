from pydantic import BaseModel


class DocumentChunk(BaseModel):
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    section_title: str | None = None


class VectorPoint(BaseModel):
    """A chunk + its embedding, ready to be written to Qdrant."""

    id: int
    vector: list[float]
    payload: dict[str, object]


class ScoredChunk(BaseModel):
    """One result returned by a semantic search query."""

    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    section_title: str | None
    score: float
    document_title: str | None = None
