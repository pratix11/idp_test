from pydantic import BaseModel


class DocumentChunk(BaseModel):
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    section_title: str | None = None
