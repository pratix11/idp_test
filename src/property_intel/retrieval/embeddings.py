from __future__ import annotations

import openai

from property_intel.retrieval.models import DocumentChunk

EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small output dimension


class EmbeddingService:
    """Generate dense vector embeddings via the OpenAI Embeddings API.

    Uses text-embedding-3-small (1536 dims) by default.  Each call sends
    a batch of texts to the API and returns normalised float vectors.
    An injectable client allows unit tests to pass a mock without network calls.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        *,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._client = client or openai.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one 1536-dim vector per input text."""
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, list[float]]]:
        """Embed each chunk's content and return (chunk, vector) pairs."""
        if not chunks:
            return []
        vectors = self.embed([c.content for c in chunks])
        return list(zip(chunks, vectors))

    @classmethod
    def from_settings(cls, settings: object) -> EmbeddingService:
        from property_intel.config import Settings

        assert isinstance(settings, Settings)
        return cls(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
