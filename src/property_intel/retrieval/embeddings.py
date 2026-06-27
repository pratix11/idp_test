from __future__ import annotations

import openai

from property_intel.retrieval.models import DocumentChunk

EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small output dimension


class EmbeddingService:
    """Generate dense vector embeddings using OpenAI text-embedding-3-small.

    Vectors returned by OpenAI are already normalized, so cosine similarity
    == dot product — the cheapest similarity op in Qdrant.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = "",
    ) -> None:
        self._model_name = model_name
        self._client = openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised 1536-dim vector per input text."""
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, list[float]]]:
        """Embed each chunk's content and return (chunk, vector) pairs."""
        if not chunks:
            return []
        vectors = self.embed([c.content for c in chunks])
        return list(zip(chunks, vectors))
