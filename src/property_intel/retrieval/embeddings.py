from __future__ import annotations

from sentence_transformers import SentenceTransformer

from property_intel.retrieval.models import DocumentChunk

EMBEDDING_DIM = 1024  # BGE-M3 output dimension


class EmbeddingService:
    """Generate dense vector embeddings using BGE-M3.

    The model is lazy-loaded on first use so that importing this module
    does not trigger a 570 MB download or GPU initialisation.

    All vectors are L2-normalised before returning, which means
    cosine similarity == dot product — the cheapest similarity op in Qdrant.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised 1024-dim vector per input text."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()  # type: ignore[union-attr]

    def embed_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, list[float]]]:
        """Embed each chunk's content and return (chunk, vector) pairs."""
        if not chunks:
            return []
        vectors = self.embed([c.content for c in chunks])
        return list(zip(chunks, vectors))
