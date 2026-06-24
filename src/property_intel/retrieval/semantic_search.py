from property_intel.retrieval.embeddings import EmbeddingService
from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.vector_store import QdrantStore


class SemanticSearch:
    """End-to-end semantic search: text → embedding → Qdrant ANN → ScoredChunks.

    Keeps EmbeddingService and QdrantStore as injected dependencies so
    unit tests can substitute fakes without touching a model or vector DB.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantStore,
    ) -> None:
        self._embedder = embedding_service
        self._store = vector_store

    def search(
        self,
        query: str,
        limit: int = 10,
        document_id: int | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-k chunks most semantically relevant to query.

        Args:
            query: Natural-language question or keyword string.
            limit: Maximum number of results to return.
            document_id: If set, restrict results to one document.
        """
        if not query.strip():
            return []
        vectors = self._embedder.embed([query])
        return self._store.search(vectors[0], limit=limit, document_id=document_id)
