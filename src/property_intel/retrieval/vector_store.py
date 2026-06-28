from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from property_intel.retrieval.models import ScoredChunk, VectorPoint

COLLECTION_NAME = "document_chunks"
VECTOR_DIM = 1536


class QdrantStore:
    """Thin wrapper around QdrantClient scoped to the document_chunks collection.

    Accepts an injectable client so unit tests can pass a mock without
    touching a real Qdrant instance.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        *,
        url: str = "",
        api_key: str = "",
        client: QdrantClient | None = None,
    ) -> None:
        if client:
            self._client = client
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key or None)
        else:
            self._client = QdrantClient(host=host, port=port)
        self._collection = COLLECTION_NAME

    # ── collection lifecycle ───────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=VECTOR_DIM,
                    distance=qm.Distance.COSINE,
                ),
            )

    def drop_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

    # ── write ──────────────────────────────────────────────────────────────

    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or overwrite vectors. Idempotent — safe to call multiple times."""
        if not points:
            return
        self._client.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )

    def delete_by_document_id(self, document_id: int) -> None:
        """Remove all vectors whose payload carries this document_id."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    # ── read ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        document_id: int | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-k most similar chunks to query_vector.

        Pass document_id to restrict results to one document (useful for
        "find similar passages within the same document" queries).
        """
        query_filter: qm.Filter | None = None
        if document_id is not None:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id",
                        match=qm.MatchValue(value=document_id),
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        results = []
        for p in response.points:
            if p.payload is None:
                continue
            results.append(
                ScoredChunk(
                    chunk_id=int(p.id),
                    document_id=p.payload["document_id"],
                    chunk_index=p.payload["chunk_index"],
                    content=p.payload["content"],
                    section_title=p.payload.get("section_title"),
                    score=p.score,
                )
            )
        return results

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
