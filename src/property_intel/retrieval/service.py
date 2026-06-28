"""RetrievalService — unified facade for Phase 3 retrieval.

Wires together chunking, embedding, Qdrant, semantic search, hybrid search,
and reranking into a single entry point.  Also exposes an indexing method
that chunks and embeds all completed documents so they are ready to query.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from property_intel.config import get_settings
from property_intel.db.repository import ChunkRepository, DocumentRepository
from property_intel.retrieval.chunking import MarkdownChunker
from property_intel.retrieval.embeddings import EmbeddingService
from property_intel.retrieval.hybrid_search import ChunkBM25Search, HybridSearch
from property_intel.retrieval.models import ScoredChunk, VectorPoint
from property_intel.retrieval.reranker import Reranker
from property_intel.retrieval.semantic_search import SemanticSearch
from property_intel.retrieval.vector_store import QdrantStore

RetrievalMode = str  # "semantic" | "hybrid"


class RetrievalService:
    """Single entry point for all Phase 3 retrieval operations.

    Usage:
        svc = RetrievalService.from_settings(session)
        svc.index_documents()           # chunk + embed all completed docs
        results = svc.search("query")   # hybrid search + optional rerank
    """

    def __init__(
        self,
        session: Session,
        chunker: MarkdownChunker,
        embedder: EmbeddingService,
        vector_store: QdrantStore,
        reranker: Reranker,
    ) -> None:
        self._session = session
        self._chunker = chunker
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker

    @classmethod
    def from_settings(cls, session: Session) -> RetrievalService:
        settings = get_settings()
        return cls(
            session=session,
            chunker=MarkdownChunker(),
            embedder=EmbeddingService(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            ),
            vector_store=QdrantStore(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            ),
            reranker=Reranker(
                api_key=settings.cohere_api_key,
                model=settings.cohere_rerank_model,
            ),
        )

    # ── indexing ───────────────────────────────────────────────────────────

    def index_documents(self, *, reindex: bool = False) -> dict[str, int]:
        """Chunk, embed, and index all completed documents into Qdrant.

        Args:
            reindex: If True, delete existing chunks/vectors and rebuild from scratch.

        Returns:
            {"documents": N, "chunks": M} counts of what was processed.
        """
        self._store.ensure_collection()
        doc_repo = DocumentRepository(self._session)
        chunk_repo = ChunkRepository(self._session)
        docs = doc_repo.list_searchable()

        total_chunks = 0
        for doc in docs:
            if reindex:
                chunk_repo.delete_by_document_id(doc.id)
                self._store.delete_by_document_id(doc.id)

            text = doc.content or ""
            if not text.strip():
                continue

            pydantic_chunks = self._chunker.chunk_document(text, document_id=doc.id)
            if not pydantic_chunks:
                continue

            db_chunks = chunk_repo.bulk_create(pydantic_chunks)
            vectors = self._embedder.embed([c.content for c in db_chunks])
            points = [
                VectorPoint(
                    id=db_chunk.id,
                    vector=vector,
                    payload={
                        "document_id": db_chunk.document_id,
                        "chunk_index": db_chunk.chunk_index,
                        "content": db_chunk.content,
                        "section_title": db_chunk.section_title,
                    },
                )
                for db_chunk, vector in zip(db_chunks, vectors)
            ]
            self._store.upsert(points)
            total_chunks += len(points)

        return {"documents": len(docs), "chunks": total_chunks}

    # ── search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        rerank_top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Search indexed chunks and optionally rerank results.

        Args:
            query:        Natural-language query string.
            limit:        Maximum results to return (after optional reranking).
            mode:         "semantic" (Qdrant only) or "hybrid" (BM25 + Qdrant).
            rerank:       If True, run cross-encoder reranking on retrieved results.
            rerank_top_k: How many results to keep after reranking (defaults to limit).
        """
        if not query.strip():
            return []

        chunk_repo = ChunkRepository(self._session)
        semantic = SemanticSearch(self._embedder, self._store)

        if mode == "semantic":
            candidates = semantic.search(query, limit=limit if not rerank else limit * 3)
        else:
            bm25 = ChunkBM25Search(chunk_repo)
            hybrid = HybridSearch(bm25, semantic)
            candidates = hybrid.search(query, limit=limit if not rerank else limit * 3)

        if rerank and candidates:
            candidates = self._reranker.rerank(
                query, candidates, top_k=rerank_top_k or limit
            )

        results = candidates[:limit]
        doc_ids = list({c.document_id for c in results})
        titles = DocumentRepository(self._session).get_titles_by_ids(doc_ids)
        return [c.model_copy(update={"document_title": titles.get(c.document_id)}) for c in results]
