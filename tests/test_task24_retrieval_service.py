"""Task 24 — RetrievalService facade.

All dependencies are injected as mocks so no real model, DB, or Qdrant needed.
"""

from unittest.mock import MagicMock, call

import pytest

from property_intel.retrieval.chunking import MarkdownChunker
from property_intel.retrieval.embeddings import EmbeddingService
from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.reranker import Reranker
from property_intel.retrieval.service import RetrievalService
from property_intel.retrieval.vector_store import QdrantStore

# ── helpers ───────────────────────────────────────────────────────────────────


def _scored(chunk_id: int, score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, document_id=1, chunk_index=chunk_id,
        content=f"Content {chunk_id}", section_title=None, score=score,
    )


def _make_service(
    *,
    semantic_results: list[ScoredChunk] | None = None,
    reranker_results: list[ScoredChunk] | None = None,
) -> tuple[RetrievalService, dict]:
    """Build a RetrievalService with all dependencies mocked."""
    from property_intel.retrieval.models import DocumentChunk

    session = MagicMock()

    # Document repo — returns one searchable doc
    doc = MagicMock()
    doc.id = 1
    doc.content = "# Section\n\nSome regulatory text about refunds."
    mock_doc_repo = MagicMock()
    mock_doc_repo.list_searchable.return_value = [doc]

    # Chunk repo
    mock_chunk = MagicMock()
    mock_chunk.id = 10
    mock_chunk.document_id = 1
    mock_chunk.chunk_index = 0
    mock_chunk.content = "Some regulatory text about refunds."
    mock_chunk.section_title = "Section"
    mock_chunk_repo = MagicMock()
    mock_chunk_repo.bulk_create.return_value = [mock_chunk]
    mock_chunk_repo.list_all.return_value = [mock_chunk]

    chunker = MagicMock(spec=MarkdownChunker)
    chunker.chunk_document.return_value = [
        DocumentChunk(document_id=1, chunk_index=0, content="text", token_count=1)
    ]

    embedder = MagicMock(spec=EmbeddingService)
    embedder.embed.return_value = [[0.1] * 1024]

    vector_store = MagicMock(spec=QdrantStore)
    vector_store.search.return_value = semantic_results or [_scored(1)]

    reranker = MagicMock(spec=Reranker)
    reranker.rerank.return_value = reranker_results or [_scored(1, 0.95)]

    svc = RetrievalService(
        session=session,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
    )

    mocks = {
        "doc_repo": mock_doc_repo,
        "chunk_repo": mock_chunk_repo,
        "chunker": chunker,
        "embedder": embedder,
        "store": vector_store,
        "reranker": reranker,
    }
    return svc, mocks


# ── search tests ──────────────────────────────────────────────────────────────


def test_search_empty_query_returns_empty() -> None:
    svc, _ = _make_service()
    assert svc.search("") == []
    assert svc.search("   ") == []


def test_search_semantic_mode_uses_vector_store() -> None:
    chunks = [_scored(1, 0.9)]
    svc, mocks = _make_service(semantic_results=chunks, reranker_results=chunks)
    svc.search("refund policy", mode="semantic", rerank=False)
    mocks["store"].search.assert_called_once()


def test_search_rerank_true_calls_reranker() -> None:
    svc, mocks = _make_service()
    svc.search("refund policy", rerank=True)
    mocks["reranker"].rerank.assert_called_once()


def test_search_rerank_false_skips_reranker() -> None:
    svc, mocks = _make_service()
    svc.search("refund policy", rerank=False)
    mocks["reranker"].rerank.assert_not_called()


def test_search_limit_applied_to_final_results() -> None:
    results = [_scored(i, 0.9 - i * 0.05) for i in range(10)]
    svc, mocks = _make_service(semantic_results=results, reranker_results=results)
    mocks["store"].search.return_value = results
    mocks["reranker"].rerank.return_value = results
    final = svc.search("query", limit=3, rerank=True)
    assert len(final) <= 3


def test_search_returns_scored_chunks() -> None:
    svc, _ = _make_service()
    results = svc.search("regulatory", rerank=False)
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_search_hybrid_mode_is_default() -> None:
    """Hybrid mode invokes both BM25 and vector store paths."""
    svc, mocks = _make_service()
    svc.search("refund", rerank=False)
    # hybrid mode fetches chunks from DB (for BM25) AND queries Qdrant
    mocks["store"].search.assert_called()


# ── index_documents tests ─────────────────────────────────────────────────────


def test_index_ensures_collection() -> None:
    from unittest.mock import patch

    svc, mocks = _make_service()
    with (
        patch("property_intel.retrieval.service.DocumentRepository", return_value=mocks["doc_repo"]),
        patch("property_intel.retrieval.service.ChunkRepository", return_value=mocks["chunk_repo"]),
    ):
        svc.index_documents()
    mocks["store"].ensure_collection.assert_called_once()


def test_index_returns_counts() -> None:
    from unittest.mock import patch

    svc, mocks = _make_service()
    with (
        patch("property_intel.retrieval.service.DocumentRepository", return_value=mocks["doc_repo"]),
        patch("property_intel.retrieval.service.ChunkRepository", return_value=mocks["chunk_repo"]),
    ):
        counts = svc.index_documents()
    assert "documents" in counts
    assert "chunks" in counts
    assert counts["documents"] >= 0
    assert counts["chunks"] >= 0


def test_index_skips_documents_with_no_content() -> None:
    from unittest.mock import patch

    svc, mocks = _make_service()
    doc_no_content = MagicMock()
    doc_no_content.id = 2
    doc_no_content.content = ""
    mocks["doc_repo"].list_searchable.return_value = [doc_no_content]

    with (
        patch("property_intel.retrieval.service.DocumentRepository", return_value=mocks["doc_repo"]),
        patch("property_intel.retrieval.service.ChunkRepository", return_value=mocks["chunk_repo"]),
    ):
        counts = svc.index_documents()
    assert counts["chunks"] == 0


def test_index_upserts_vectors_to_store() -> None:
    from unittest.mock import patch

    svc, mocks = _make_service()
    with (
        patch("property_intel.retrieval.service.DocumentRepository", return_value=mocks["doc_repo"]),
        patch("property_intel.retrieval.service.ChunkRepository", return_value=mocks["chunk_repo"]),
    ):
        svc.index_documents()
    mocks["store"].upsert.assert_called()


def test_index_reindex_deletes_existing_chunks() -> None:
    from unittest.mock import patch

    svc, mocks = _make_service()
    with (
        patch("property_intel.retrieval.service.DocumentRepository", return_value=mocks["doc_repo"]),
        patch("property_intel.retrieval.service.ChunkRepository", return_value=mocks["chunk_repo"]),
    ):
        svc.index_documents(reindex=True)
    mocks["chunk_repo"].delete_by_document_id.assert_called_once_with(1)
    mocks["store"].delete_by_document_id.assert_called_once_with(1)
