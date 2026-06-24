"""Task 21 — Semantic Search Pipeline.

All tests use stub EmbeddingService and QdrantStore so no model download
or running Qdrant is required. The pipeline logic is fully tested here;
the individual components are already tested in tasks 19 and 20.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from property_intel.retrieval.embeddings import EmbeddingService
from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.semantic_search import SemanticSearch
from property_intel.retrieval.vector_store import QdrantStore

# ── helpers ───────────────────────────────────────────────────────────────────

VECTOR_DIM = 1024


def _unit_vector() -> list[float]:
    v = np.random.randn(VECTOR_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _fake_embedder(vector: list[float] | None = None) -> EmbeddingService:
    """Stub EmbeddingService that returns a fixed (or random) vector."""
    mock = MagicMock(spec=EmbeddingService)
    mock.embed.return_value = [vector or _unit_vector()]
    return mock


def _fake_store(results: list[ScoredChunk] | None = None) -> QdrantStore:
    mock = MagicMock(spec=QdrantStore)
    mock.search.return_value = results or []
    return mock


def _scored_chunk(chunk_id: int = 1, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=0,
        content=f"Content {chunk_id}",
        section_title="Section",
        score=score,
    )


# ── unit tests ────────────────────────────────────────────────────────────────


def test_empty_query_returns_no_results() -> None:
    svc = SemanticSearch(_fake_embedder(), _fake_store())
    assert svc.search("") == []
    assert svc.search("   ") == []


def test_embed_called_with_query_text() -> None:
    embedder = _fake_embedder()
    store = _fake_store()
    SemanticSearch(embedder, store).search("refund policy")
    embedder.embed.assert_called_once_with(["refund policy"])


def test_store_search_called_with_embedding() -> None:
    vector = _unit_vector()
    embedder = _fake_embedder(vector)
    store = _fake_store()
    SemanticSearch(embedder, store).search("refund policy", limit=5)
    store.search.assert_called_once_with(vector, limit=5, document_id=None)


def test_document_id_filter_forwarded_to_store() -> None:
    embedder = _fake_embedder()
    store = _fake_store()
    SemanticSearch(embedder, store).search("cancellation", limit=3, document_id=7)
    store.search.assert_called_once_with(
        embedder.embed.return_value[0], limit=3, document_id=7
    )


def test_results_returned_as_is() -> None:
    chunks = [_scored_chunk(1, 0.95), _scored_chunk(2, 0.80)]
    svc = SemanticSearch(_fake_embedder(), _fake_store(chunks))
    result = svc.search("promoter obligations")
    assert result == chunks


def test_empty_store_results_returns_empty_list() -> None:
    svc = SemanticSearch(_fake_embedder(), _fake_store([]))
    assert svc.search("something") == []


def test_embed_not_called_for_blank_query() -> None:
    embedder = _fake_embedder()
    SemanticSearch(embedder, _fake_store()).search("  \n  ")
    embedder.embed.assert_not_called()


def test_limit_forwarded_correctly() -> None:
    embedder = _fake_embedder()
    store = _fake_store()
    SemanticSearch(embedder, store).search("query", limit=20)
    assert store.search.call_args.kwargs["limit"] == 20


def test_result_count_respects_limit() -> None:
    chunks = [_scored_chunk(i, 0.9 - i * 0.05) for i in range(10)]
    store = _fake_store(chunks[:5])  # store returns at most limit
    store.search.return_value = chunks[:5]
    svc = SemanticSearch(_fake_embedder(), store)
    result = svc.search("query", limit=5)
    assert len(result) <= 5


def test_results_preserve_all_scored_chunk_fields() -> None:
    expected = ScoredChunk(
        chunk_id=42,
        document_id=7,
        chunk_index=3,
        content="Important regulatory text",
        section_title="Chapter 2",
        score=0.87,
    )
    svc = SemanticSearch(_fake_embedder(), _fake_store([expected]))
    result = svc.search("regulatory")
    assert result[0] == expected
