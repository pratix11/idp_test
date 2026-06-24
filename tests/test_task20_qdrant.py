"""Task 20 — Qdrant Vector Store.

Unit tests use an injected mock QdrantClient — no Qdrant process needed.
Integration tests (marked `qdrant`) require `docker compose up -d qdrant`.
"""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from property_intel.retrieval.models import ScoredChunk, VectorPoint
from property_intel.retrieval.vector_store import COLLECTION_NAME, VECTOR_DIM, QdrantStore

# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_client() -> MagicMock:
    from qdrant_client import QdrantClient

    return MagicMock(spec=QdrantClient)


def _unit_vector(dim: int = VECTOR_DIM) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_point(point_id: int, document_id: int = 1) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        vector=_unit_vector(),
        payload={
            "document_id": document_id,
            "chunk_index": point_id,
            "content": f"Content of chunk {point_id}",
            "section_title": "Section A",
        },
    )


# ── unit tests (mocked client) ────────────────────────────────────────────────


def test_ensure_collection_creates_when_absent() -> None:
    client = _mock_client()
    client.collection_exists.return_value = False

    store = QdrantStore(client=client)
    store.ensure_collection()

    client.create_collection.assert_called_once()
    args = client.create_collection.call_args
    assert args.kwargs["collection_name"] == COLLECTION_NAME


def test_ensure_collection_skips_when_already_exists() -> None:
    client = _mock_client()
    client.collection_exists.return_value = True

    store = QdrantStore(client=client)
    store.ensure_collection()

    client.create_collection.assert_not_called()


def test_upsert_empty_list_is_no_op() -> None:
    client = _mock_client()
    store = QdrantStore(client=client)
    store.upsert([])
    client.upsert.assert_not_called()


def test_upsert_passes_correct_point_count() -> None:
    client = _mock_client()
    store = QdrantStore(client=client)
    points = [_make_point(i) for i in range(3)]

    store.upsert(points)

    call_kwargs = client.upsert.call_args.kwargs
    assert len(call_kwargs["points"]) == 3


def test_upsert_uses_correct_collection_name() -> None:
    client = _mock_client()
    store = QdrantStore(client=client)
    store.upsert([_make_point(1)])

    assert client.upsert.call_args.kwargs["collection_name"] == COLLECTION_NAME


def test_upsert_point_id_and_vector_preserved() -> None:
    from qdrant_client.models import PointStruct

    client = _mock_client()
    store = QdrantStore(client=client)
    p = _make_point(42)
    store.upsert([p])

    sent: PointStruct = client.upsert.call_args.kwargs["points"][0]
    assert sent.id == 42
    assert sent.vector == p.vector


def test_search_returns_scored_chunks() -> None:
    from qdrant_client.models import ScoredPoint

    client = _mock_client()
    mock_point = MagicMock(spec=ScoredPoint)
    mock_point.id = 7
    mock_point.score = 0.92
    mock_point.payload = {
        "document_id": 1,
        "chunk_index": 0,
        "content": "Some text",
        "section_title": "Intro",
    }
    client.query_points.return_value.points = [mock_point]

    store = QdrantStore(client=client)
    results = store.search(_unit_vector(), limit=5)

    assert len(results) == 1
    assert isinstance(results[0], ScoredChunk)
    assert results[0].chunk_id == 7
    assert results[0].score == 0.92
    assert results[0].content == "Some text"


def test_search_with_document_id_passes_filter() -> None:
    from qdrant_client.models import ScoredPoint

    client = _mock_client()
    client.query_points.return_value.points = []

    store = QdrantStore(client=client)
    store.search(_unit_vector(), limit=5, document_id=99)

    call_kwargs = client.query_points.call_args.kwargs
    assert call_kwargs["query_filter"] is not None


def test_search_without_document_id_passes_no_filter() -> None:
    client = _mock_client()
    client.query_points.return_value.points = []

    store = QdrantStore(client=client)
    store.search(_unit_vector(), limit=5)

    call_kwargs = client.query_points.call_args.kwargs
    assert call_kwargs["query_filter"] is None


def test_delete_by_document_id_calls_delete() -> None:
    client = _mock_client()
    store = QdrantStore(client=client)
    store.delete_by_document_id(document_id=5)
    client.delete.assert_called_once()


def test_count_delegates_to_client() -> None:
    client = _mock_client()
    client.count.return_value.count = 42
    store = QdrantStore(client=client)
    assert store.count() == 42


def test_drop_collection_only_if_exists() -> None:
    client = _mock_client()
    client.collection_exists.return_value = False
    store = QdrantStore(client=client)
    store.drop_collection()
    client.delete_collection.assert_not_called()


# ── integration tests (require running Qdrant) ────────────────────────────────


@pytest.mark.qdrant
def test_ensure_collection_is_idempotent(qdrant_store: QdrantStore) -> None:
    qdrant_store.ensure_collection()
    qdrant_store.ensure_collection()
    # no exception = pass


@pytest.mark.qdrant
def test_upsert_and_count(qdrant_store: QdrantStore) -> None:
    points = [_make_point(i) for i in range(5)]
    qdrant_store.upsert(points)
    assert qdrant_store.count() == 5


@pytest.mark.qdrant
def test_upsert_is_idempotent(qdrant_store: QdrantStore) -> None:
    points = [_make_point(1)]
    qdrant_store.upsert(points)
    qdrant_store.upsert(points)
    assert qdrant_store.count() == 1


@pytest.mark.qdrant
def test_search_returns_results_in_score_order(qdrant_store: QdrantStore) -> None:
    query = _unit_vector()
    points = [_make_point(i) for i in range(10)]
    qdrant_store.upsert(points)

    results = qdrant_store.search(query, limit=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.qdrant
def test_search_filter_by_document_id(qdrant_store: QdrantStore) -> None:
    points = [_make_point(i, document_id=1) for i in range(3)]
    points += [_make_point(i + 3, document_id=2) for i in range(3)]
    qdrant_store.upsert(points)

    results = qdrant_store.search(_unit_vector(), limit=10, document_id=2)
    assert all(r.document_id == 2 for r in results)


@pytest.mark.qdrant
def test_delete_by_document_id(qdrant_store: QdrantStore) -> None:
    points = [_make_point(i, document_id=1) for i in range(3)]
    points += [_make_point(i + 3, document_id=2) for i in range(2)]
    qdrant_store.upsert(points)
    assert qdrant_store.count() == 5

    qdrant_store.delete_by_document_id(document_id=1)
    assert qdrant_store.count() == 2


@pytest.mark.qdrant
def test_search_scored_chunk_fields_populated(qdrant_store: QdrantStore) -> None:
    p = _make_point(99, document_id=7)
    qdrant_store.upsert([p])

    results = qdrant_store.search(p.vector, limit=1)
    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == 99
    assert r.document_id == 7
    assert r.content == "Content of chunk 99"
    assert r.section_title == "Section A"
    assert r.score > 0.99  # exact self-match may exceed 1.0 due to float32 rounding
