"""Task 23 — BGE Reranker (cross-encoder).

Unit tests mock CrossEncoder — no model download.
Slow tests (marked `slow`) use the real BAAI/bge-reranker-v2-m3 model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.reranker import Reranker

# ── helpers ───────────────────────────────────────────────────────────────────


def _chunk(chunk_id: int, content: str = "text", score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=chunk_id,
        content=content,
        section_title=None,
        score=score,
    )


def _fake_cross_encoder(scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = np.array(scores, dtype=np.float32)
    return mock


# ── unit tests (mocked model) ─────────────────────────────────────────────────


@pytest.fixture
def reranker_with_scores(request: pytest.FixtureRequest):
    """Return a Reranker whose CrossEncoder produces the given scores."""
    scores = getattr(request, "param", [0.9, 0.5, 0.1])
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder(scores)
        r = Reranker()
        yield r


def test_empty_chunks_returns_empty() -> None:
    reranker = Reranker.__new__(Reranker)
    reranker._model = None
    reranker._model_name = "BAAI/bge-reranker-v2-m3"
    assert reranker.rerank("query", []) == []


def test_blank_query_returns_chunks_unchanged() -> None:
    chunks = [_chunk(1), _chunk(2)]
    reranker = Reranker.__new__(Reranker)
    reranker._model = None
    reranker._model_name = "BAAI/bge-reranker-v2-m3"
    assert reranker.rerank("  ", chunks) == chunks


def test_model_not_loaded_at_instantiation() -> None:
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        Reranker()
        mock_cls.assert_not_called()


def test_model_loaded_on_first_rerank_call() -> None:
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.8])
        r = Reranker()
        r.rerank("query", [_chunk(1)])
        mock_cls.assert_called_once()


def test_model_loaded_only_once_across_calls() -> None:
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.8, 0.5])
        r = Reranker()
        r.rerank("q", [_chunk(1), _chunk(2)])
        r.rerank("q", [_chunk(1), _chunk(2)])
        assert mock_cls.call_count == 1


def test_results_sorted_by_reranker_score_descending() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        # Cross-encoder assigns chunk 3 the highest score
        mock_cls.return_value = _fake_cross_encoder([0.1, 0.5, 0.9])
        r = Reranker()
        results = r.rerank("query", chunks)
    assert [res.chunk_id for res in results] == [3, 2, 1]


def test_reranker_scores_replace_original_scores() -> None:
    chunks = [_chunk(1, score=0.99)]
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.42])
        results = Reranker().rerank("query", chunks)
    assert abs(results[0].score - 0.42) < 1e-5


def test_top_k_truncates_results() -> None:
    chunks = [_chunk(i) for i in range(5)]
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.9, 0.8, 0.7, 0.6, 0.5])
        results = Reranker().rerank("query", chunks, top_k=3)
    assert len(results) == 3


def test_top_k_none_returns_all() -> None:
    chunks = [_chunk(i) for i in range(4)]
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.9, 0.8, 0.7, 0.6])
        results = Reranker().rerank("query", chunks, top_k=None)
    assert len(results) == 4


def test_pairs_passed_to_model_correctly() -> None:
    chunks = [_chunk(1, content="About refunds"), _chunk(2, content="About registration")]
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.9, 0.4])
        Reranker().rerank("refund policy", chunks)
        pairs = mock_cls.return_value.predict.call_args[0][0]
    assert pairs == [("refund policy", "About refunds"), ("refund policy", "About registration")]


def test_result_fields_preserved_after_reranking() -> None:
    original = ScoredChunk(
        chunk_id=77, document_id=3, chunk_index=5,
        content="Important text", section_title="Chapter 4", score=0.2,
    )
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.95])
        result = Reranker().rerank("query", [original])[0]
    assert result.chunk_id == 77
    assert result.document_id == 3
    assert result.section_title == "Chapter 4"
    assert result.content == "Important text"


def test_custom_model_name_used() -> None:
    with patch("property_intel.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_cls.return_value = _fake_cross_encoder([0.5])
        Reranker(model_name="custom/reranker").rerank("q", [_chunk(1)])
        mock_cls.assert_called_once_with("custom/reranker")


# ── slow tests (real model) ───────────────────────────────────────────────────


@pytest.mark.slow
def test_real_reranker_returns_correct_count() -> None:
    r = Reranker()
    chunks = [_chunk(i, f"Document text {i}") for i in range(5)]
    results = r.rerank("regulatory compliance", chunks)
    assert len(results) == 5


@pytest.mark.slow
def test_real_reranker_relevant_chunk_ranks_first() -> None:
    r = Reranker()
    chunks = [
        _chunk(1, "monsoon rainfall statistics for coastal districts"),
        _chunk(2, "developer must refund amount within 45 days of cancellation"),
        _chunk(3, "annual report of MahaRERA for fiscal year 2023"),
    ]
    results = r.rerank("refund on cancellation of flat booking", chunks)
    assert results[0].chunk_id == 2
