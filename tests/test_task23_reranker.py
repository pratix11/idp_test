"""Task 23 — Cohere Reranker.

Unit tests mock cohere.ClientV2 — no API calls made.
"""

from unittest.mock import MagicMock, patch

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


def _fake_cohere_client(scores: list[float]) -> MagicMock:
    """Mock cohere.ClientV2 whose rerank() returns results ordered by descending score."""

    def _rerank(
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> MagicMock:
        # Cohere returns results sorted by relevance_score descending, each with .index
        indexed = sorted(enumerate(scores[:top_n]), key=lambda x: x[1], reverse=True)
        results = []
        for orig_idx, score in indexed:
            r = MagicMock()
            r.index = orig_idx
            r.relevance_score = score
            results.append(r)
        response = MagicMock()
        response.results = results
        return response

    client = MagicMock()
    client.rerank.side_effect = _rerank
    return client


# ── unit tests (mocked client) ────────────────────────────────────────────────


@pytest.fixture
def reranker_with_scores(request: pytest.FixtureRequest) -> Reranker:
    """Return a Reranker whose Cohere client produces the given scores."""
    scores = getattr(request, "param", [0.9, 0.5, 0.1])
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client(scores)
        r = Reranker()
        yield r


def test_empty_chunks_returns_empty() -> None:
    with patch("property_intel.retrieval.reranker.cohere.ClientV2"):
        reranker = Reranker()
        assert reranker.rerank("query", []) == []


def test_blank_query_returns_chunks_unchanged() -> None:
    chunks = [_chunk(1), _chunk(2)]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2"):
        reranker = Reranker()
        assert reranker.rerank("  ", chunks) == chunks


def test_client_created_at_instantiation() -> None:
    """cohere.ClientV2 is created eagerly in __init__."""
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        Reranker()
        mock_cls.assert_called_once()


def test_results_sorted_by_reranker_score_descending() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client([0.1, 0.5, 0.9])
        r = Reranker()
        results = r.rerank("query", chunks)
    assert [res.chunk_id for res in results] == [3, 2, 1]


def test_reranker_scores_replace_original_scores() -> None:
    chunks = [_chunk(1, score=0.99)]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client([0.42])
        results = Reranker().rerank("query", chunks)
    assert abs(results[0].score - 0.42) < 1e-5


def test_top_k_truncates_results() -> None:
    chunks = [_chunk(i) for i in range(5)]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client([0.9, 0.8, 0.7, 0.6, 0.5])
        results = Reranker().rerank("query", chunks, top_k=3)
    assert len(results) == 3


def test_top_k_none_returns_all() -> None:
    chunks = [_chunk(i) for i in range(4)]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client([0.9, 0.8, 0.7, 0.6])
        results = Reranker().rerank("query", chunks, top_k=None)
    assert len(results) == 4


def test_rerank_api_called_with_correct_args() -> None:
    chunks = [_chunk(1, content="About refunds"), _chunk(2, content="About registration")]
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        fake_client = _fake_cohere_client([0.9, 0.4])
        mock_cls.return_value = fake_client
        Reranker().rerank("refund policy", chunks)
        call_kwargs = fake_client.rerank.call_args.kwargs
    assert call_kwargs["query"] == "refund policy"
    assert call_kwargs["documents"] == ["About refunds", "About registration"]


def test_result_fields_preserved_after_reranking() -> None:
    original = ScoredChunk(
        chunk_id=77, document_id=3, chunk_index=5,
        content="Important text", section_title="Chapter 4", score=0.2,
    )
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        mock_cls.return_value = _fake_cohere_client([0.95])
        result = Reranker().rerank("query", [original])[0]
    assert result.chunk_id == 77
    assert result.document_id == 3
    assert result.section_title == "Chapter 4"
    assert result.content == "Important text"


def test_custom_model_name_passed_to_api() -> None:
    with patch("property_intel.retrieval.reranker.cohere.ClientV2") as mock_cls:
        fake_client = _fake_cohere_client([0.5])
        mock_cls.return_value = fake_client
        Reranker(model_name="custom/reranker").rerank("q", [_chunk(1)])
        call_kwargs = fake_client.rerank.call_args.kwargs
    assert call_kwargs["model"] == "custom/reranker"
