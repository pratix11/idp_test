"""Task 23 — Cohere Reranker.

Unit tests inject a mock cohere.ClientV2 client so no API calls are made.
The slow tests (marked `slow`) call the real Cohere API and require COHERE_API_KEY.
"""

from unittest.mock import MagicMock

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
    """Mock cohere.ClientV2 whose rerank() returns results sorted by score desc."""
    mock = MagicMock()

    def _rerank(
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: object,
    ) -> MagicMock:
        n = len(documents)
        indexed = sorted(
            enumerate(scores[:n]), key=lambda x: x[1], reverse=True
        )
        if top_n is not None:
            indexed = indexed[:top_n]
        results = []
        for idx, score in indexed:
            r = MagicMock()
            r.index = idx
            r.relevance_score = score
            results.append(r)
        response = MagicMock()
        response.results = results
        return response

    mock.rerank.side_effect = _rerank
    return mock


def _make_reranker(scores: list[float]) -> Reranker:
    return Reranker(api_key="test-key", client=_fake_cohere_client(scores))


# ── unit tests (mocked client) ────────────────────────────────────────────────


def test_empty_chunks_returns_empty() -> None:
    r = Reranker(api_key="test-key", client=MagicMock())
    assert r.rerank("query", []) == []


def test_blank_query_returns_chunks_unchanged() -> None:
    chunks = [_chunk(1), _chunk(2)]
    r = Reranker(api_key="test-key", client=MagicMock())
    assert r.rerank("  ", chunks) == chunks


def test_results_sorted_by_reranker_score_descending() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    # Cohere assigns chunk 3 the highest score
    r = _make_reranker([0.1, 0.5, 0.9])
    results = r.rerank("query", chunks)
    assert [res.chunk_id for res in results] == [3, 2, 1]


def test_reranker_scores_replace_original_scores() -> None:
    chunks = [_chunk(1, score=0.99)]
    r = _make_reranker([0.42])
    results = r.rerank("query", chunks)
    assert abs(results[0].score - 0.42) < 1e-5


def test_top_k_truncates_results() -> None:
    chunks = [_chunk(i) for i in range(5)]
    r = _make_reranker([0.9, 0.8, 0.7, 0.6, 0.5])
    results = r.rerank("query", chunks, top_k=3)
    assert len(results) == 3


def test_top_k_none_returns_all() -> None:
    chunks = [_chunk(i) for i in range(4)]
    r = _make_reranker([0.9, 0.8, 0.7, 0.6])
    results = r.rerank("query", chunks, top_k=None)
    assert len(results) == 4


def test_documents_passed_to_api_correctly() -> None:
    chunks = [_chunk(1, content="About refunds"), _chunk(2, content="About registration")]
    mock_client = _fake_cohere_client([0.9, 0.4])
    r = Reranker(api_key="test-key", client=mock_client)
    r.rerank("refund policy", chunks)
    call_kwargs = mock_client.rerank.call_args.kwargs
    assert call_kwargs["query"] == "refund policy"
    assert call_kwargs["documents"] == ["About refunds", "About registration"]


def test_result_fields_preserved_after_reranking() -> None:
    original = ScoredChunk(
        chunk_id=77, document_id=3, chunk_index=5,
        content="Important text", section_title="Chapter 4", score=0.2,
    )
    r = _make_reranker([0.95])
    result = r.rerank("query", [original])[0]
    assert result.chunk_id == 77
    assert result.document_id == 3
    assert result.section_title == "Chapter 4"
    assert result.content == "Important text"


def test_custom_model_name_used() -> None:
    mock_client = _fake_cohere_client([0.5])
    r = Reranker(api_key="test-key", model="rerank-english-v3.0", client=mock_client)
    r.rerank("q", [_chunk(1)])
    assert mock_client.rerank.call_args.kwargs["model"] == "rerank-english-v3.0"


def test_top_k_passed_as_top_n_to_api() -> None:
    mock_client = _fake_cohere_client([0.9, 0.8, 0.7])
    r = Reranker(api_key="test-key", client=mock_client)
    r.rerank("query", [_chunk(i) for i in range(3)], top_k=2)
    assert mock_client.rerank.call_args.kwargs.get("top_n") == 2


def test_top_k_none_does_not_pass_top_n_to_api() -> None:
    mock_client = _fake_cohere_client([0.9, 0.8])
    r = Reranker(api_key="test-key", client=mock_client)
    r.rerank("query", [_chunk(i) for i in range(2)], top_k=None)
    assert "top_n" not in mock_client.rerank.call_args.kwargs


# ── slow tests (real Cohere API) ──────────────────────────────────────────────


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
