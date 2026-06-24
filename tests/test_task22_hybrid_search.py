"""Task 22 — Hybrid Search (BM25 + Semantic via RRF).

Tests are split into three groups:
  1. rrf_fusion — pure function, no mocks.
  2. ChunkBM25Search — mocked ChunkRepository.
  3. HybridSearch — mocked ChunkBM25Search + SemanticSearch.
"""

from unittest.mock import MagicMock

import pytest

from property_intel.db.repository import ChunkRepository
from property_intel.retrieval.hybrid_search import ChunkBM25Search, HybridSearch, rrf_fusion
from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.semantic_search import SemanticSearch

# ── helpers ───────────────────────────────────────────────────────────────────


def _chunk(
    chunk_id: int,
    content: str = "text",
    score: float = 1.0,
    document_id: int = 1,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_id,
        content=content,
        section_title=None,
        score=score,
    )


def _db_chunk(chunk_id: int, content: str = "some text") -> MagicMock:
    m = MagicMock()
    m.id = chunk_id
    m.document_id = 1
    m.chunk_index = chunk_id
    m.content = content
    m.section_title = None
    return m


def _fake_repo(chunks: list) -> ChunkRepository:
    mock = MagicMock(spec=ChunkRepository)
    mock.list_all.return_value = chunks
    return mock


def _fake_bm25(results: list[ScoredChunk] | None = None) -> ChunkBM25Search:
    mock = MagicMock(spec=ChunkBM25Search)
    mock.search.return_value = results or []
    return mock


def _fake_semantic(results: list[ScoredChunk] | None = None) -> SemanticSearch:
    mock = MagicMock(spec=SemanticSearch)
    mock.search.return_value = results or []
    return mock


# ── rrf_fusion unit tests (pure function) ────────────────────────────────────


def test_rrf_empty_both_lists_returns_empty() -> None:
    assert rrf_fusion([], []) == []


def test_rrf_single_list_a_only() -> None:
    results = rrf_fusion([_chunk(1), _chunk(2)], [])
    assert [r.chunk_id for r in results] == [1, 2]


def test_rrf_single_list_b_only() -> None:
    results = rrf_fusion([], [_chunk(3), _chunk(4)])
    assert [r.chunk_id for r in results] == [3, 4]


def test_rrf_item_in_both_lists_gets_boosted_score() -> None:
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(1), _chunk(3)]
    results = rrf_fusion(a, b)
    # chunk 1 appears in both lists at rank 1 → highest RRF score
    assert results[0].chunk_id == 1


def test_rrf_scores_are_positive_and_ordered_descending() -> None:
    a = [_chunk(i) for i in range(5)]
    b = [_chunk(i) for i in reversed(range(5))]
    results = rrf_fusion(a, b)
    scores = [r.score for r in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplicates_chunk_ids() -> None:
    chunk = _chunk(42)
    results = rrf_fusion([chunk], [chunk])
    assert sum(1 for r in results if r.chunk_id == 42) == 1


def test_rrf_item_in_both_has_higher_score_than_item_in_one() -> None:
    both = _chunk(1)
    only_a = _chunk(2)
    results = rrf_fusion([both, only_a], [both])
    both_result = next(r for r in results if r.chunk_id == 1)
    one_result = next(r for r in results if r.chunk_id == 2)
    assert both_result.score > one_result.score


def test_rrf_custom_k_affects_scores() -> None:
    a = [_chunk(1)]
    r_k60 = rrf_fusion(a, [], k=60)[0].score
    r_k1 = rrf_fusion(a, [], k=1)[0].score
    assert r_k1 > r_k60  # smaller k → less damping → higher score for top rank


def test_rrf_result_fields_preserved_from_lookup() -> None:
    c = ScoredChunk(
        chunk_id=99, document_id=5, chunk_index=3,
        content="Regulatory text", section_title="Chapter 1", score=0.8,
    )
    results = rrf_fusion([c], [])
    assert results[0].document_id == 5
    assert results[0].content == "Regulatory text"
    assert results[0].section_title == "Chapter 1"


# ── ChunkBM25Search unit tests ────────────────────────────────────────────────


def test_bm25_empty_query_returns_empty() -> None:
    svc = ChunkBM25Search(_fake_repo([]))
    assert svc.search("") == []
    assert svc.search("   ") == []


def test_bm25_empty_corpus_returns_empty() -> None:
    svc = ChunkBM25Search(_fake_repo([]))
    assert svc.search("cancellation") == []


def test_bm25_matching_term_returns_result() -> None:
    # BM25 IDF requires multiple documents — IDF is 0/negative for 100%-frequent terms
    repo = _fake_repo([
        _db_chunk(1, "refund policy for homebuyers"),
        _db_chunk(2, "registration of real estate project"),
    ])
    results = ChunkBM25Search(repo).search("refund")
    assert len(results) >= 1
    assert results[0].chunk_id == 1


def test_bm25_non_matching_term_returns_empty() -> None:
    repo = _fake_repo([_db_chunk(1, "monsoon rainfall data")])
    svc = ChunkBM25Search(repo)
    results = svc.search("developer cancellation rights")
    assert results == []


def test_bm25_results_are_scored_chunks() -> None:
    repo = _fake_repo([_db_chunk(1, "registration of project")])
    results = ChunkBM25Search(repo).search("registration")
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_bm25_limit_respected() -> None:
    chunks = [_db_chunk(i, f"refund refund word{i}") for i in range(10)]
    repo = _fake_repo(chunks)
    results = ChunkBM25Search(repo).search("refund", limit=3)
    assert len(results) <= 3


def test_bm25_returns_descending_scores() -> None:
    chunks = [
        _db_chunk(1, "refund refund refund policy"),
        _db_chunk(2, "refund policy"),
        _db_chunk(3, "unrelated content here"),
    ]
    results = ChunkBM25Search(_fake_repo(chunks)).search("refund")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ── HybridSearch unit tests ───────────────────────────────────────────────────


def test_hybrid_empty_query_returns_empty() -> None:
    svc = HybridSearch(_fake_bm25(), _fake_semantic())
    assert svc.search("") == []
    assert svc.search("  ") == []


def test_hybrid_calls_both_backends() -> None:
    bm25 = _fake_bm25()
    semantic = _fake_semantic()
    HybridSearch(bm25, semantic).search("builder obligations")
    bm25.search.assert_called_once()
    semantic.search.assert_called_once()


def test_hybrid_result_in_both_lists_ranks_first() -> None:
    shared = _chunk(1, score=0.5)
    bm25 = _fake_bm25([shared, _chunk(2, score=0.3)])
    semantic = _fake_semantic([shared, _chunk(3, score=0.6)])
    results = HybridSearch(bm25, semantic).search("query")
    assert results[0].chunk_id == 1


def test_hybrid_limit_applied_to_output() -> None:
    bm25 = _fake_bm25([_chunk(i) for i in range(10)])
    semantic = _fake_semantic([_chunk(i) for i in range(10, 20)])
    results = HybridSearch(bm25, semantic).search("query", limit=5)
    assert len(results) <= 5


def test_hybrid_empty_bm25_falls_back_to_semantic() -> None:
    semantic_chunks = [_chunk(i) for i in range(3)]
    svc = HybridSearch(_fake_bm25([]), _fake_semantic(semantic_chunks))
    results = svc.search("something")
    assert {r.chunk_id for r in results} == {0, 1, 2}


def test_hybrid_empty_semantic_falls_back_to_bm25() -> None:
    bm25_chunks = [_chunk(i) for i in range(3)]
    svc = HybridSearch(_fake_bm25(bm25_chunks), _fake_semantic([]))
    results = svc.search("something")
    assert {r.chunk_id for r in results} == {0, 1, 2}


def test_hybrid_result_scores_are_rrf_not_raw() -> None:
    """RRF scores are always in (0, 1/(k+1)] range, not raw BM25/cosine values."""
    bm25 = _fake_bm25([_chunk(1, score=5.7)])
    semantic = _fake_semantic([_chunk(1, score=0.95)])
    results = HybridSearch(_fake_bm25([_chunk(1, score=5.7)]), _fake_semantic([_chunk(1, score=0.95)])).search("q")
    assert results[0].score < 1.0  # RRF score is always < 1 for finite k
