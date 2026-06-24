"""Task 19 — BGE-M3 Embedding Generation.

Unit tests mock SentenceTransformer so no model download is needed.
Slow tests (marked `slow`) use the real BGE-M3 model and are skipped
in fast iteration (uv run pytest -m 'not slow').
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from property_intel.retrieval.embeddings import EMBEDDING_DIM, EmbeddingService
from property_intel.retrieval.models import DocumentChunk

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_model(dim: int = EMBEDDING_DIM) -> MagicMock:
    """Mock SentenceTransformer that returns random unit vectors."""
    mock = MagicMock()

    def _encode(texts: list[str], **kwargs: object) -> np.ndarray:
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.encode.side_effect = _encode
    return mock


def _make_chunk(content: str = "test text", idx: int = 0) -> DocumentChunk:
    return DocumentChunk(document_id=1, chunk_index=idx, content=content, token_count=2)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── unit tests (mocked model, no download) ───────────────────────────────────


@pytest.fixture
def service() -> EmbeddingService:
    """EmbeddingService with patched SentenceTransformer."""
    with patch("property_intel.retrieval.embeddings.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _fake_model()
        svc = EmbeddingService(model_name="BAAI/bge-m3", batch_size=8)
        yield svc


def test_embed_empty_list_returns_empty(service: EmbeddingService) -> None:
    assert service.embed([]) == []


def test_embed_returns_one_vector_per_text(service: EmbeddingService) -> None:
    result = service.embed(["a", "b", "c"])
    assert len(result) == 3


def test_embed_returns_correct_dimension(service: EmbeddingService) -> None:
    result = service.embed(["hello world"])
    assert len(result[0]) == EMBEDDING_DIM


def test_embed_returns_lists_of_python_floats(service: EmbeddingService) -> None:
    result = service.embed(["test"])
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)


def test_embed_chunks_empty_returns_empty(service: EmbeddingService) -> None:
    assert service.embed_chunks([]) == []


def test_embed_chunks_returns_paired_tuples(service: EmbeddingService) -> None:
    chunks = [_make_chunk("Text A", 0), _make_chunk("Text B", 1)]
    pairs = service.embed_chunks(chunks)
    assert len(pairs) == 2
    assert pairs[0][0] is chunks[0]
    assert pairs[1][0] is chunks[1]


def test_embed_chunks_vectors_have_correct_dim(service: EmbeddingService) -> None:
    pairs = service.embed_chunks([_make_chunk()])
    assert len(pairs[0][1]) == EMBEDDING_DIM


def test_model_not_loaded_at_instantiation() -> None:
    """Importing or constructing should not trigger a model download."""
    with patch("property_intel.retrieval.embeddings.SentenceTransformer") as mock_cls:
        EmbeddingService()
        mock_cls.assert_not_called()


def test_model_loaded_on_first_embed_call() -> None:
    with patch("property_intel.retrieval.embeddings.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _fake_model()
        svc = EmbeddingService()
        svc.embed(["trigger"])
        mock_cls.assert_called_once_with("BAAI/bge-m3")


def test_model_loaded_only_once_across_multiple_calls() -> None:
    with patch("property_intel.retrieval.embeddings.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _fake_model()
        svc = EmbeddingService()
        svc.embed(["first"])
        svc.embed(["second"])
        svc.embed(["third"])
        assert mock_cls.call_count == 1


def test_custom_model_name_passed_to_sentence_transformer() -> None:
    with patch("property_intel.retrieval.embeddings.SentenceTransformer") as mock_cls:
        mock_cls.return_value = _fake_model()
        svc = EmbeddingService(model_name="custom/model")
        svc.embed(["x"])
        mock_cls.assert_called_once_with("custom/model")


# ── slow tests (real BGE-M3, skipped in fast runs) ───────────────────────────


@pytest.mark.slow
def test_real_model_produces_correct_dimension() -> None:
    svc = EmbeddingService()
    result = svc.embed(["MahaRERA circular on refund policy for homebuyers"])
    assert len(result[0]) == EMBEDDING_DIM


@pytest.mark.slow
def test_real_model_vectors_are_unit_normalised() -> None:
    svc = EmbeddingService()
    result = svc.embed(["test document about property regulations"])
    norm = math.sqrt(sum(x * x for x in result[0]))
    assert abs(norm - 1.0) < 1e-5


@pytest.mark.slow
def test_similar_texts_have_higher_similarity_than_dissimilar() -> None:
    """The whole point of semantic embeddings — similar meaning = close vectors."""
    svc = EmbeddingService()
    # Semantically similar: same concept, different words
    vecs_sim = svc.embed([
        "builder cannot cancel flat booking after allotment",
        "developer not allowed to withdraw from agreement with homebuyer",
    ])
    # Semantically dissimilar
    vecs_dis = svc.embed([
        "builder cannot cancel flat booking after allotment",
        "annual monsoon rainfall statistics for coastal Maharashtra",
    ])
    sim_score = _dot(vecs_sim[0], vecs_sim[1])
    dis_score = _dot(vecs_dis[0], vecs_dis[1])
    assert sim_score > dis_score


@pytest.mark.slow
def test_real_model_embed_chunks_integration() -> None:
    svc = EmbeddingService()
    chunks = [
        _make_chunk("Registration of real estate projects under MahaRERA", 0),
        _make_chunk("Obligations of promoter towards homebuyer", 1),
    ]
    pairs = svc.embed_chunks(chunks)
    assert len(pairs) == 2
    for chunk, vector in pairs:
        assert isinstance(chunk, DocumentChunk)
        assert len(vector) == EMBEDDING_DIM
