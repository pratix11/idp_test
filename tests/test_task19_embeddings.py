"""Task 19 — OpenAI Embedding Generation.

Unit tests inject a mock openai.OpenAI client so no API calls are made.
The slow tests (marked `slow`) call the real OpenAI API and require OPENAI_API_KEY.
"""

from unittest.mock import MagicMock

import pytest

from property_intel.retrieval.embeddings import EMBEDDING_DIM, EmbeddingService
from property_intel.retrieval.models import DocumentChunk


# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_openai_client(dim: int = EMBEDDING_DIM) -> MagicMock:
    """Mock openai.OpenAI that returns random float vectors from embeddings.create()."""
    mock = MagicMock()

    def _create(input: list[str], model: str, **kwargs: object) -> MagicMock:
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[float(i % 10) / 10 for i in range(dim)])
            for _ in input
        ]
        return response

    mock.embeddings.create.side_effect = _create
    return mock


def _make_chunk(content: str = "test text", idx: int = 0) -> DocumentChunk:
    return DocumentChunk(document_id=1, chunk_index=idx, content=content, token_count=2)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── unit tests (mocked client, no API calls) ──────────────────────────────────


@pytest.fixture
def service() -> EmbeddingService:
    return EmbeddingService(api_key="test-key", client=_fake_openai_client())


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


def test_api_called_with_correct_model() -> None:
    mock_client = _fake_openai_client()
    svc = EmbeddingService(api_key="test-key", model="text-embedding-3-small", client=mock_client)
    svc.embed(["hello"])
    mock_client.embeddings.create.assert_called_once()
    call_kwargs = mock_client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


def test_api_called_once_per_batch() -> None:
    """All texts in one embed() call go to the API in a single request."""
    mock_client = _fake_openai_client()
    svc = EmbeddingService(api_key="test-key", client=mock_client)
    svc.embed(["a", "b", "c"])
    assert mock_client.embeddings.create.call_count == 1


def test_custom_model_name_used() -> None:
    mock_client = _fake_openai_client()
    svc = EmbeddingService(api_key="test-key", model="text-embedding-3-large", client=mock_client)
    svc.embed(["x"])
    call_kwargs = mock_client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-large"


def test_embedding_dim_constant_is_1536() -> None:
    assert EMBEDDING_DIM == 1536


# ── slow tests (real OpenAI API, skipped in fast runs) ───────────────────────


@pytest.mark.slow
def test_real_api_produces_correct_dimension() -> None:
    svc = EmbeddingService()
    result = svc.embed(["MahaRERA circular on refund policy for homebuyers"])
    assert len(result[0]) == EMBEDDING_DIM


@pytest.mark.slow
def test_real_api_returns_float_lists() -> None:
    svc = EmbeddingService()
    result = svc.embed(["test document about property regulations"])
    assert isinstance(result[0], list)
    assert all(isinstance(v, float) for v in result[0])


@pytest.mark.slow
def test_similar_texts_have_higher_similarity_than_dissimilar() -> None:
    """The whole point of semantic embeddings — similar meaning = close vectors."""
    svc = EmbeddingService()
    vecs_sim = svc.embed([
        "builder cannot cancel flat booking after allotment",
        "developer not allowed to withdraw from agreement with homebuyer",
    ])
    vecs_dis = svc.embed([
        "builder cannot cancel flat booking after allotment",
        "annual monsoon rainfall statistics for coastal Maharashtra",
    ])
    sim_score = _dot(vecs_sim[0], vecs_sim[1])
    dis_score = _dot(vecs_dis[0], vecs_dis[1])
    assert sim_score > dis_score
