"""Task 19 — OpenAI Embedding Generation.

Unit tests mock openai.OpenAI so no API calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from property_intel.retrieval.embeddings import EMBEDDING_DIM, EmbeddingService
from property_intel.retrieval.models import DocumentChunk

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_openai_client(dim: int = EMBEDDING_DIM) -> MagicMock:
    """Mock openai.OpenAI whose embeddings.create() returns fake unit vectors."""
    import random

    def _create(model: str, input: list[str]) -> MagicMock:  # noqa: A002
        response = MagicMock()
        items = []
        for _ in input:
            raw = [random.gauss(0, 1) for _ in range(dim)]
            norm = sum(x * x for x in raw) ** 0.5
            item = MagicMock()
            item.embedding = [x / norm for x in raw]
            items.append(item)
        response.data = items
        return response

    client = MagicMock()
    client.embeddings.create.side_effect = _create
    return client


def _make_chunk(content: str = "test text", idx: int = 0) -> DocumentChunk:
    return DocumentChunk(document_id=1, chunk_index=idx, content=content, token_count=2)


# ── unit tests (mocked client) ────────────────────────────────────────────────


@pytest.fixture
def service() -> EmbeddingService:
    """EmbeddingService with patched openai.OpenAI."""
    with patch("property_intel.retrieval.embeddings.openai.OpenAI") as mock_cls:
        mock_cls.return_value = _fake_openai_client()
        svc = EmbeddingService(model_name="text-embedding-3-small", api_key="test-key")
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


def test_client_created_at_instantiation() -> None:
    """openai.OpenAI client is created eagerly in __init__."""
    with patch("property_intel.retrieval.embeddings.openai.OpenAI") as mock_cls:
        mock_cls.return_value = _fake_openai_client()
        EmbeddingService(api_key="key")
        mock_cls.assert_called_once()


def test_model_name_passed_to_embeddings_create() -> None:
    with patch("property_intel.retrieval.embeddings.openai.OpenAI") as mock_cls:
        fake_client = _fake_openai_client()
        mock_cls.return_value = fake_client
        svc = EmbeddingService(model_name="text-embedding-3-small", api_key="key")
        svc.embed(["hello"])
        call_kwargs = fake_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


def test_embed_passes_all_texts_in_one_request() -> None:
    with patch("property_intel.retrieval.embeddings.openai.OpenAI") as mock_cls:
        fake_client = _fake_openai_client()
        mock_cls.return_value = fake_client
        svc = EmbeddingService(api_key="key")
        svc.embed(["a", "b", "c"])
        call_kwargs = fake_client.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == ["a", "b", "c"]
