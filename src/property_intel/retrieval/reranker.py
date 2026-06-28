from __future__ import annotations

import cohere

from property_intel.retrieval.models import ScoredChunk


class Reranker:
    """Re-score retrieved chunks using the Cohere Rerank API.

    A bi-encoder (text-embedding-3-small) embeds query and document separately —
    fast but loses cross-attention context.  Cohere's reranker sees the
    (query, document) pair together — more accurate but only practical on a
    small candidate set (~30 results).

    Two-stage pattern:
      1. Retrieve ~30 candidates quickly with BM25 / vector search.
      2. Re-rank those 30 via Cohere API — one API call, no local model.

    An injectable client allows unit tests to pass a mock without API calls.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "rerank-v3.5",
        *,
        client: cohere.ClientV2 | None = None,
    ) -> None:
        self._client = client or cohere.ClientV2(api_key=api_key)
        self._model = model

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return chunks re-sorted by Cohere relevance score.

        Args:
            query:  The original user query string.
            chunks: Candidate chunks from retrieval (BM25, semantic, or hybrid).
            top_k:  If set, return only the top k results after reranking.
        """
        if not chunks or not query.strip():
            return chunks

        docs = [c.content for c in chunks]
        if top_k is not None:
            response = self._client.rerank(
                model=self._model, query=query, documents=docs, top_n=top_k
            )
        else:
            response = self._client.rerank(
                model=self._model, query=query, documents=docs
            )
        return [
            ScoredChunk(
                chunk_id=chunks[r.index].chunk_id,
                document_id=chunks[r.index].document_id,
                chunk_index=chunks[r.index].chunk_index,
                content=chunks[r.index].content,
                section_title=chunks[r.index].section_title,
                score=r.relevance_score,
            )
            for r in response.results
        ]

    @classmethod
    def from_settings(cls, settings: object) -> Reranker:
        from property_intel.config import Settings

        assert isinstance(settings, Settings)
        return cls(api_key=settings.cohere_api_key, model=settings.cohere_rerank_model)
