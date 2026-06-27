from __future__ import annotations

import cohere

from property_intel.retrieval.models import ScoredChunk


class Reranker:
    """Re-score retrieved chunks with Cohere Rerank for higher precision.

    A bi-encoder (OpenAI embeddings) embeds query and document *separately* —
    it's fast but loses context about how they relate.  Cohere Rerank sees the
    (query, document) pair together — far more accurate but too slow to run on
    the entire corpus.

    The two-stage pattern:
      1. Retrieve ~30 candidates quickly with BM25 / vector search.
      2. Re-rank those 30 with Cohere Rerank — only 30 API calls.
    """

    def __init__(self, model_name: str = "rerank-v3.5", api_key: str = "") -> None:
        self._model_name = model_name
        self._co = cohere.ClientV2(api_key=api_key)

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

        response = self._co.rerank(
            model=self._model_name,
            query=query,
            documents=[c.content for c in chunks],
            top_n=top_k if top_k is not None else len(chunks),
        )

        return [
            ScoredChunk(
                chunk_id=chunks[r.index].chunk_id,
                document_id=chunks[r.index].document_id,
                chunk_index=chunks[r.index].chunk_index,
                content=chunks[r.index].content,
                section_title=chunks[r.index].section_title,
                score=float(r.relevance_score),
            )
            for r in response.results
        ]
