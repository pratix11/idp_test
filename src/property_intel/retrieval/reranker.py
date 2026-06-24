from __future__ import annotations

from sentence_transformers import CrossEncoder

from property_intel.retrieval.models import ScoredChunk


class Reranker:
    """Re-score retrieved chunks with a cross-encoder for higher precision.

    A bi-encoder (BGE-M3) embeds query and document *separately* — it's fast
    but loses context about how they relate.  A cross-encoder sees the
    (query, document) pair together with full attention — far more accurate
    but too slow to run on the entire corpus.

    The two-stage pattern:
      1. Retrieve ~30 candidates quickly with BM25 / vector search.
      2. Re-rank those 30 with the cross-encoder — only 30 forward passes.

    This gives retrieval quality close to running the cross-encoder everywhere
    at a fraction of the cost.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return chunks re-sorted by cross-encoder relevance score.

        Args:
            query:  The original user query string.
            chunks: Candidate chunks from retrieval (BM25, semantic, or hybrid).
            top_k:  If set, return only the top k results after reranking.
        """
        if not chunks or not query.strip():
            return chunks

        model = self._load_model()
        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores: list[float] = model.predict(pairs).tolist()  # type: ignore[arg-type,union-attr]

        ranked = sorted(zip(chunks, raw_scores), key=lambda pair: pair[1], reverse=True)
        reranked = [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section_title=chunk.section_title,
                score=float(score),
            )
            for chunk, score in ranked
        ]
        return reranked[:top_k] if top_k is not None else reranked
