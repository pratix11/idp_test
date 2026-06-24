import re

from rank_bm25 import BM25Okapi

from property_intel.db.repository import ChunkRepository
from property_intel.retrieval.models import ScoredChunk
from property_intel.retrieval.semantic_search import SemanticSearch

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class ChunkBM25Search:
    """BM25 keyword search over chunk content (not document-level).

    Rebuilds the in-memory index on every call, same as the document-level
    BM25Search in Phase 2. Acceptable for the current corpus size (~730 chunks).
    """

    def __init__(self, repository: ChunkRepository) -> None:
        self._repo = repository

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        query = query.strip()
        if not query:
            return []
        chunks = self._repo.list_all()
        if not chunks:
            return []

        corpus = [_tokenize(c.content) for c in chunks]
        bm25 = BM25Okapi(corpus)
        query_tokens = _tokenize(query)
        query_token_set = set(query_tokens)
        raw_scores = bm25.get_scores(query_tokens)

        # Filter by token overlap, not score > 0: BM25 IDF can be 0 or negative
        # for terms that appear in a large fraction of a small corpus, but those
        # documents are still relevant matches.
        ranked = sorted(
            [
                (chunk, float(score))
                for chunk, score, doc_tokens in zip(chunks, raw_scores, corpus)
                if query_token_set & set(doc_tokens)
            ],
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            ScoredChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section_title=chunk.section_title,
                score=score,
            )
            for chunk, score in ranked[:limit]
        ]


def rrf_fusion(
    results_a: list[ScoredChunk],
    results_b: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion — merge two ranked lists without comparing raw scores.

    Formula: rrf(d) = sum over lists L of 1 / (k + rank_L(d))
    Items that appear in only one list still get a partial score.
    k=60 is the value from the original RRF paper (Cormack et al., 2009).
    """
    rrf_scores: dict[int, float] = {}
    lookup: dict[int, ScoredChunk] = {}

    for rank, chunk in enumerate(results_a, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        lookup[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(results_b, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        lookup[chunk.chunk_id] = chunk

    fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        ScoredChunk(
            chunk_id=cid,
            document_id=lookup[cid].document_id,
            chunk_index=lookup[cid].chunk_index,
            content=lookup[cid].content,
            section_title=lookup[cid].section_title,
            score=score,
        )
        for cid, score in fused
    ]


class HybridSearch:
    """Combines chunk-level BM25 and semantic search via Reciprocal Rank Fusion.

    Why hybrid beats either alone:
      - BM25 nails exact terms ("MahaRERA circular 42/2022") but misses synonyms.
      - Semantic search catches synonyms but can miss rare proper nouns.
      - RRF fusion lets results that appear in *both* lists bubble to the top,
        giving a recall + precision benefit over either signal alone.
    """

    def __init__(
        self,
        bm25: ChunkBM25Search,
        semantic: SemanticSearch,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25
        self._semantic = semantic
        self._rrf_k = rrf_k

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        if not query.strip():
            return []
        # Fetch 3× candidates so RRF has enough material to rerank
        candidates = limit * 3
        bm25_results = self._bm25.search(query, limit=candidates)
        semantic_results = self._semantic.search(query, limit=candidates)
        fused = rrf_fusion(bm25_results, semantic_results, k=self._rrf_k)
        return fused[:limit]
