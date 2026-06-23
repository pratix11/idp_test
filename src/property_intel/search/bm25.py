import re

from rank_bm25 import BM25Okapi

from property_intel.db.models import DocumentModel
from property_intel.db.repository import DocumentRepository
from property_intel.search.schema import SearchFilters, SearchQuery, SearchResultItem, SearchResultPage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[DocumentModel]) -> None:
        self._documents = documents
        self._corpus = [_tokenize(f"{doc.title} {doc.content or ''}") for doc in documents]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def rank(self, text: str) -> list[tuple[DocumentModel, float]]:
        tokens = _tokenize(text)
        if self._bm25 is None or not tokens:
            return []
        token_set = set(tokens)
        scores = self._bm25.get_scores(tokens)
        candidates = [
            (doc, float(score))
            for doc, score, doc_tokens in zip(self._documents, scores, self._corpus)
            if token_set & set(doc_tokens)
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates


class BM25Search:
    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    def search(self, query: SearchQuery) -> SearchResultPage:
        text = (query.text or "").strip()
        if not text:
            return SearchResultPage(items=[], total=0, page=query.page, page_size=query.page_size)

        index = BM25Index(self.repository.list_searchable())
        ranked = [
            (doc, score)
            for doc, score in index.rank(text)
            if self._matches_filters(doc, query.filters)
        ]

        total = len(ranked)
        page_items = ranked[query.offset : query.offset + query.page_size]

        items = [
            SearchResultItem(
                document_id=doc.id,
                title=doc.title,
                category=doc.category,
                source=doc.source,
                document_type=doc.document_type,
                date=doc.date,
                score=score,
            )
            for doc, score in page_items
        ]
        return SearchResultPage(items=items, total=total, page=query.page, page_size=query.page_size)

    @staticmethod
    def _matches_filters(doc: DocumentModel, filters: SearchFilters) -> bool:
        if filters.category is not None and doc.category != filters.category.value:
            return False
        if filters.source is not None and doc.source != filters.source:
            return False
        if filters.document_type is not None and doc.document_type != filters.document_type:
            return False
        if filters.date_from is not None and doc.date < filters.date_from:
            return False
        if filters.date_to is not None and doc.date > filters.date_to:
            return False
        return True
