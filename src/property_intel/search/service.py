from typing import Literal, Protocol

from sqlalchemy.orm import Session

from property_intel.db.repository import DocumentRepository
from property_intel.search.bm25 import BM25Search
from property_intel.search.fulltext import FullTextSearch
from property_intel.search.metadata_search import MetadataSearch
from property_intel.search.schema import SearchQuery, SearchResultPage

SearchMode = Literal["fulltext", "bm25", "metadata"]


class SearchBackend(Protocol):
    def search(self, query: SearchQuery) -> SearchResultPage: ...


class SearchService:
    def __init__(
        self,
        session: Session,
        *,
        fulltext: SearchBackend | None = None,
        bm25: SearchBackend | None = None,
        metadata: SearchBackend | None = None,
    ) -> None:
        self.session = session
        self._fulltext: SearchBackend = fulltext or FullTextSearch(session)
        self._bm25: SearchBackend = bm25 or BM25Search(DocumentRepository(session))
        self._metadata: SearchBackend = metadata or MetadataSearch(session)

    def search(self, query: SearchQuery, mode: SearchMode = "fulltext") -> SearchResultPage:
        if mode == "fulltext":
            return self._fulltext.search(query)
        if mode == "bm25":
            return self._bm25.search(query)
        if mode == "metadata":
            return self._metadata.search(query)
        raise ValueError(f"Unknown search mode: {mode}")
