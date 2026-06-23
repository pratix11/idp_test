import pytest

from property_intel.search.schema import SearchQuery, SearchResultPage
from property_intel.search.service import SearchService

pytestmark = pytest.mark.db


class _StubBackend:
    def __init__(self) -> None:
        self.received_query: SearchQuery | None = None

    def search(self, query: SearchQuery) -> SearchResultPage:
        self.received_query = query
        return SearchResultPage(items=[], total=0, page=query.page, page_size=query.page_size)


def _service(db_session) -> tuple[SearchService, _StubBackend, _StubBackend, _StubBackend]:
    fulltext, bm25, metadata = _StubBackend(), _StubBackend(), _StubBackend()
    service = SearchService(db_session, fulltext=fulltext, bm25=bm25, metadata=metadata)
    return service, fulltext, bm25, metadata


def test_default_mode_dispatches_to_fulltext(db_session) -> None:
    service, fulltext, bm25, metadata = _service(db_session)

    service.search(SearchQuery(text="x"))

    assert fulltext.received_query is not None
    assert bm25.received_query is None
    assert metadata.received_query is None


def test_bm25_mode_dispatches_to_bm25(db_session) -> None:
    service, fulltext, bm25, metadata = _service(db_session)

    service.search(SearchQuery(text="x"), mode="bm25")

    assert bm25.received_query is not None
    assert fulltext.received_query is None
    assert metadata.received_query is None


def test_metadata_mode_dispatches_to_metadata(db_session) -> None:
    service, fulltext, bm25, metadata = _service(db_session)

    service.search(SearchQuery(), mode="metadata")

    assert metadata.received_query is not None
    assert fulltext.received_query is None
    assert bm25.received_query is None


def test_unknown_mode_raises(db_session) -> None:
    service, _, _, _ = _service(db_session)

    with pytest.raises(ValueError):
        service.search(SearchQuery(text="x"), mode="bogus")  # type: ignore[arg-type]
