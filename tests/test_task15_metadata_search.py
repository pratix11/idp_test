from datetime import date

import pytest

from property_intel.db.repository import DocumentRepository
from property_intel.search.metadata_search import MetadataSearch
from property_intel.search.schema import SearchFilters, SearchQuery

pytestmark = pytest.mark.db


def _make_document(repo, tmp_path, suffix, title, category, source, doc_date):
    pdf_path = tmp_path / f"doc{suffix}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return repo.create(
        title=title,
        source=source,
        category=category,
        document_type=category,
        date=doc_date,
        pages=1,
        file_path=str(pdf_path),
        content_hash=f"hash-{suffix}",
    )


def test_filter_by_title_substring(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    match = _make_document(
        repo, tmp_path, "1", "RERA Registration Circular", "circulars", "maharera", date(2024, 1, 1)
    )
    _make_document(
        repo, tmp_path, "2", "Parking Allotment Order", "orders", "maharera", date(2024, 1, 1)
    )

    search = MetadataSearch(db_session)
    page = search.search(SearchQuery(text="registration"))

    assert page.total == 1
    assert page.items[0].document_id == match.id


def test_filter_by_category(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    circular = _make_document(
        repo, tmp_path, "1", "Doc A", "circulars", "maharera", date(2024, 1, 1)
    )
    _make_document(repo, tmp_path, "2", "Doc B", "acts", "maharera", date(2024, 1, 1))

    search = MetadataSearch(db_session)
    page = search.search(SearchQuery(filters=SearchFilters(category="circulars")))

    assert page.total == 1
    assert page.items[0].document_id == circular.id


def test_filter_by_source(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    maharera_doc = _make_document(
        repo, tmp_path, "1", "Doc A", "circulars", "maharera", date(2024, 1, 1)
    )
    _make_document(repo, tmp_path, "2", "Doc B", "circulars", "mhada", date(2024, 1, 1))

    search = MetadataSearch(db_session)
    page = search.search(SearchQuery(filters=SearchFilters(source="maharera")))

    assert page.total == 1
    assert page.items[0].document_id == maharera_doc.id


def test_filter_by_document_type(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    circular = _make_document(
        repo, tmp_path, "1", "Doc A", "circulars", "maharera", date(2024, 1, 1)
    )
    _make_document(repo, tmp_path, "2", "Doc B", "acts", "maharera", date(2024, 1, 1))

    search = MetadataSearch(db_session)
    page = search.search(SearchQuery(filters=SearchFilters(document_type="circulars")))

    assert page.total == 1
    assert page.items[0].document_id == circular.id


def test_filter_by_date_range(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    in_range = _make_document(
        repo, tmp_path, "1", "Doc A", "circulars", "maharera", date(2024, 3, 15)
    )
    _make_document(repo, tmp_path, "2", "Doc B", "circulars", "maharera", date(2023, 1, 1))
    _make_document(repo, tmp_path, "3", "Doc C", "circulars", "maharera", date(2025, 1, 1))

    search = MetadataSearch(db_session)
    page = search.search(
        SearchQuery(filters=SearchFilters(date_from=date(2024, 1, 1), date_to=date(2024, 12, 31)))
    )

    assert page.total == 1
    assert page.items[0].document_id == in_range.id


def test_combined_filters(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    match = _make_document(
        repo, tmp_path, "1", "RERA Registration Circular", "circulars", "maharera", date(2024, 1, 1)
    )
    _make_document(
        repo, tmp_path, "2", "RERA Registration Order", "orders", "maharera", date(2024, 1, 1)
    )
    _make_document(
        repo, tmp_path, "3", "RERA Registration Circular", "circulars", "mhada", date(2024, 1, 1)
    )

    search = MetadataSearch(db_session)
    page = search.search(
        SearchQuery(
            text="registration",
            filters=SearchFilters(category="circulars", source="maharera"),
        )
    )

    assert page.total == 1
    assert page.items[0].document_id == match.id


def test_pagination(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    ids = set()
    for i in range(5):
        doc = _make_document(
            repo, tmp_path, f"p{i}", f"Doc {i}", "circulars", "maharera", date(2024, 1, 1 + i)
        )
        ids.add(doc.id)

    search = MetadataSearch(db_session)
    page1 = search.search(SearchQuery(page=1, page_size=2))
    page2 = search.search(SearchQuery(page=2, page_size=2))
    page3 = search.search(SearchQuery(page=3, page_size=2))

    assert page1.total == 5
    assert page1.total_pages == 3
    assert len(page1.items) == 2
    assert len(page2.items) == 2
    assert len(page3.items) == 1
    seen_ids = {item.document_id for item in page1.items + page2.items + page3.items}
    assert seen_ids == ids


def test_empty_results(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    _make_document(repo, tmp_path, "1", "Doc A", "circulars", "maharera", date(2024, 1, 1))

    search = MetadataSearch(db_session)
    page = search.search(SearchQuery(filters=SearchFilters(source="cidco")))

    assert page.total == 0
    assert page.items == []
