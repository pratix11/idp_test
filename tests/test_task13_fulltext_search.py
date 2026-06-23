from datetime import date

import pytest

from property_intel.db.repository import DocumentRepository
from property_intel.search.fulltext import FullTextSearch
from property_intel.search.schema import SearchFilters, SearchQuery

pytestmark = pytest.mark.db


def _make_document(repo, tmp_path, suffix, content, category="circulars", doc_date=date(2024, 1, 1)):
    pdf_path = tmp_path / f"doc{suffix}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return repo.create(
        title=f"Document {suffix}",
        source="maharera",
        category=category,
        document_type=category,
        date=doc_date,
        pages=1,
        file_path=str(pdf_path),
        content_hash=f"hash-{suffix}",
        content=content,
    )


def test_relevant_term_ranks_above_irrelevant(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    relevant = _make_document(repo, tmp_path, "rel", "extension of registration deadline for promoters")
    _make_document(repo, tmp_path, "irrel", "parking allotment rules for residential buildings")

    search = FullTextSearch(db_session)
    page = search.search(SearchQuery(text="registration"))

    assert page.total == 1
    assert page.items[0].document_id == relevant.id
    assert page.items[0].score is not None and page.items[0].score > 0


def test_filters_narrow_results(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    circular = _make_document(repo, tmp_path, "circ", "amendment to registration rules", category="circulars")
    _make_document(repo, tmp_path, "act", "amendment to registration rules", category="acts")

    search = FullTextSearch(db_session)
    page = search.search(
        SearchQuery(text="amendment", filters=SearchFilters(category="circulars"))
    )

    assert page.total == 1
    assert page.items[0].document_id == circular.id


def test_pagination_across_pages(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    ids = set()
    for i in range(5):
        doc = _make_document(repo, tmp_path, f"page{i}", "circular regarding registration process")
        ids.add(doc.id)

    search = FullTextSearch(db_session)
    page1 = search.search(SearchQuery(text="circular", page=1, page_size=2))
    page2 = search.search(SearchQuery(text="circular", page=2, page_size=2))
    page3 = search.search(SearchQuery(text="circular", page=3, page_size=2))

    assert page1.total == 5
    assert page1.total_pages == 3
    assert len(page1.items) == 2
    assert len(page2.items) == 2
    assert len(page3.items) == 1

    seen_ids = {item.document_id for item in page1.items + page2.items + page3.items}
    assert seen_ids == ids


def test_empty_query_text_returns_empty_page(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    _make_document(repo, tmp_path, "anydoc", "some content here")

    search = FullTextSearch(db_session)
    page = search.search(SearchQuery(text=None))

    assert page.total == 0
    assert page.items == []


def test_no_match_returns_empty_page(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    _make_document(repo, tmp_path, "anydoc2", "registration deadline extension")

    search = FullTextSearch(db_session)
    page = search.search(SearchQuery(text="earthquake"))

    assert page.total == 0
    assert page.items == []
