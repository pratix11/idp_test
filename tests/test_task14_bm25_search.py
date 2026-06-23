from datetime import date

import pytest

from property_intel.db.repository import DocumentRepository
from property_intel.registry.state_machine import DocumentState
from property_intel.search.bm25 import BM25Index, BM25Search
from property_intel.search.schema import SearchFilters, SearchQuery

pytestmark = pytest.mark.db


def _make_document(
    repo, tmp_path, suffix, content, category="circulars", state=DocumentState.COMPLETED
):
    pdf_path = tmp_path / f"doc{suffix}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    doc = repo.create(
        title=f"Document {suffix}",
        source="maharera",
        category=category,
        document_type=category,
        date=date(2024, 1, 1),
        pages=1,
        file_path=str(pdf_path),
        content_hash=f"hash-{suffix}",
        content=content,
    )
    if state != DocumentState.UPLOADED:
        repo.update_state(doc.id, DocumentState.PROCESSING)
        repo.update_state(doc.id, state)
    return repo.get_by_id(doc.id)


def test_relevant_document_ranks_above_irrelevant(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    relevant = _make_document(
        repo, tmp_path, "rel", "registration deadline extension for promoters " * 3
    )
    _make_document(repo, tmp_path, "irrel", "parking allotment rules for residential buildings")

    search = BM25Search(repo)
    page = search.search(SearchQuery(text="registration deadline"))

    assert page.total == 1
    assert page.items[0].document_id == relevant.id


def test_excludes_documents_not_in_completed_state(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    _make_document(
        repo, tmp_path, "uploaded", "registration deadline extension", state=DocumentState.UPLOADED
    )

    search = BM25Search(repo)
    page = search.search(SearchQuery(text="registration"))

    assert page.total == 0


def test_filters_narrow_results(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    circular = _make_document(
        repo, tmp_path, "circ", "amendment to registration rules", category="circulars"
    )
    _make_document(repo, tmp_path, "act", "amendment to registration rules", category="acts")

    search = BM25Search(repo)
    page = search.search(SearchQuery(text="amendment", filters=SearchFilters(category="circulars")))

    assert page.total == 1
    assert page.items[0].document_id == circular.id


def test_empty_query_text_returns_empty_page(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    _make_document(repo, tmp_path, "anydoc", "some content here")

    search = BM25Search(repo)
    page = search.search(SearchQuery(text=None))

    assert page.total == 0
    assert page.items == []


def test_empty_corpus_returns_empty_page(db_session) -> None:
    repo = DocumentRepository(db_session)

    search = BM25Search(repo)
    page = search.search(SearchQuery(text="registration"))

    assert page.total == 0
    assert page.items == []


def test_bm25_index_rank_with_no_documents() -> None:
    index = BM25Index([])

    assert index.rank("anything") == []
