from datetime import date

import pytest

from property_intel.db.repository import DocumentNotFoundError, DocumentRepository
from property_intel.registry.state_machine import (
    DocumentState,
    DuplicateDocumentError,
    InvalidTransitionError,
)

pytestmark = pytest.mark.db


def _make_fields(tmp_path, suffix: str = "1") -> dict:
    pdf_path = tmp_path / f"doc{suffix}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return dict(
        title=f"Circular {suffix}",
        source="maharera",
        category="circulars",
        document_type="circular",
        date=date(2024, 1, 1),
        pages=3,
        file_path=str(pdf_path),
        content_hash=f"hash-{suffix}",
    )


def test_insert_document(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(**_make_fields(tmp_path, "insert"))

    assert document.id is not None
    assert document.state == DocumentState.UPLOADED.value


def test_retrieve_document_by_id_and_file_path(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    fields = _make_fields(tmp_path, "retrieve")
    created = repo.create(**fields)

    by_id = repo.get_by_id(created.id)
    by_path = repo.get_by_file_path(fields["file_path"])

    assert by_id.id == created.id
    assert by_path is not None
    assert by_path.id == created.id


def test_get_by_id_missing_raises(db_session) -> None:
    repo = DocumentRepository(db_session)
    with pytest.raises(DocumentNotFoundError):
        repo.get_by_id(999_999)


def test_get_by_file_path_missing_returns_none(db_session) -> None:
    repo = DocumentRepository(db_session)
    assert repo.get_by_file_path("/does/not/exist.pdf") is None


def test_update_document(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    created = repo.create(**_make_fields(tmp_path, "update"))

    updated = repo.update(created.id, title="New Title", pages=10)

    assert updated.title == "New Title"
    assert updated.pages == 10


def test_update_state_valid_transition(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    created = repo.create(**_make_fields(tmp_path, "state"))

    updated = repo.update_state(created.id, DocumentState.PROCESSING)
    assert updated.state == DocumentState.PROCESSING.value

    completed = repo.update_state(updated.id, DocumentState.COMPLETED)
    assert completed.state == DocumentState.COMPLETED.value


def test_update_state_invalid_transition_raises(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    created = repo.create(**_make_fields(tmp_path, "badstate"))

    with pytest.raises(InvalidTransitionError):
        repo.update_state(created.id, DocumentState.COMPLETED)


def test_update_state_records_error_message_on_failure(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    created = repo.create(**_make_fields(tmp_path, "failmsg"))
    repo.update_state(created.id, DocumentState.PROCESSING)

    failed = repo.update_state(created.id, DocumentState.FAILED, error_message="parser exploded")

    assert failed.state == DocumentState.FAILED.value
    assert failed.error_message == "parser exploded"


def test_delete_document(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    created = repo.create(**_make_fields(tmp_path, "delete"))

    repo.delete(created.id)

    with pytest.raises(DocumentNotFoundError):
        repo.get_by_id(created.id)


def test_duplicate_file_path_rejected(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    fields = _make_fields(tmp_path, "dup")
    repo.create(**fields)

    duplicate_fields = dict(fields)
    duplicate_fields["content_hash"] = "different-hash"
    with pytest.raises(DuplicateDocumentError):
        repo.create(**duplicate_fields)


def test_duplicate_content_hash_rejected(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    fields = _make_fields(tmp_path, "duphash")
    repo.create(**fields)

    other_pdf = tmp_path / "other.pdf"
    other_pdf.write_bytes(b"%PDF-1.4\n")
    duplicate_fields = dict(fields)
    duplicate_fields["file_path"] = str(other_pdf)
    with pytest.raises(DuplicateDocumentError):
        repo.create(**duplicate_fields)


def test_list_all_returns_created_documents(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    repo.create(**_make_fields(tmp_path, "list1"))
    repo.create(**_make_fields(tmp_path, "list2"))

    documents = repo.list_all()
    assert len(documents) >= 2
