from datetime import date

import pytest
from sqlalchemy import func, select

from property_intel.db.models import DocumentModel
from property_intel.db.repository import DocumentRepository

pytestmark = pytest.mark.db


def _make_fields(tmp_path, suffix: str, content: str | None) -> dict:
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
        content=content,
    )


def test_content_persisted_on_create(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(**_make_fields(tmp_path, "content", "extension of registration deadline"))

    assert document.content == "extension of registration deadline"


def test_content_persisted_on_update(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(**_make_fields(tmp_path, "update", None))

    updated = repo.update(document.id, content="amendment to development rules")

    assert updated.content == "amendment to development rules"


def test_search_vector_generated_from_title_and_content(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(
        **_make_fields(tmp_path, "vector", "extension of registration deadline")
    )

    stmt = select(DocumentModel).where(
        DocumentModel.id == document.id,
        DocumentModel.search_vector.op("@@")(func.to_tsquery("english", "extension")),
    )
    found = db_session.scalars(stmt).first()

    assert found is not None
    assert found.id == document.id


def test_search_vector_does_not_match_unrelated_terms(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(
        **_make_fields(tmp_path, "novector", "extension of registration deadline")
    )

    stmt = select(DocumentModel).where(
        DocumentModel.id == document.id,
        DocumentModel.search_vector.op("@@")(func.to_tsquery("english", "earthquake")),
    )
    found = db_session.scalars(stmt).first()

    assert found is None


def test_search_vector_matches_on_title_when_content_is_null(db_session, tmp_path) -> None:
    repo = DocumentRepository(db_session)
    document = repo.create(**_make_fields(tmp_path, "titleonly", None))

    stmt = select(DocumentModel).where(
        DocumentModel.id == document.id,
        DocumentModel.search_vector.op("@@")(func.to_tsquery("english", "circular")),
    )
    found = db_session.scalars(stmt).first()

    assert found is not None
