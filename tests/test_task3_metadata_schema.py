from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from property_intel.metadata.schema import DocumentCategory, DocumentMetadata


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    return pdf_path


def _valid_kwargs(sample_pdf):
    return dict(
        title="Circular No. 51 of 2025",
        source="maharera",
        category=DocumentCategory.CIRCULARS,
        document_type="circular",
        date=date(2025, 1, 15),
        pages=3,
        file_path=sample_pdf,
    )


def test_valid_construction(sample_pdf) -> None:
    metadata = DocumentMetadata(**_valid_kwargs(sample_pdf))
    assert metadata.category == DocumentCategory.CIRCULARS
    assert metadata.markdown_path is None


def test_valid_construction_with_markdown_path(sample_pdf, tmp_path) -> None:
    md_path = tmp_path / "sample.md"
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["markdown_path"] = md_path
    metadata = DocumentMetadata(**kwargs)
    assert metadata.markdown_path == md_path


@pytest.mark.parametrize(
    "missing_field",
    ["title", "source", "category", "document_type", "date", "pages", "file_path"],
)
def test_required_field_missing_raises(sample_pdf, missing_field) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


@pytest.mark.parametrize("field", ["title", "source", "document_type"])
def test_blank_string_fields_raise(sample_pdf, field) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_future_date_raises(sample_pdf) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["date"] = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_today_date_is_allowed(sample_pdf) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["date"] = date.today()
    DocumentMetadata(**kwargs)


def test_invalid_date_string_raises(sample_pdf) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["date"] = "not-a-date"
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_nonexistent_file_path_raises(tmp_path) -> None:
    kwargs = _valid_kwargs(tmp_path / "missing.pdf")
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_non_pdf_file_path_raises(tmp_path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello")
    kwargs = _valid_kwargs(txt_path)
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_non_md_markdown_path_raises(sample_pdf, tmp_path) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["markdown_path"] = tmp_path / "sample.txt"
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_invalid_category_raises(sample_pdf) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["category"] = "not-a-real-category"
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_zero_or_negative_pages_raises(sample_pdf) -> None:
    kwargs = _valid_kwargs(sample_pdf)
    kwargs["pages"] = 0
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)

    kwargs["pages"] = -5
    with pytest.raises(ValidationError):
        DocumentMetadata(**kwargs)


def test_all_categories_are_valid(sample_pdf) -> None:
    for category in DocumentCategory:
        kwargs = _valid_kwargs(sample_pdf)
        kwargs["category"] = category
        DocumentMetadata(**kwargs)
