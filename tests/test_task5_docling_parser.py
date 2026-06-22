from pathlib import Path

import pandas as pd
import pytest
from docling.datamodel.base_models import ConversionStatus

from property_intel.config import get_settings
from property_intel.parsing.base import CorruptedFileError, ParserError
from property_intel.parsing.docling_parser import DoclingParser
from tests.fixtures.make_sample_pdfs import make_blank_pdf, make_corrupted_pdf

REAL_SMALL_PDF = (
    "acts/Notification_of_GOI_regarding_commencement_of_Act_26_04_16.pdf"
)


def _real_dataset_path(relative: str) -> Path:
    return get_settings().resolved_data_raw_dir() / "maharera" / relative


# ---------------------------------------------------------------------------
# Fast tests against a mocked Docling converter — exercise our own logic
# (status handling, table-row extraction, exception mapping) without paying
# for a real Docling conversion.
# ---------------------------------------------------------------------------


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no


class _FakeTable:
    def __init__(self, dataframe: pd.DataFrame, page_no: int = 1) -> None:
        self._dataframe = dataframe
        self.prov = [_FakeProv(page_no)]

    def export_to_dataframe(self) -> pd.DataFrame:
        return self._dataframe


class _FakeDocument:
    def __init__(
        self,
        markdown: str = "",
        text: str = "",
        tables: list | None = None,
        pictures: list | None = None,
        pages: int = 1,
    ) -> None:
        self._markdown = markdown
        self._text = text
        self.tables = tables or []
        self.pictures = pictures or []
        self._pages = pages

    def export_to_markdown(self) -> str:
        return self._markdown

    def export_to_text(self) -> str:
        return self._text

    def num_pages(self) -> int:
        return self._pages


class _FakeResult:
    def __init__(self, status: ConversionStatus, document: _FakeDocument | None) -> None:
        self.status = status
        self.document = document


@pytest.fixture(scope="module")
def parser() -> DoclingParser:
    return DoclingParser()


def test_table_rows_extracted_from_dataframe(parser, monkeypatch, tmp_path) -> None:
    dataframe = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": ["30", "40"]})
    fake_doc = _FakeDocument(markdown="# doc", text="doc", tables=[_FakeTable(dataframe, page_no=2)])
    monkeypatch.setattr(
        parser._converter, "convert", lambda *a, **k: _FakeResult(ConversionStatus.SUCCESS, fake_doc)
    )

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    result = parser.parse_document(pdf_path)

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.rows[0] == ["Name", "Age"]
    assert table.rows[1] == ["Alice", "30"]
    assert table.page_number == 2
    assert result.parser_used == "docling"


def test_extract_text_passthrough(parser, monkeypatch, tmp_path) -> None:
    fake_doc = _FakeDocument(markdown="# hi", text="hello world")
    monkeypatch.setattr(
        parser._converter, "convert", lambda *a, **k: _FakeResult(ConversionStatus.SUCCESS, fake_doc)
    )
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    assert parser.extract_text(pdf_path) == "hello world"


def test_extract_metadata_passthrough(parser, monkeypatch, tmp_path) -> None:
    fake_doc = _FakeDocument(markdown="# hi", text="hello")
    monkeypatch.setattr(
        parser._converter, "convert", lambda *a, **k: _FakeResult(ConversionStatus.SUCCESS, fake_doc)
    )
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    metadata = parser.extract_metadata(pdf_path)
    assert metadata["status"] == str(ConversionStatus.SUCCESS)


def test_failure_status_raises_corrupted_file_error(parser, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        parser._converter, "convert", lambda *a, **k: _FakeResult(ConversionStatus.FAILURE, None)
    )
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(CorruptedFileError):
        parser.parse_document(pdf_path)


def test_converter_exception_is_wrapped_in_parser_error(parser, monkeypatch, tmp_path) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(parser._converter, "convert", _raise)
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ParserError):
        parser.parse_document(pdf_path)


def test_missing_file_raises_parser_error(parser, tmp_path) -> None:
    with pytest.raises(ParserError):
        parser.parse_document(tmp_path / "does_not_exist.pdf")


def test_table_dataframe_export_failure_returns_empty_rows(parser, monkeypatch, tmp_path) -> None:
    class _BrokenTable:
        prov: list = []

        def export_to_dataframe(self):
            raise RuntimeError("cannot export")

    fake_doc = _FakeDocument(markdown="# hi", text="hi", tables=[_BrokenTable()])
    monkeypatch.setattr(
        parser._converter, "convert", lambda *a, **k: _FakeResult(ConversionStatus.SUCCESS, fake_doc)
    )
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    result = parser.parse_document(pdf_path)
    assert result.tables[0].rows == []


# ---------------------------------------------------------------------------
# Slow, end-to-end tests against the real Docling engine. A module-scoped
# parser amortizes the one-time model load across this file's slow tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_parser() -> DoclingParser:
    return DoclingParser()


@pytest.mark.slow
def test_markdown_generated_for_real_pdf(real_parser) -> None:
    pdf_path = _real_dataset_path(REAL_SMALL_PDF)
    if not pdf_path.exists():
        pytest.skip("real data/raw dataset not present")

    result = real_parser.parse_document(pdf_path)

    assert result.markdown.strip() != ""
    assert result.page_count >= 1
    assert result.parser_used == "docling"


@pytest.mark.slow
def test_images_detected_in_real_pdf(real_parser) -> None:
    pdf_path = _real_dataset_path(REAL_SMALL_PDF)
    if not pdf_path.exists():
        pytest.skip("real data/raw dataset not present")

    result = real_parser.parse_document(pdf_path)

    assert result.image_count >= 1


@pytest.mark.slow
def test_blank_document_handled_without_raising(real_parser, tmp_path) -> None:
    pdf_path = make_blank_pdf(tmp_path / "blank.pdf")

    result = real_parser.parse_document(pdf_path)

    assert result.text.strip() == ""
    assert result.page_count >= 1


@pytest.mark.slow
def test_corrupted_pdf_raises_corrupted_file_error(real_parser, tmp_path) -> None:
    pdf_path = make_corrupted_pdf(tmp_path / "corrupted.pdf")

    with pytest.raises(CorruptedFileError):
        real_parser.parse_document(pdf_path)
