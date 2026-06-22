from pathlib import Path
from typing import Any

import pytest

from property_intel.config import get_settings
from property_intel.parsing.base import (
    CorruptedFileError,
    DocumentParser,
    ParsedDocument,
    ParsedTable,
    ParserError,
)
from property_intel.parsing.markitdown_parser import MarkItDownParser
from property_intel.parsing.router import ParserRouter

REAL_SMALL_PDF = "acts/Notification_of_GOI_regarding_commencement_of_Act_26_04_16.pdf"


def _real_dataset_path(relative: str) -> Path:
    return get_settings().resolved_data_raw_dir() / "maharera" / relative


# ---------------------------------------------------------------------------
# MarkItDownParser
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def markitdown_parser() -> MarkItDownParser:
    return MarkItDownParser()


def test_conversion_success_on_real_pdf(markitdown_parser) -> None:
    pdf_path = _real_dataset_path(REAL_SMALL_PDF)
    if not pdf_path.exists():
        pytest.skip("real data/raw dataset not present")

    result = markitdown_parser.parse_document(pdf_path)

    assert result.markdown.strip() != ""
    assert result.parser_used == "markitdown"
    assert result.tables == []
    assert result.image_count == 0


def test_output_validation_extract_text_and_metadata(markitdown_parser) -> None:
    pdf_path = _real_dataset_path(REAL_SMALL_PDF)
    if not pdf_path.exists():
        pytest.skip("real data/raw dataset not present")

    text = markitdown_parser.extract_text(pdf_path)
    metadata = markitdown_parser.extract_metadata(pdf_path)
    tables = markitdown_parser.extract_tables(pdf_path)

    assert isinstance(text, str) and text.strip() != ""
    assert isinstance(metadata, dict)
    assert tables == []


def test_truly_unreadable_file_raises_corrupted_file_error(markitdown_parser, tmp_path) -> None:
    import os

    pdf_path = tmp_path / "garbage.pdf"
    pdf_path.write_bytes(os.urandom(2000))

    with pytest.raises(CorruptedFileError):
        markitdown_parser.parse_document(pdf_path)


def test_missing_file_raises_parser_error(markitdown_parser, tmp_path) -> None:
    with pytest.raises(ParserError):
        markitdown_parser.parse_document(tmp_path / "missing.pdf")


# ---------------------------------------------------------------------------
# ParserRouter — fallback wiring
# ---------------------------------------------------------------------------


class _StubParser(DocumentParser):
    def __init__(self, *, fail: bool = False, parser_used: str = "stub") -> None:
        self.fail = fail
        self.parser_used = parser_used
        self.calls: list[Path] = []

    def parse_document(self, path: Path) -> ParsedDocument:
        self.calls.append(path)
        if self.fail:
            raise ParserError(f"{self.parser_used} failed for {path}")
        return ParsedDocument(markdown="# ok", text="ok", parser_used=self.parser_used)

    def extract_text(self, path: Path) -> str:
        return self.parse_document(path).text

    def extract_tables(self, path: Path) -> list[ParsedTable]:
        return []

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        return {}


def test_router_uses_primary_when_it_succeeds(tmp_path) -> None:
    primary = _StubParser(fail=False, parser_used="primary")
    fallback = _StubParser(fail=False, parser_used="fallback")
    router = ParserRouter(primary=primary, fallback=fallback)

    result = router.parse(tmp_path / "x.pdf")

    assert result.parser_used == "primary"
    assert fallback.calls == []


def test_router_falls_back_when_primary_fails(tmp_path) -> None:
    primary = _StubParser(fail=True, parser_used="primary")
    fallback = _StubParser(fail=False, parser_used="fallback")
    router = ParserRouter(primary=primary, fallback=fallback)

    result = router.parse(tmp_path / "x.pdf")

    assert result.parser_used == "fallback"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_router_raises_when_both_parsers_fail(tmp_path) -> None:
    primary = _StubParser(fail=True, parser_used="primary")
    fallback = _StubParser(fail=True, parser_used="fallback")
    router = ParserRouter(primary=primary, fallback=fallback)

    with pytest.raises(ParserError):
        router.parse(tmp_path / "x.pdf")


def test_router_fallback_path_with_real_docling_failure(monkeypatch, tmp_path) -> None:
    from property_intel.parsing.docling_parser import DoclingParser

    docling = DoclingParser()
    monkeypatch.setattr(
        docling,
        "parse_document",
        lambda path: (_ for _ in ()).throw(ParserError("forced docling failure")),
    )
    fallback = _StubParser(fail=False, parser_used="markitdown")
    router = ParserRouter(primary=docling, fallback=fallback)

    result = router.parse(tmp_path / "x.pdf")

    assert result.parser_used == "markitdown"
    assert len(fallback.calls) == 1
