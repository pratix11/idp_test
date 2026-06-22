from pathlib import Path
from typing import Any

import pytest

from property_intel.parsing.base import (
    CorruptedFileError,
    DocumentParser,
    ParsedDocument,
    ParsedTable,
    ParserError,
    UnsupportedFileTypeError,
)


def test_document_parser_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        DocumentParser()  # type: ignore[abstract]


class _DummyParser(DocumentParser):
    def parse_document(self, path: Path) -> ParsedDocument:
        return ParsedDocument(markdown="# hi", text="hi", parser_used="dummy")

    def extract_text(self, path: Path) -> str:
        return "hi"

    def extract_tables(self, path: Path) -> list[ParsedTable]:
        return []

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        return {}


def test_dummy_subclass_can_be_instantiated() -> None:
    parser = _DummyParser()
    result = parser.parse_document(Path("whatever.pdf"))
    assert result.markdown == "# hi"
    assert result.parser_used == "dummy"


def test_supports_default_extensions() -> None:
    parser = _DummyParser()
    assert parser.supports(Path("doc.pdf")) is True
    assert parser.supports(Path("doc.PDF")) is True
    assert parser.supports(Path("doc.docx")) is False


def test_parsed_document_defaults() -> None:
    doc = ParsedDocument(markdown="", text="")
    assert doc.tables == []
    assert doc.image_count == 0
    assert doc.page_count == 0
    assert doc.metadata == {}
    assert doc.parser_used == ""


def test_exception_hierarchy() -> None:
    assert issubclass(CorruptedFileError, ParserError)
    assert issubclass(UnsupportedFileTypeError, ParserError)


def test_parsed_table_holds_rows() -> None:
    table = ParsedTable(rows=[["a", "b"], ["1", "2"]], page_number=2)
    assert table.rows[0] == ["a", "b"]
    assert table.page_number == 2
