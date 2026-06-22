from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from property_intel.logging_setup import get_logger
from property_intel.parsing.base import (
    CorruptedFileError,
    DocumentParser,
    ParsedDocument,
    ParsedTable,
    ParserError,
)

logger = get_logger("parsing.docling")


class DoclingParser(DocumentParser):
    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def _convert(self, path: Path) -> ConversionResult:
        if not path.exists():
            raise ParserError(f"File does not exist: {path}")
        try:
            result = self._converter.convert(path, raises_on_error=False)
        except Exception as exc:
            raise ParserError(f"Docling raised while converting {path}: {exc}") from exc

        if result.status == ConversionStatus.FAILURE:
            raise CorruptedFileError(f"Docling could not parse {path} (status=FAILURE)")
        return result

    def parse_document(self, path: Path) -> ParsedDocument:
        result = self._convert(path)
        document = result.document

        tables = [
            ParsedTable(rows=self._table_to_rows(table), page_number=self._table_page(table))
            for table in document.tables
        ]

        return ParsedDocument(
            markdown=document.export_to_markdown(),
            text=document.export_to_text(),
            tables=tables,
            image_count=len(document.pictures),
            page_count=document.num_pages(),
            metadata={"status": str(result.status)},
            parser_used="docling",
        )

    def extract_text(self, path: Path) -> str:
        return self.parse_document(path).text

    def extract_tables(self, path: Path) -> list[ParsedTable]:
        return self.parse_document(path).tables

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        return self.parse_document(path).metadata

    @staticmethod
    def _table_to_rows(table: Any) -> list[list[str]]:
        try:
            dataframe = table.export_to_dataframe()
        except Exception:
            logger.warning("Failed to export table to dataframe; returning empty rows")
            return []
        rows = [[str(col) for col in dataframe.columns.tolist()]]
        rows.extend([str(cell) for cell in row] for row in dataframe.values.tolist())
        return rows

    @staticmethod
    def _table_page(table: Any) -> int | None:
        prov = getattr(table, "prov", None)
        if prov:
            return prov[0].page_no
        return None
