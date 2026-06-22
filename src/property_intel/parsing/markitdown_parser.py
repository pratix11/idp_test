from pathlib import Path
from typing import Any

from markitdown import MarkItDown
from markitdown._exceptions import MarkItDownException

from property_intel.parsing.base import (
    CorruptedFileError,
    DocumentParser,
    ParsedDocument,
    ParsedTable,
    ParserError,
)


class MarkItDownParser(DocumentParser):
    def __init__(self) -> None:
        self._converter = MarkItDown()

    def _convert(self, path: Path) -> Any:
        if not path.exists():
            raise ParserError(f"File does not exist: {path}")
        try:
            return self._converter.convert(path)
        except MarkItDownException as exc:
            raise CorruptedFileError(f"MarkItDown failed to convert {path}: {exc}") from exc
        except Exception as exc:
            raise ParserError(f"MarkItDown raised an unexpected error for {path}: {exc}") from exc

    def parse_document(self, path: Path) -> ParsedDocument:
        result = self._convert(path)
        markdown = result.markdown or ""
        metadata: dict[str, Any] = {}
        if result.title:
            metadata["title"] = result.title

        return ParsedDocument(
            markdown=markdown,
            text=result.text_content or markdown,
            tables=[],
            image_count=0,
            page_count=0,
            metadata=metadata,
            parser_used="markitdown",
        )

    def extract_text(self, path: Path) -> str:
        return self.parse_document(path).text

    def extract_tables(self, path: Path) -> list[ParsedTable]:
        return []

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        return self.parse_document(path).metadata
