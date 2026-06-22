import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


class ParserError(Exception):
    pass


class CorruptedFileError(ParserError):
    pass


class UnsupportedFileTypeError(ParserError):
    pass


@dataclass
class ParsedTable:
    rows: list[list[str]]
    page_number: int | None = None


@dataclass
class ParsedDocument:
    markdown: str
    text: str
    tables: list[ParsedTable] = field(default_factory=list)
    image_count: int = 0
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_used: str = ""


class DocumentParser(abc.ABC):
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".pdf"}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    @abc.abstractmethod
    def parse_document(self, path: Path) -> ParsedDocument: ...

    @abc.abstractmethod
    def extract_text(self, path: Path) -> str: ...

    @abc.abstractmethod
    def extract_tables(self, path: Path) -> list[ParsedTable]: ...

    @abc.abstractmethod
    def extract_metadata(self, path: Path) -> dict[str, Any]: ...
