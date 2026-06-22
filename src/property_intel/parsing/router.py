from pathlib import Path

from property_intel.logging_setup import get_logger
from property_intel.parsing.base import DocumentParser, ParsedDocument, ParserError

logger = get_logger("parsing.router")


class ParserRouter:
    def __init__(self, primary: DocumentParser, fallback: DocumentParser) -> None:
        self.primary = primary
        self.fallback = fallback

    def parse(self, path: Path) -> ParsedDocument:
        try:
            return self.primary.parse_document(path)
        except ParserError as primary_exc:
            logger.warning("Primary parser failed for %s (%s); falling back", path, primary_exc)
            try:
                return self.fallback.parse_document(path)
            except ParserError as fallback_exc:
                logger.error("Fallback parser also failed for %s", path)
                raise ParserError(f"Both parsers failed for {path}") from fallback_exc
