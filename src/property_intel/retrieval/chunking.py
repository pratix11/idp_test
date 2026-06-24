import re
from dataclasses import dataclass

from property_intel.retrieval.models import DocumentChunk

_CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class _Section:
    title: str | None
    body: str


class MarkdownChunker:
    """Split markdown into overlapping chunks sized for vector retrieval.

    Strategy (hierarchical):
      1. Split on markdown headers — each section stays together if it fits.
      2. If a section exceeds chunk_size, split on paragraph breaks (blank lines).
      3. If a single paragraph is still too long, slice character-by-character
         with overlap so no sentence is silently dropped at a boundary.

    Token counting uses a fast approximation (1 token ≈ 4 chars) so the
    chunker stays dependency-free — the real tokeniser is loaded only when
    generating embeddings (Task 19).
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._max_chars = chunk_size * _CHARS_PER_TOKEN
        self._overlap_chars = overlap * _CHARS_PER_TOKEN

    def chunk_document(self, markdown: str, document_id: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in self._split_by_headers(markdown):
            text = section.body.strip()
            if not text:
                continue
            for piece in self._split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=len(chunks),
                        content=piece,
                        token_count=_count_tokens(piece),
                        section_title=section.title,
                    )
                )
        return chunks

    # ── private helpers ────────────────────────────────────────────────────

    def _split_by_headers(self, markdown: str) -> list[_Section]:
        header_re = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
        sections: list[_Section] = []
        last_end = 0
        current_title: str | None = None

        for match in header_re.finditer(markdown):
            body = markdown[last_end : match.start()]
            if body.strip() or current_title is not None:
                sections.append(_Section(title=current_title, body=body))
            current_title = match.group(1).strip()
            last_end = match.end()

        tail = markdown[last_end:]
        if tail.strip() or current_title is not None:
            sections.append(_Section(title=current_title, body=tail))

        return sections

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]

        paragraphs = re.split(r"\n\n+", text)
        result: list[str] = []
        current = ""

        for para in paragraphs:
            sep = 2 if current else 0
            if len(current) + sep + len(para) <= self._max_chars:
                current = (current + "\n\n" + para) if current else para
            else:
                if current:
                    result.append(current)
                if len(para) <= self._max_chars:
                    current = para
                else:
                    result.extend(self._slice_with_overlap(para))
                    current = ""

        if current:
            result.append(current)

        return result

    def _slice_with_overlap(self, text: str) -> list[str]:
        step = max(1, self._max_chars - self._overlap_chars)
        return [text[i : i + self._max_chars] for i in range(0, len(text), step)]
