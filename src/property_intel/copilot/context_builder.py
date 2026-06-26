"""Context builder: formats retrieved chunks into an LLM prompt window.

Converts a list of ScoredChunks into:
  - A formatted context string with numbered citation markers [1], [2], …
  - A parallel list of Citation objects the caller can attach to the response

Token counting uses tiktoken (same tokenizer OpenAI uses) so we never exceed
the model's context budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from property_intel.retrieval.models import ScoredChunk

_DEFAULT_ENCODING = "cl100k_base"  # encoding used by gpt-4o / gpt-4o-mini


@dataclass(frozen=True)
class Citation:
    index: int          # 1-based, matches [N] in the context string
    chunk_id: int
    document_id: int
    section_title: str | None
    content_snippet: str  # first 200 chars for display
    document_title: str | None = None


@dataclass(frozen=True)
class BuiltContext:
    context: str            # formatted block to insert into the prompt
    citations: list[Citation]
    token_count: int


def _count_tokens(text: str, encoding_name: str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


class ContextBuilder:
    """Formats ScoredChunks into a numbered context block with citations.

    Args:
        max_context_tokens: Maximum tokens allowed for the full context block.
        encoding_name:      tiktoken encoding — must match the target model.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self._encoding_name = encoding_name

    def build(self, chunks: list[ScoredChunk]) -> BuiltContext:
        """Format chunks into a context string, stopping when token budget runs out.

        Returns the context string, citations list, and actual token count.
        Empty input returns empty context.
        """
        if not chunks:
            return BuiltContext(context="", citations=[], token_count=0)

        parts: list[str] = []
        citations: list[Citation] = []
        total_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            section = chunk.section_title or "—"
            if chunk.document_title:
                header = f"[{idx}] Document: {chunk.document_title} | Section: {section}"
            else:
                header = f"[{idx}] Section: {section}"
            block = f"{header}\n{chunk.content}"
            block_tokens = _count_tokens(block, self._encoding_name)

            if total_tokens + block_tokens > self.max_context_tokens:
                break

            parts.append(block)
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    section_title=chunk.section_title,
                    content_snippet=chunk.content[:200],
                )
            )
            total_tokens += block_tokens

        context = "\n\n".join(parts)
        return BuiltContext(context=context, citations=citations, token_count=total_tokens)
