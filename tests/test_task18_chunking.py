"""Task 18 — Chunking Strategy.

Unit tests cover the MarkdownChunker in isolation (no DB, no markers).
DB tests (marked `db`) verify the ChunkRepository round-trips and cascades.
"""

from datetime import date

import pytest

from property_intel.retrieval.chunking import MarkdownChunker
from property_intel.retrieval.models import DocumentChunk


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_doc(db_session, tmp_path, suffix: str) -> "DocumentModel":  # type: ignore[name-defined]
    from property_intel.db.repository import DocumentRepository

    pdf = tmp_path / f"doc_{suffix}.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return DocumentRepository(db_session).create(
        title=f"Doc {suffix}",
        source="maharera",
        category="circulars",
        document_type="circular",
        date=date(2024, 1, 1),
        pages=2,
        file_path=str(pdf),
        content_hash=f"hash-chunk-{suffix}",
    )


# ── unit tests (pure Python, no DB) ──────────────────────────────────────────


def test_empty_markdown_returns_no_chunks() -> None:
    assert MarkdownChunker().chunk_document("", document_id=1) == []


def test_whitespace_only_returns_no_chunks() -> None:
    assert MarkdownChunker().chunk_document("   \n\n   ", document_id=1) == []


def test_short_document_is_single_chunk() -> None:
    chunks = MarkdownChunker().chunk_document("# Title\n\nShort content.", document_id=1)
    assert len(chunks) == 1
    assert "Short content." in chunks[0].content


def test_section_title_captured() -> None:
    chunks = MarkdownChunker().chunk_document("# My Section\n\nSome text.", document_id=1)
    assert chunks[0].section_title == "My Section"


def test_no_header_gives_none_section_title() -> None:
    chunks = MarkdownChunker().chunk_document("Plain text, no headers.", document_id=1)
    assert chunks[0].section_title is None


def test_multiple_headers_create_separate_section_titles() -> None:
    md = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
    chunks = MarkdownChunker().chunk_document(md, document_id=5)
    titles = [c.section_title for c in chunks]
    assert "Section A" in titles
    assert "Section B" in titles


def test_preamble_before_first_header_has_no_title() -> None:
    md = "Preamble.\n\n# Section\n\nBody."
    chunks = MarkdownChunker().chunk_document(md, document_id=1)
    assert chunks[0].section_title is None
    assert chunks[1].section_title == "Section"


def test_chunk_indices_are_zero_based_sequential() -> None:
    md = "\n\n".join(f"# Section {i}\n\nContent {i}." for i in range(5))
    chunks = MarkdownChunker().chunk_document(md, document_id=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_document_id_propagated_to_all_chunks() -> None:
    md = "# A\n\nText.\n\n# B\n\nMore text."
    chunks = MarkdownChunker().chunk_document(md, document_id=42)
    assert all(c.document_id == 42 for c in chunks)


def test_all_results_are_document_chunk_instances() -> None:
    chunks = MarkdownChunker().chunk_document("# A\n\nText.", document_id=1)
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_token_count_is_positive_and_bounded() -> None:
    chunks = MarkdownChunker().chunk_document("# Title\n\nHello world.", document_id=1)
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.token_count <= len(chunk.content)


def test_long_section_splits_into_multiple_chunks() -> None:
    # ~5000 chars across 10 paragraphs — well above the 512-token (2048-char) limit
    para = "word " * 100  # ~500 chars
    md = "# Big Section\n\n" + "\n\n".join([para] * 10)
    chunks = MarkdownChunker(chunk_size=512, overlap=64).chunk_document(md, document_id=1)
    assert len(chunks) > 1


def test_each_chunk_respects_size_limit() -> None:
    para = "word " * 100
    md = "# Big Section\n\n" + "\n\n".join([para] * 10)
    chunker = MarkdownChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk_document(md, document_id=1)
    # Each chunk's token count must not exceed chunk_size + small rounding
    for chunk in chunks:
        assert chunk.token_count <= chunker.chunk_size + 5


def test_overlap_produces_shared_content_between_adjacent_chunks() -> None:
    # One paragraph of 3000 chars forces character-level splitting with overlap
    long_text = "x" * 3000
    chunker = MarkdownChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk_document(f"# Section\n\n{long_text}", document_id=1)
    assert len(chunks) >= 2
    # The tail of chunk[0] should appear at the start of chunk[1]
    overlap_chars = 64 * 4
    assert chunks[0].content[-overlap_chars:] == chunks[1].content[:overlap_chars]


def test_custom_chunk_size_respected() -> None:
    # Tiny chunk_size=10 (40 chars) forces many splits on any real paragraph
    para = "a" * 200
    chunker = MarkdownChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk_document(f"# S\n\n{para}", document_id=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 10 * 4 + 1  # +1 rounding tolerance


# ── DB tests ──────────────────────────────────────────────────────────────────


@pytest.mark.db
def test_bulk_create_and_retrieve(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository

    doc = _make_doc(db_session, tmp_path, "bulk")
    chunks = MarkdownChunker().chunk_document("# Intro\n\nHello world.", document_id=doc.id)

    repo = ChunkRepository(db_session)
    saved = repo.bulk_create(chunks)

    assert len(saved) == 1
    assert saved[0].id is not None
    retrieved = repo.get_by_document_id(doc.id)
    assert len(retrieved) == 1
    assert retrieved[0].content == chunks[0].content
    assert retrieved[0].section_title == "Intro"


@pytest.mark.db
def test_retrieved_chunks_ordered_by_index(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository

    doc = _make_doc(db_session, tmp_path, "order")
    md = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C."
    chunks = MarkdownChunker().chunk_document(md, document_id=doc.id)

    repo = ChunkRepository(db_session)
    repo.bulk_create(chunks)

    retrieved = repo.get_by_document_id(doc.id)
    indices = [r.chunk_index for r in retrieved]
    assert indices == sorted(indices)


@pytest.mark.db
def test_count_by_document_id(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository

    doc = _make_doc(db_session, tmp_path, "count")
    md = "# A\n\nText A.\n\n# B\n\nText B."
    chunks = MarkdownChunker().chunk_document(md, document_id=doc.id)

    repo = ChunkRepository(db_session)
    repo.bulk_create(chunks)

    assert repo.count_by_document_id(doc.id) == len(chunks)


@pytest.mark.db
def test_delete_by_document_id(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository

    doc = _make_doc(db_session, tmp_path, "del")
    chunks = MarkdownChunker().chunk_document("# X\n\nContent.", document_id=doc.id)

    repo = ChunkRepository(db_session)
    repo.bulk_create(chunks)
    assert repo.count_by_document_id(doc.id) == 1

    repo.delete_by_document_id(doc.id)
    assert repo.count_by_document_id(doc.id) == 0


@pytest.mark.db
def test_cascade_on_document_delete(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository, DocumentRepository

    doc = _make_doc(db_session, tmp_path, "cascade")
    chunks = MarkdownChunker().chunk_document("# Y\n\nText.", document_id=doc.id)

    chunk_repo = ChunkRepository(db_session)
    chunk_repo.bulk_create(chunks)
    assert chunk_repo.count_by_document_id(doc.id) == 1

    DocumentRepository(db_session).delete(doc.id)
    assert chunk_repo.count_by_document_id(doc.id) == 0


@pytest.mark.db
def test_empty_chunks_list_is_a_no_op(db_session, tmp_path) -> None:
    from property_intel.db.repository import ChunkRepository

    doc = _make_doc(db_session, tmp_path, "empty")
    repo = ChunkRepository(db_session)
    saved = repo.bulk_create([])
    assert saved == []
    assert repo.count_by_document_id(doc.id) == 0
