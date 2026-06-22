from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from property_intel.ingestion.organizer import DatasetOrganizer
from property_intel.parsing.base import DocumentParser, ParsedDocument, ParsedTable, ParserError
from property_intel.parsing.router import ParserRouter
from property_intel.pipeline.batch_processor import BatchProcessor, DocumentProcessor
from property_intel.registry.state_machine import DocumentState, validate_transition


# ---------------------------------------------------------------------------
# Fakes: in-memory repository + conditional stub parser
# ---------------------------------------------------------------------------


@dataclass
class FakeRecord:
    id: int
    title: str
    source: str
    category: str
    document_type: str
    date: Any
    pages: int
    file_path: str
    content_hash: str
    state: str
    markdown_path: str | None = None
    error_message: str | None = None


class FakeDocumentRepository:
    def __init__(self) -> None:
        self._records: dict[int, FakeRecord] = {}
        self._next_id = 1

    def create(self, **fields: Any) -> FakeRecord:
        record = FakeRecord(
            id=self._next_id, state=DocumentState.UPLOADED.value, **fields
        )
        self._records[self._next_id] = record
        self._next_id += 1
        return record

    def get_by_content_hash(self, content_hash: str) -> FakeRecord | None:
        for record in self._records.values():
            if record.content_hash == content_hash:
                return record
        return None

    def update(self, document_id: int, **fields: Any) -> FakeRecord:
        record = self._records[document_id]
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    def update_state(
        self, document_id: int, target: DocumentState, error_message: str | None = None
    ) -> FakeRecord:
        record = self._records[document_id]
        validate_transition(DocumentState(record.state), target)
        record.state = target.value
        if error_message is not None:
            record.error_message = error_message
        return record

    def list_all(self) -> list[FakeRecord]:
        return list(self._records.values())


class _ConditionalParser(DocumentParser):
    def __init__(self, fail_paths: set[Path] | None = None, parser_used: str = "primary") -> None:
        self.fail_paths = fail_paths or set()
        self.parser_used = parser_used

    def parse_document(self, path: Path) -> ParsedDocument:
        if path in self.fail_paths:
            raise ParserError(f"forced failure for {path}")
        return ParsedDocument(
            markdown=f"# {path.stem}", text=path.stem, page_count=1, parser_used=self.parser_used
        )

    def extract_text(self, path: Path) -> str:
        return self.parse_document(path).text

    def extract_tables(self, path: Path) -> list[ParsedTable]:
        return []

    def extract_metadata(self, path: Path) -> dict[str, Any]:
        return {}


def _make_pdf(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_process_one_success(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    pdf_path = _make_pdf(raw_dir / "maharera" / "circulars" / "doc1.pdf", b"%PDF-1.4\ncontent-1\n")
    organizer = DatasetOrganizer(raw_dir)
    scanned = organizer.scan()[0]

    repo = FakeDocumentRepository()
    router = ParserRouter(primary=_ConditionalParser(), fallback=_ConditionalParser())
    processed_dir = tmp_path / "processed"
    processor = DocumentProcessor(repo, router, processed_dir)

    status = processor.process_one(scanned)

    assert status == "completed"
    record = repo.list_all()[0]
    assert record.state == DocumentState.COMPLETED.value
    assert (processed_dir / "circulars" / "doc1.md").exists()
    assert pdf_path.exists()


def test_process_one_failure_does_not_raise(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _make_pdf(raw_dir / "maharera" / "circulars" / "bad.pdf", b"%PDF-1.4\nbad\n")
    organizer = DatasetOrganizer(raw_dir)
    scanned = organizer.scan()[0]

    repo = FakeDocumentRepository()
    failing_path = {scanned.file_path}
    router = ParserRouter(
        primary=_ConditionalParser(fail_paths=failing_path, parser_used="primary"),
        fallback=_ConditionalParser(fail_paths=failing_path, parser_used="fallback"),
    )
    processor = DocumentProcessor(repo, router, tmp_path / "processed")

    status = processor.process_one(scanned)

    assert status == "failed"
    record = repo.list_all()[0]
    assert record.state == DocumentState.FAILED.value
    assert record.error_message is not None


def test_batch_processing_multiple_files(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    for i in range(3):
        _make_pdf(raw_dir / "maharera" / "acts" / f"act{i}.pdf", f"%PDF-1.4\ncontent-{i}\n".encode())
    organizer = DatasetOrganizer(raw_dir)

    repo = FakeDocumentRepository()
    router = ParserRouter(primary=_ConditionalParser(), fallback=_ConditionalParser())
    processor = DocumentProcessor(repo, router, tmp_path / "processed")
    batch = BatchProcessor(organizer, processor)

    summary = batch.run()

    assert summary.total == 3
    assert summary.completed == 3
    assert summary.failed == 0


def test_failure_isolation_one_bad_file_does_not_block_others(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    good1 = _make_pdf(raw_dir / "maharera" / "acts" / "good1.pdf", b"%PDF-1.4\ngood-1\n")
    bad = _make_pdf(raw_dir / "maharera" / "acts" / "bad.pdf", b"%PDF-1.4\nbad\n")
    good2 = _make_pdf(raw_dir / "maharera" / "acts" / "good2.pdf", b"%PDF-1.4\ngood-2\n")
    organizer = DatasetOrganizer(raw_dir)

    repo = FakeDocumentRepository()
    failing_path = {bad}
    router = ParserRouter(
        primary=_ConditionalParser(fail_paths=failing_path),
        fallback=_ConditionalParser(fail_paths=failing_path),
    )
    processor = DocumentProcessor(repo, router, tmp_path / "processed")
    batch = BatchProcessor(organizer, processor)

    summary = batch.run()

    assert summary.total == 3
    assert summary.completed == 2
    assert summary.failed == 1

    states = {Path(r.file_path): r.state for r in repo.list_all()}
    assert states[good1] == DocumentState.COMPLETED.value
    assert states[good2] == DocumentState.COMPLETED.value
    assert states[bad] == DocumentState.FAILED.value


def test_retry_failed_recovers_after_parser_fixed(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    pdf_path = _make_pdf(raw_dir / "maharera" / "rules" / "flaky.pdf", b"%PDF-1.4\nflaky\n")
    organizer = DatasetOrganizer(raw_dir)

    repo = FakeDocumentRepository()
    processed_dir = tmp_path / "processed"
    failing_router = ParserRouter(
        primary=_ConditionalParser(fail_paths={pdf_path}),
        fallback=_ConditionalParser(fail_paths={pdf_path}),
    )
    processor = DocumentProcessor(repo, failing_router, processed_dir)
    batch = BatchProcessor(organizer, processor)

    first_summary = batch.run()
    assert first_summary.failed == 1

    processor.router = ParserRouter(primary=_ConditionalParser(), fallback=_ConditionalParser())
    retry_summary = batch.retry_failed()

    assert retry_summary.total == 1
    assert retry_summary.completed == 1
    record = repo.list_all()[0]
    assert record.state == DocumentState.COMPLETED.value


def test_duplicate_skip_on_rerun(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _make_pdf(raw_dir / "maharera" / "regulations" / "doc.pdf", b"%PDF-1.4\nstable-content\n")
    organizer = DatasetOrganizer(raw_dir)

    repo = FakeDocumentRepository()
    router = ParserRouter(primary=_ConditionalParser(), fallback=_ConditionalParser())
    processor = DocumentProcessor(repo, router, tmp_path / "processed")
    batch = BatchProcessor(organizer, processor)

    first_summary = batch.run()
    assert first_summary.completed == 1
    assert first_summary.skipped == 0

    second_summary = batch.run()
    assert second_summary.completed == 0
    assert second_summary.skipped == 1
    assert len(repo.list_all()) == 1
