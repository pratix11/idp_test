from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from property_intel.hashing import hash_file
from property_intel.ingestion.organizer import DatasetOrganizer, ScannedDocument
from property_intel.logging_setup import get_logger
from property_intel.parsing.base import ParserError
from property_intel.parsing.router import ParserRouter
from property_intel.registry.state_machine import DocumentState

logger = get_logger("pipeline.batch_processor")


class DocumentRecord(Protocol):
    id: int
    state: str
    file_path: str
    category: str
    error_message: str | None


class Repository(Protocol):
    def get_by_content_hash(self, content_hash: str) -> DocumentRecord | None: ...
    def create(self, **fields: object) -> DocumentRecord: ...
    def update(self, document_id: int, **fields: object) -> DocumentRecord: ...
    def update_state(
        self, document_id: int, target: DocumentState, error_message: str | None = None
    ) -> DocumentRecord: ...
    def list_all(self) -> Sequence[DocumentRecord]: ...


@dataclass
class BatchSummary:
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[tuple[Path, str]] = field(default_factory=list)


class DocumentProcessor:
    def __init__(self, repository: Repository, router: ParserRouter, processed_dir: Path) -> None:
        self.repository = repository
        self.router = router
        self.processed_dir = Path(processed_dir)

    def _write_markdown(self, category: str, stem: str, markdown: str) -> Path:
        markdown_path = self.processed_dir / category / f"{stem}.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        return markdown_path

    def process_one(self, scanned: ScannedDocument) -> str:
        content_hash = hash_file(scanned.file_path)
        existing = self.repository.get_by_content_hash(content_hash)

        if existing is not None and existing.state == DocumentState.COMPLETED.value:
            logger.info("Skipping already-completed document: %s", scanned.file_path)
            return "skipped"

        if existing is None:
            document = self.repository.create(
                title=scanned.file_path.stem,
                source=scanned.source,
                category=scanned.category.value,
                document_type=scanned.category.value,
                date=datetime.fromtimestamp(scanned.file_path.stat().st_mtime).date(),
                pages=0,
                file_path=str(scanned.file_path),
                content_hash=content_hash,
            )
        else:
            document = existing

        self.repository.update_state(document.id, DocumentState.PROCESSING)
        return self._parse_and_finalize(document.id, scanned.category.value, scanned.file_path)

    def _parse_and_finalize(self, document_id: int, category: str, file_path: Path) -> str:
        try:
            parsed = self.router.parse(file_path)
        except ParserError as exc:
            logger.error("Failed to process %s: %s", file_path, exc)
            self.repository.update_state(document_id, DocumentState.FAILED, error_message=str(exc))
            return "failed"

        markdown_path = self._write_markdown(category, file_path.stem, parsed.markdown)
        self.repository.update(document_id, pages=parsed.page_count, markdown_path=str(markdown_path))
        self.repository.update_state(document_id, DocumentState.COMPLETED)
        return "completed"


class BatchProcessor:
    def __init__(self, organizer: DatasetOrganizer, processor: DocumentProcessor) -> None:
        self.organizer = organizer
        self.processor = processor

    def run(self) -> BatchSummary:
        summary = BatchSummary()
        for scanned in self.organizer.scan():
            summary.total += 1
            try:
                status = self.processor.process_one(scanned)
            except Exception as exc:
                logger.exception("Unexpected error processing %s", scanned.file_path)
                summary.failed += 1
                summary.errors.append((scanned.file_path, str(exc)))
                continue

            if status == "completed":
                summary.completed += 1
            elif status == "failed":
                summary.failed += 1
            elif status == "skipped":
                summary.skipped += 1
        return summary

    def retry_failed(self) -> BatchSummary:
        summary = BatchSummary()
        failed_documents = [
            doc
            for doc in self.processor.repository.list_all()
            if doc.state == DocumentState.FAILED.value
        ]

        for doc in failed_documents:
            summary.total += 1
            self.processor.repository.update_state(doc.id, DocumentState.PROCESSING)
            file_path = Path(doc.file_path)
            status = self.processor._parse_and_finalize(doc.id, doc.category, file_path)

            if status == "completed":
                summary.completed += 1
            else:
                summary.failed += 1
                summary.errors.append((file_path, doc.error_message or "retry failed"))

        return summary
