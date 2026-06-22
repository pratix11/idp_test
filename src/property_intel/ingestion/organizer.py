from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from property_intel.hashing import hash_file
from property_intel.logging_setup import get_logger
from property_intel.metadata.schema import DocumentCategory

logger = get_logger("ingestion.organizer")


class UnknownCategoryError(Exception):
    pass


class MissingDocumentsError(Exception):
    pass


@dataclass(frozen=True)
class ScannedDocument:
    source: str
    category: DocumentCategory
    file_path: Path


class DatasetOrganizer:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)

    def _categorize_path(self, file_path: Path) -> ScannedDocument:
        relative = file_path.resolve().relative_to(self.raw_dir.resolve())
        parts = relative.parts
        if len(parts) < 3:
            raise UnknownCategoryError(
                f"Cannot determine source/category for {file_path}: "
                "expected <raw_dir>/<source>/<category>/...file.pdf"
            )
        source, category_name = parts[0], parts[1]
        try:
            category = DocumentCategory(category_name.lower())
        except ValueError as exc:
            raise UnknownCategoryError(
                f"Unknown category '{category_name}' for file {file_path}"
            ) from exc
        return ScannedDocument(source=source, category=category, file_path=file_path)

    def categorize(self, file_path: Path) -> DocumentCategory:
        return self._categorize_path(file_path).category

    def scan(self) -> list[ScannedDocument]:
        documents: list[ScannedDocument] = []
        for pdf_path in sorted(self.raw_dir.rglob("*.pdf")):
            try:
                documents.append(self._categorize_path(pdf_path))
            except UnknownCategoryError:
                logger.warning("Skipping uncategorizable file: %s", pdf_path)
        return documents

    def verify_no_missing(self) -> None:
        all_pdfs = set(self.raw_dir.rglob("*.pdf"))
        scanned_paths = {doc.file_path for doc in self.scan()}
        missing = all_pdfs - scanned_paths
        if missing:
            raise MissingDocumentsError(
                f"{len(missing)} file(s) could not be categorized: {sorted(missing)}"
            )

    def find_duplicates(
        self, documents: list[ScannedDocument] | None = None
    ) -> dict[str, list[Path]]:
        if documents is None:
            documents = self.scan()
        hash_to_paths: dict[str, list[Path]] = defaultdict(list)
        for doc in documents:
            content_hash = hash_file(doc.file_path)
            hash_to_paths[content_hash].append(doc.file_path)
        return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
