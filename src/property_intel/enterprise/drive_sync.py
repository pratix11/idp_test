"""Drive Sync — Phase 7 enterprise feature.

DriveSyncService monitors a configured folder source (e.g. Google Drive)
and tracks which files have already been ingested.  Each sync run returns
only new files that haven't been seen before, enabling idempotent, repeated
syncs without duplicate processing.

This module is deliberately I/O-free: file listing is injected via a
callable so it's trivially testable without real network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class DriveFile:
    """Metadata about a file discovered in the remote source.

    Attributes:
        file_id:   Stable unique identifier in the remote system.
        name:      Display name / filename.
        mime_type: MIME type (e.g. "application/pdf").
        modified:  Last-modified time reported by the remote (UTC).
        metadata:  Any extra fields (size, owner, URL, etc.).
    """

    file_id: str
    name: str
    mime_type: str = "application/pdf"
    modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncRecord:
    """A record of a successfully ingested file.

    Attributes:
        file_id:   The remote file's stable ID.
        name:      Filename at ingestion time.
        synced_at: UTC timestamp of when the sync occurred.
    """

    file_id: str
    name: str
    synced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DriveSyncConfig:
    """Configuration for a DriveSyncService.

    Attributes:
        source_id:  Identifier for the remote folder/drive/bucket.
        mime_types: Whitelist of MIME types to accept (empty = accept all).
        label:      Human-readable name for this sync source.
    """

    source_id: str
    mime_types: list[str] = field(default_factory=lambda: ["application/pdf"])
    label: str = ""


@dataclass
class SyncResult:
    """Outcome of a single sync run.

    Attributes:
        new_files:    Files discovered for the first time this run.
        skipped:      Files already recorded (not re-processed).
        total_seen:   Total files returned by the listing call.
    """

    new_files: list[DriveFile]
    skipped: int
    total_seen: int

    @property
    def new_count(self) -> int:
        return len(self.new_files)


# Type alias for the injected listing function
FileLister = Callable[[str], list[DriveFile]]


class DriveSyncService:
    """Tracks remote files and returns only new ones each sync run.

    Usage::

        config = DriveSyncConfig(source_id="my-drive-folder-id")
        service = DriveSyncService(config, lister=my_list_function)

        result = service.sync()
        for f in result.new_files:
            ingest(f)          # your ingestion pipeline
            service.mark_synced(f)   # record as done

    The ``lister`` callable receives ``source_id`` and returns
    ``list[DriveFile]``.  Inject a mock in tests; wire a real Google Drive
    API client in production.
    """

    def __init__(
        self,
        config: DriveSyncConfig,
        lister: FileLister | None = None,
    ) -> None:
        self._config = config
        self._lister: FileLister = lister if lister is not None else self._noop_lister
        self._records: dict[str, SyncRecord] = {}

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> DriveSyncConfig:
        return self._config

    @property
    def records(self) -> dict[str, SyncRecord]:
        return dict(self._records)

    def sync(self) -> SyncResult:
        """List remote files and return those not yet recorded."""
        all_files = self._lister(self._config.source_id)
        filtered = self._apply_mime_filter(all_files)
        new_files = [f for f in filtered if f.file_id not in self._records]
        skipped = len(filtered) - len(new_files)
        return SyncResult(
            new_files=new_files,
            skipped=skipped,
            total_seen=len(all_files),
        )

    def mark_synced(self, file: DriveFile) -> SyncRecord:
        """Record *file* as successfully ingested."""
        record = SyncRecord(file_id=file.file_id, name=file.name)
        self._records[file.file_id] = record
        return record

    def is_synced(self, file_id: str) -> bool:
        """Return True if the file has already been recorded."""
        return file_id in self._records

    def clear_records(self) -> None:
        """Reset sync state (use with caution — will re-ingest everything)."""
        self._records.clear()

    def remove_record(self, file_id: str) -> None:
        """Remove a single record so the file will be re-ingested next sync."""
        self._records.pop(file_id, None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_mime_filter(self, files: list[DriveFile]) -> list[DriveFile]:
        if not self._config.mime_types:
            return files
        return [f for f in files if f.mime_type in self._config.mime_types]

    @staticmethod
    def _noop_lister(source_id: str) -> list[DriveFile]:
        return []
