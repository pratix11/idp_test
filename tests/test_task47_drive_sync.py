"""Tests for Phase 7 Drive Sync (Task 51).

Tests cover:
- DriveFile and SyncRecord dataclass creation with defaults
- DriveSyncConfig defaults
- DriveSyncService.sync() returns new files only
- DriveSyncService.sync() skips already-recorded files
- DriveSyncService.mark_synced() records files and prevents re-return
- DriveSyncService.is_synced() reflects recorded state
- MIME type filtering keeps only configured types
- DriveSyncService.clear_records() resets state
- DriveSyncService.remove_record() removes a single record
- SyncResult properties: new_count, skipped, total_seen
- No-op lister returns empty list (default behaviour)
"""

from __future__ import annotations

from datetime import timezone

import pytest

from property_intel.enterprise.drive_sync import (
    DriveSyncConfig,
    DriveSyncService,
    DriveFile,
    SyncRecord,
    SyncResult,
)


def _file(file_id: str = "f1", name: str = "doc.pdf", mime_type: str = "application/pdf") -> DriveFile:
    return DriveFile(file_id=file_id, name=name, mime_type=mime_type)


def _lister(*files: DriveFile):
    return lambda source_id: list(files)


# ── DriveFile / SyncRecord ────────────────────────────────────────────────────


def test_drive_file_defaults() -> None:
    f = DriveFile(file_id="abc", name="file.pdf")
    assert f.mime_type == "application/pdf"
    assert f.metadata == {}
    assert f.modified.tzinfo == timezone.utc


def test_sync_record_defaults() -> None:
    r = SyncRecord(file_id="abc", name="file.pdf")
    assert r.synced_at.tzinfo == timezone.utc


# ── DriveSyncConfig ───────────────────────────────────────────────────────────


def test_drive_sync_config_defaults() -> None:
    cfg = DriveSyncConfig(source_id="my-folder")
    assert cfg.mime_types == ["application/pdf"]
    assert cfg.label == ""


def test_drive_sync_config_custom_mime_types() -> None:
    cfg = DriveSyncConfig(source_id="s", mime_types=["application/pdf", "text/plain"])
    assert "text/plain" in cfg.mime_types


# ── SyncResult ────────────────────────────────────────────────────────────────


def test_sync_result_new_count() -> None:
    result = SyncResult(new_files=[_file("a"), _file("b")], skipped=3, total_seen=5)
    assert result.new_count == 2
    assert result.skipped == 3
    assert result.total_seen == 5


# ── DriveSyncService ──────────────────────────────────────────────────────────


def test_sync_returns_all_new_files_on_first_run() -> None:
    files = [_file("f1"), _file("f2"), _file("f3")]
    service = DriveSyncService(DriveSyncConfig("src"), lister=_lister(*files))
    result = service.sync()
    assert result.new_count == 3
    assert result.skipped == 0
    assert result.total_seen == 3


def test_sync_skips_already_synced_files() -> None:
    f1 = _file("f1")
    f2 = _file("f2")
    service = DriveSyncService(DriveSyncConfig("src"), lister=_lister(f1, f2))
    service.mark_synced(f1)
    result = service.sync()
    assert result.new_count == 1
    assert result.new_files[0].file_id == "f2"
    assert result.skipped == 1


def test_sync_returns_empty_when_all_already_synced() -> None:
    f = _file("f1")
    service = DriveSyncService(DriveSyncConfig("src"), lister=_lister(f))
    service.mark_synced(f)
    result = service.sync()
    assert result.new_count == 0
    assert result.skipped == 1


def test_mark_synced_records_file() -> None:
    service = DriveSyncService(DriveSyncConfig("src"))
    f = _file("f1")
    record = service.mark_synced(f)
    assert record.file_id == "f1"
    assert service.is_synced("f1")


def test_is_synced_false_before_marking() -> None:
    service = DriveSyncService(DriveSyncConfig("src"))
    assert not service.is_synced("unknown")


def test_records_returns_copy() -> None:
    service = DriveSyncService(DriveSyncConfig("src"))
    service.mark_synced(_file("f1"))
    copy = service.records
    copy.clear()
    assert "f1" in service.records  # original unaffected


def test_mime_filter_excludes_wrong_types() -> None:
    files = [
        _file("f1", mime_type="application/pdf"),
        _file("f2", mime_type="text/html"),
        _file("f3", mime_type="application/pdf"),
    ]
    service = DriveSyncService(
        DriveSyncConfig("src", mime_types=["application/pdf"]),
        lister=_lister(*files),
    )
    result = service.sync()
    assert result.new_count == 2
    assert all(f.mime_type == "application/pdf" for f in result.new_files)


def test_mime_filter_empty_accepts_all() -> None:
    files = [
        _file("f1", mime_type="application/pdf"),
        _file("f2", mime_type="text/html"),
    ]
    service = DriveSyncService(
        DriveSyncConfig("src", mime_types=[]),
        lister=_lister(*files),
    )
    result = service.sync()
    assert result.new_count == 2


def test_clear_records_resets_state() -> None:
    f = _file("f1")
    service = DriveSyncService(DriveSyncConfig("src"), lister=_lister(f))
    service.mark_synced(f)
    service.clear_records()
    result = service.sync()
    assert result.new_count == 1


def test_remove_record_allows_reingestion() -> None:
    f = _file("f1")
    service = DriveSyncService(DriveSyncConfig("src"), lister=_lister(f))
    service.mark_synced(f)
    assert service.is_synced("f1")
    service.remove_record("f1")
    assert not service.is_synced("f1")
    result = service.sync()
    assert result.new_count == 1


def test_remove_record_nonexistent_is_noop() -> None:
    service = DriveSyncService(DriveSyncConfig("src"))
    service.remove_record("does-not-exist")  # should not raise


def test_default_no_lister_returns_empty_sync() -> None:
    service = DriveSyncService(DriveSyncConfig("src"))
    result = service.sync()
    assert result.new_count == 0
    assert result.total_seen == 0
