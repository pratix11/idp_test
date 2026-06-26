"""Tests for Phase 7 Document Versioning (Task 52).

Tests cover:
- DocumentVersion creation and __str__
- VersionManager.add_version() auto-increments version numbers
- VersionManager.history() returns versions in order
- VersionManager.latest() returns the newest version
- VersionManager.at() retrieves by version number
- VersionManager.version_count() and all_document_ids()
- VersionManager.remove_document() clears history
- VersionManager.diff() computes correct VersionDiff
- VersionDiff.hash_changed, author_changed, changelogs, time_delta_seconds
- diff() raises ValueError for unknown version numbers
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from property_intel.enterprise.versioning import (
    DocumentVersion,
    VersionDiff,
    VersionManager,
)


# ── DocumentVersion ───────────────────────────────────────────────────────────


def test_document_version_str() -> None:
    v = DocumentVersion(
        document_id="doc-1",
        version_number=3,
        content_hash="abc",
        author="alice",
    )
    assert str(v) == "doc-1@v3"


def test_document_version_default_timestamp_utc() -> None:
    v = DocumentVersion(
        document_id="doc-1", version_number=1, content_hash="abc", author="alice"
    )
    assert v.created_at.tzinfo == timezone.utc


def test_document_version_default_changelog_empty() -> None:
    v = DocumentVersion(
        document_id="doc-1", version_number=1, content_hash="abc", author="alice"
    )
    assert v.changelog == ""


def test_document_version_metadata_stored() -> None:
    v = DocumentVersion(
        document_id="doc-1",
        version_number=1,
        content_hash="abc",
        author="alice",
        metadata={"size_bytes": 1024},
    )
    assert v.metadata["size_bytes"] == 1024


# ── VersionManager — add & read ───────────────────────────────────────────────


def test_add_version_increments_numbers() -> None:
    manager = VersionManager()
    v1 = manager.add_version("doc-1", content_hash="h1", author="alice")
    v2 = manager.add_version("doc-1", content_hash="h2", author="bob")
    assert v1.version_number == 1
    assert v2.version_number == 2


def test_add_version_independent_per_document() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.add_version("doc-1", content_hash="h2", author="alice")
    v = manager.add_version("doc-2", content_hash="h3", author="bob")
    assert v.version_number == 1


def test_add_version_stores_changelog() -> None:
    manager = VersionManager()
    v = manager.add_version("doc-1", content_hash="h1", author="alice", changelog="Initial")
    assert v.changelog == "Initial"


def test_add_version_stores_extra_metadata() -> None:
    manager = VersionManager()
    v = manager.add_version("doc-1", content_hash="h1", author="alice", source="drive")
    assert v.metadata["source"] == "drive"


def test_history_returns_versions_in_order() -> None:
    manager = VersionManager()
    for i in range(5):
        manager.add_version("doc-1", content_hash=f"h{i}", author="alice")
    history = manager.history("doc-1")
    assert [v.version_number for v in history] == [1, 2, 3, 4, 5]


def test_history_returns_copy() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    h = manager.history("doc-1")
    h.clear()
    assert manager.version_count("doc-1") == 1


def test_history_empty_for_unknown_document() -> None:
    manager = VersionManager()
    assert manager.history("unknown") == []


def test_latest_returns_newest_version() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    v2 = manager.add_version("doc-1", content_hash="h2", author="bob")
    assert manager.latest("doc-1") is v2


def test_latest_returns_none_for_unknown_document() -> None:
    manager = VersionManager()
    assert manager.latest("unknown") is None


def test_at_returns_correct_version() -> None:
    manager = VersionManager()
    v1 = manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.add_version("doc-1", content_hash="h2", author="bob")
    assert manager.at("doc-1", 1) is v1


def test_at_returns_none_for_missing_version() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    assert manager.at("doc-1", 99) is None


def test_version_count() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.add_version("doc-1", content_hash="h2", author="alice")
    assert manager.version_count("doc-1") == 2
    assert manager.version_count("doc-2") == 0


def test_all_document_ids() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.add_version("doc-2", content_hash="h2", author="bob")
    ids = set(manager.all_document_ids())
    assert ids == {"doc-1", "doc-2"}


def test_remove_document() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.remove_document("doc-1")
    assert manager.version_count("doc-1") == 0
    assert "doc-1" not in manager.all_document_ids()


def test_remove_document_nonexistent_is_noop() -> None:
    manager = VersionManager()
    manager.remove_document("nonexistent")  # should not raise


# ── VersionManager — diff ─────────────────────────────────────────────────────


def _make_manager_with_versions() -> tuple[VersionManager, list[DocumentVersion]]:
    manager = VersionManager()
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 4, tzinfo=timezone.utc)
    versions = [
        DocumentVersion("doc-1", 1, "hash_a", "alice", changelog="Initial", created_at=t0),
        DocumentVersion("doc-1", 2, "hash_b", "bob", changelog="Fix typo", created_at=t1),
        DocumentVersion("doc-1", 3, "hash_c", "carol", changelog="Major update", created_at=t2),
    ]
    manager._versions["doc-1"] = versions
    return manager, versions


def test_diff_hash_changed() -> None:
    manager, _ = _make_manager_with_versions()
    d = manager.diff("doc-1", 1, 2)
    assert d.hash_changed is True


def test_diff_same_hash_not_changed() -> None:
    manager = VersionManager()
    v1 = manager.add_version("doc-1", content_hash="same", author="alice")
    v2 = manager.add_version("doc-1", content_hash="same", author="alice")
    d = manager.diff("doc-1", 1, 2)
    assert d.hash_changed is False


def test_diff_author_changed() -> None:
    manager, _ = _make_manager_with_versions()
    d = manager.diff("doc-1", 1, 2)
    assert d.author_changed is True


def test_diff_same_author_not_changed() -> None:
    manager = VersionManager()
    manager.add_version("doc-1", content_hash="h1", author="alice")
    manager.add_version("doc-1", content_hash="h2", author="alice")
    d = manager.diff("doc-1", 1, 2)
    assert d.author_changed is False


def test_diff_time_delta() -> None:
    manager, _ = _make_manager_with_versions()
    d = manager.diff("doc-1", 1, 2)
    assert d.time_delta_seconds == 86400.0  # 1 day


def test_diff_changelogs_between_versions() -> None:
    manager, _ = _make_manager_with_versions()
    d = manager.diff("doc-1", 1, 3)
    assert "Fix typo" in d.changelogs
    assert "Major update" in d.changelogs
    assert "Initial" not in d.changelogs  # from_version is excluded


def test_diff_raises_for_unknown_from_version() -> None:
    manager, _ = _make_manager_with_versions()
    with pytest.raises(ValueError, match="99"):
        manager.diff("doc-1", 99, 2)


def test_diff_raises_for_unknown_to_version() -> None:
    manager, _ = _make_manager_with_versions()
    with pytest.raises(ValueError, match="99"):
        manager.diff("doc-1", 1, 99)


def test_diff_version_numbers_in_result() -> None:
    manager, _ = _make_manager_with_versions()
    d = manager.diff("doc-1", 1, 3)
    assert d.from_version == 1
    assert d.to_version == 3
    assert d.document_id == "doc-1"
