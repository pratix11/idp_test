"""Document Versioning — Phase 7 enterprise feature.

Every time a document is updated, a new DocumentVersion is created
capturing the content hash, the author, a changelog message, and the
timestamp.  VersionManager maintains an ordered history per document and
exposes queries like latest(), at(), history(), and diff_summary().

Versions are numbered sequentially starting at 1.  The content itself is
stored externally (e.g. the processed-markdown store); only the hash is
kept here for integrity verification and change detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DocumentVersion:
    """A single immutable version of a document.

    Attributes:
        document_id:    Stable identifier for the document.
        version_number: 1-based sequence number within this document.
        content_hash:   SHA-256 (or any) hash of the document content.
        author:         User who created this version.
        changelog:      Human-readable description of changes.
        created_at:     UTC timestamp of when this version was created.
        metadata:       Extra fields (source_path, byte_size, etc.).
    """

    document_id: str
    version_number: int
    content_hash: str
    author: str
    changelog: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.document_id}@v{self.version_number}"


@dataclass
class VersionDiff:
    """Summary of differences between two versions of the same document.

    Attributes:
        document_id: The document these versions belong to.
        from_version: Older version number.
        to_version:   Newer version number.
        hash_changed: Whether the content hash changed.
        author_changed: Whether the author changed.
        time_delta_seconds: Seconds elapsed between the two versions.
        changelogs: List of changelog messages between the two versions.
    """

    document_id: str
    from_version: int
    to_version: int
    hash_changed: bool
    author_changed: bool
    time_delta_seconds: float
    changelogs: list[str]


class VersionManager:
    """Maintains version history for one or more documents.

    Usage::

        manager = VersionManager()
        v1 = manager.add_version("doc-1", content_hash="abc", author="alice",
                                  changelog="Initial upload")
        v2 = manager.add_version("doc-1", content_hash="def", author="bob",
                                  changelog="Updated section 3")

        manager.latest("doc-1")          # → v2
        manager.history("doc-1")         # → [v1, v2]
        manager.at("doc-1", 1)           # → v1
        manager.diff("doc-1", 1, 2)      # → VersionDiff(...)
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[DocumentVersion]] = {}

    # ── write ─────────────────────────────────────────────────────────────────

    def add_version(
        self,
        document_id: str,
        content_hash: str,
        author: str,
        *,
        changelog: str = "",
        **metadata: Any,
    ) -> DocumentVersion:
        """Create and store a new version for *document_id*.

        The version number is assigned automatically as len(history) + 1.
        """
        versions = self._versions.setdefault(document_id, [])
        version_number = len(versions) + 1
        version = DocumentVersion(
            document_id=document_id,
            version_number=version_number,
            content_hash=content_hash,
            author=author,
            changelog=changelog,
            metadata=dict(metadata),
        )
        versions.append(version)
        return version

    def remove_document(self, document_id: str) -> None:
        """Remove all version history for a document."""
        self._versions.pop(document_id, None)

    # ── read ──────────────────────────────────────────────────────────────────

    def history(self, document_id: str) -> list[DocumentVersion]:
        """Return all versions in ascending order (oldest first)."""
        return list(self._versions.get(document_id, []))

    def latest(self, document_id: str) -> DocumentVersion | None:
        """Return the most recent version, or None if no history exists."""
        versions = self._versions.get(document_id)
        return versions[-1] if versions else None

    def at(self, document_id: str, version_number: int) -> DocumentVersion | None:
        """Return the version at *version_number*, or None if not found."""
        for v in self._versions.get(document_id, []):
            if v.version_number == version_number:
                return v
        return None

    def version_count(self, document_id: str) -> int:
        """Return the number of versions stored for *document_id*."""
        return len(self._versions.get(document_id, []))

    def all_document_ids(self) -> list[str]:
        """Return all document IDs that have at least one version."""
        return list(self._versions.keys())

    # ── diff ──────────────────────────────────────────────────────────────────

    def diff(
        self, document_id: str, from_version: int, to_version: int
    ) -> VersionDiff:
        """Compute a diff summary between two version numbers.

        Raises:
            ValueError: If either version number is not found.
        """
        v_from = self.at(document_id, from_version)
        v_to = self.at(document_id, to_version)
        if v_from is None:
            raise ValueError(f"Version {from_version} not found for '{document_id}'")
        if v_to is None:
            raise ValueError(f"Version {to_version} not found for '{document_id}'")

        # Collect changelogs for all versions in [from+1 .. to]
        changelogs = [
            v.changelog
            for v in self.history(document_id)
            if from_version < v.version_number <= to_version and v.changelog
        ]

        delta = (v_to.created_at - v_from.created_at).total_seconds()
        return VersionDiff(
            document_id=document_id,
            from_version=from_version,
            to_version=to_version,
            hash_changed=v_from.content_hash != v_to.content_hash,
            author_changed=v_from.author != v_to.author,
            time_delta_seconds=delta,
            changelogs=changelogs,
        )
