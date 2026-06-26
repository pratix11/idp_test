"""Audit Logs — Phase 7 enterprise feature.

Every sensitive action on the platform is recorded as an AuditEvent:
who did what, when, on which resource, with what result.

AuditLogger writes events to an in-memory store (default) or to a JSONL
file.  AuditLog is the read/query interface.

This provides the immutable audit trail required for SOC2/GDPR compliance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    """A single auditable action on the platform.

    Attributes:
        user_id:   Who performed the action.
        action:    What they did (e.g. "read", "execute", "delete").
        resource:  What they acted on (e.g. "documents", "agents").
        result:    Outcome — "success" or "denied".
        timestamp: When the action occurred (UTC).
        metadata:  Extra context (document_id, query, agent_type, etc.).
    """

    user_id: str
    action: str
    resource: str
    result: str = "success"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        ts_raw = data.get("timestamp", "")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
        return cls(
            user_id=data["user_id"],
            action=data["action"],
            resource=data["resource"],
            result=data.get("result", "success"),
            timestamp=ts,
            metadata=data.get("metadata", {}),
        )


class AuditLogger:
    """Records AuditEvents to an in-memory list and optionally a JSONL file.

    Usage:
        logger = AuditLogger()
        logger.log(AuditEvent(user_id="alice", action="read", resource="documents"))

        # Persist to disk:
        logger = AuditLogger(path="audit.jsonl")
        logger.log(...)
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._path = Path(path) if path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        """Record *event* in memory and append to file if configured."""
        self._events.append(event)
        if self._path:
            with open(self._path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        *,
        result: str = "success",
        **metadata: Any,
    ) -> AuditEvent:
        """Convenience: build and log an event in one call."""
        event = AuditEvent(
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            metadata=dict(metadata),
        )
        self.log(event)
        return event

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()


class AuditLog:
    """Query interface over a collection of AuditEvents.

    Usage:
        log = AuditLog.from_logger(logger)
        log.by_user("alice")
        log.by_action("delete")
        log.denied()

        # Load from file:
        log = AuditLog.from_jsonl("audit.jsonl")
    """

    def __init__(self, events: list[AuditEvent]) -> None:
        self._events = events

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Any:
        return iter(self._events)

    @classmethod
    def from_logger(cls, logger: AuditLogger) -> AuditLog:
        return cls(list(logger.events))

    @classmethod
    def from_jsonl(cls, path: Path | str) -> AuditLog:
        events: list[AuditEvent] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(AuditEvent.from_dict(json.loads(line)))
        return cls(events)

    # ── filters ───────────────────────────────────────────────────────────

    def by_user(self, user_id: str) -> list[AuditEvent]:
        return [e for e in self._events if e.user_id == user_id]

    def by_action(self, action: str) -> list[AuditEvent]:
        return [e for e in self._events if e.action == action]

    def by_resource(self, resource: str) -> list[AuditEvent]:
        return [e for e in self._events if e.resource == resource]

    def denied(self) -> list[AuditEvent]:
        """Return all events where access was denied."""
        return [e for e in self._events if e.result == "denied"]

    def since(self, dt: datetime) -> list[AuditEvent]:
        """Return events after *dt* (inclusive)."""
        return [e for e in self._events if e.timestamp >= dt]

    def summary(self) -> dict[str, int]:
        """Return counts grouped by action."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.action] = counts.get(e.action, 0) + 1
        return counts
