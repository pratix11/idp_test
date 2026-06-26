"""AuditEvent and AuditLogger — stub, full implementation in Task 49."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditEvent:
    """A single auditable action. Full implementation in Task 49."""

    user_id: str
    action: str
    resource: str
    result: str = "success"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Records AuditEvents. Full implementation in Task 49."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        self._events.append(event)
