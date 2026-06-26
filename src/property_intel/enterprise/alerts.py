"""Alerts — Phase 7 enterprise feature.

AlertRules watch the audit stream for suspicious or noteworthy patterns.
When an AuditEvent matches a rule the AlertEngine fires an alert, and the
registered AlertNotifier handles delivery (in-memory list, callback, etc.).

Typical use-cases:
  - Too many denied-access attempts by one user
  - Any destructive action (delete/admin) on a sensitive resource
  - First-time access to the audit log itself
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


@dataclass
class Alert:
    """A fired alert — one rule matched one event.

    Attributes:
        rule_name: Name of the rule that fired.
        event:     The AuditEvent that triggered it.
        fired_at:  UTC timestamp of when the alert was generated.
        metadata:  Any extra context the rule attached.
    """

    rule_name: str
    event: Any  # AuditEvent — avoid circular import at runtime; type-checked via Any
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """A named predicate evaluated against every AuditEvent.

    Attributes:
        name:        Unique human-readable identifier.
        predicate:   Returns True when the event should trigger an alert.
        cooldown:    Minimum time between consecutive firings of this rule.
                     None means fire every time.
        description: Optional human-readable explanation.
    """

    name: str
    predicate: Callable[[Any], bool]
    cooldown: timedelta | None = None
    description: str = ""
    _last_fired: datetime | None = field(default=None, init=False, repr=False, compare=False)

    def matches(self, event: Any) -> bool:
        """Return True if the predicate fires and the cooldown has elapsed."""
        if not self.predicate(event):
            return False
        if self.cooldown is None:
            return True
        now = datetime.now(timezone.utc)
        if self._last_fired is None or (now - self._last_fired) >= self.cooldown:
            return True
        return False

    def _record_fire(self) -> None:
        self._last_fired = datetime.now(timezone.utc)


class AlertNotifier:
    """Receives fired alerts.

    The default implementation appends to an in-memory list so tests can
    assert easily.  Override ``notify`` to route to Slack, email, etc.
    """

    def __init__(self) -> None:
        self._alerts: list[Alert] = []

    def notify(self, alert: Alert) -> None:
        """Handle a fired alert.  Default: append to in-memory list."""
        self._alerts.append(alert)

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()


class AlertEngine:
    """Evaluates AuditEvents against registered AlertRules.

    Usage::

        engine = AlertEngine(notifier=AlertNotifier())
        engine.add_rule(AlertRule(
            name="denied-access",
            predicate=lambda e: e.result == "denied",
        ))
        engine.evaluate(event)   # fires notifier.notify() if a rule matches
    """

    def __init__(self, notifier: AlertNotifier | None = None) -> None:
        self._rules: list[AlertRule] = []
        self._notifier = notifier or AlertNotifier()

    @property
    def notifier(self) -> AlertNotifier:
        return self._notifier

    def add_rule(self, rule: AlertRule) -> AlertEngine:
        """Register a rule.  Returns self for chaining."""
        self._rules.append(rule)
        return self

    def remove_rule(self, name: str) -> None:
        """Remove all rules with the given name."""
        self._rules = [r for r in self._rules if r.name != name]

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules)

    def evaluate(self, event: Any) -> list[Alert]:
        """Check *event* against all rules; fire and return matching alerts."""
        fired: list[Alert] = []
        for rule in self._rules:
            if rule.matches(event):
                rule._record_fire()
                alert = Alert(rule_name=rule.name, event=event)
                self._notifier.notify(alert)
                fired.append(alert)
        return fired

    def evaluate_many(self, events: list[Any]) -> list[Alert]:
        """Evaluate a batch of events in order."""
        results: list[Alert] = []
        for event in events:
            results.extend(self.evaluate(event))
        return results


# ── Built-in rule factories ───────────────────────────────────────────────────

def make_denied_access_rule(cooldown: timedelta | None = None) -> AlertRule:
    """Fire on any denied access attempt."""
    return AlertRule(
        name="denied-access",
        predicate=lambda e: e.result == "denied",
        cooldown=cooldown,
        description="Fires whenever an access attempt is denied.",
    )


def make_destructive_action_rule(
    resources: list[str] | None = None,
    cooldown: timedelta | None = None,
) -> AlertRule:
    """Fire on delete or admin actions, optionally restricted to given resources."""
    _resources = set(resources) if resources else None

    def _predicate(e: Any) -> bool:
        if e.action not in ("delete", "admin"):
            return False
        return _resources is None or e.resource in _resources

    return AlertRule(
        name="destructive-action",
        predicate=_predicate,
        cooldown=cooldown,
        description="Fires on delete/admin actions on sensitive resources.",
    )


def make_audit_access_rule(cooldown: timedelta | None = None) -> AlertRule:
    """Fire whenever the audit log itself is accessed."""
    return AlertRule(
        name="audit-log-access",
        predicate=lambda e: e.resource == "audit",
        cooldown=cooldown,
        description="Fires when the audit resource is read or written.",
    )
