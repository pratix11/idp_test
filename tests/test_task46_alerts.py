"""Tests for Phase 7 Alerts (Task 50).

Tests cover:
- Alert dataclass creation and defaults
- AlertRule.matches() — predicate true/false
- AlertRule cooldown suppresses repeated firing
- AlertNotifier.notify() appends and .alerts returns copy
- AlertEngine.add_rule() / remove_rule() / rules
- AlertEngine.evaluate() fires matching rules and returns alerts
- AlertEngine.evaluate() does not fire non-matching rules
- AlertEngine.evaluate_many() processes a batch
- Built-in rule factories: denied_access, destructive_action, audit_access
"""

from __future__ import annotations

from datetime import timedelta, timezone, datetime
from unittest.mock import MagicMock

import pytest

from property_intel.enterprise.alerts import (
    Alert,
    AlertEngine,
    AlertNotifier,
    AlertRule,
    make_audit_access_rule,
    make_denied_access_rule,
    make_destructive_action_rule,
)
from property_intel.enterprise.audit import AuditEvent


def _event(**kwargs: str) -> AuditEvent:
    defaults = {"user_id": "alice", "action": "read", "resource": "documents"}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return AuditEvent(**defaults)  # type: ignore[arg-type]


# ── Alert ─────────────────────────────────────────────────────────────────────


def test_alert_defaults() -> None:
    event = _event()
    a = Alert(rule_name="test-rule", event=event)
    assert a.rule_name == "test-rule"
    assert a.event is event
    assert a.metadata == {}
    assert a.fired_at.tzinfo == timezone.utc


def test_alert_custom_metadata() -> None:
    a = Alert(rule_name="r", event=_event(), metadata={"severity": "high"})
    assert a.metadata["severity"] == "high"


# ── AlertRule ─────────────────────────────────────────────────────────────────


def test_alert_rule_matches_true() -> None:
    rule = AlertRule(name="denied", predicate=lambda e: e.result == "denied")
    event = _event(result="denied")
    assert rule.matches(event)


def test_alert_rule_matches_false() -> None:
    rule = AlertRule(name="denied", predicate=lambda e: e.result == "denied")
    event = _event(result="success")
    assert not rule.matches(event)


def test_alert_rule_no_cooldown_fires_every_time() -> None:
    rule = AlertRule(name="always", predicate=lambda e: True)
    event = _event()
    assert rule.matches(event)
    rule._record_fire()
    assert rule.matches(event)


def test_alert_rule_cooldown_suppresses_immediate_refiring() -> None:
    rule = AlertRule(
        name="rate-limited",
        predicate=lambda e: True,
        cooldown=timedelta(hours=1),
    )
    event = _event()
    assert rule.matches(event)
    rule._record_fire()
    assert not rule.matches(event)


def test_alert_rule_cooldown_allows_after_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = AlertRule(
        name="rate-limited",
        predicate=lambda e: True,
        cooldown=timedelta(seconds=1),
    )
    event = _event()
    rule._record_fire()

    # Advance _last_fired to be 2 seconds ago
    past = datetime.now(timezone.utc) - timedelta(seconds=2)
    rule._last_fired = past
    assert rule.matches(event)


# ── AlertNotifier ─────────────────────────────────────────────────────────────


def test_alert_notifier_notify_appends() -> None:
    notifier = AlertNotifier()
    alert = Alert(rule_name="r", event=_event())
    notifier.notify(alert)
    assert len(notifier.alerts) == 1
    assert notifier.alerts[0] is alert


def test_alert_notifier_alerts_returns_copy() -> None:
    notifier = AlertNotifier()
    notifier.notify(Alert(rule_name="r", event=_event()))
    copy = notifier.alerts
    copy.clear()
    assert len(notifier.alerts) == 1


def test_alert_notifier_clear() -> None:
    notifier = AlertNotifier()
    notifier.notify(Alert(rule_name="r", event=_event()))
    notifier.clear()
    assert len(notifier.alerts) == 0


# ── AlertEngine ───────────────────────────────────────────────────────────────


def test_alert_engine_add_rule() -> None:
    engine = AlertEngine()
    rule = AlertRule(name="r", predicate=lambda e: True)
    engine.add_rule(rule)
    assert len(engine.rules) == 1


def test_alert_engine_add_rule_chaining() -> None:
    engine = AlertEngine()
    result = engine.add_rule(AlertRule(name="r", predicate=lambda e: True))
    assert result is engine


def test_alert_engine_remove_rule() -> None:
    engine = AlertEngine()
    engine.add_rule(AlertRule(name="keep", predicate=lambda e: True))
    engine.add_rule(AlertRule(name="remove", predicate=lambda e: True))
    engine.remove_rule("remove")
    assert [r.name for r in engine.rules] == ["keep"]


def test_alert_engine_evaluate_matching_rule() -> None:
    notifier = AlertNotifier()
    engine = AlertEngine(notifier=notifier)
    engine.add_rule(AlertRule(name="denied", predicate=lambda e: e.result == "denied"))
    event = _event(result="denied")
    alerts = engine.evaluate(event)
    assert len(alerts) == 1
    assert alerts[0].rule_name == "denied"
    assert len(notifier.alerts) == 1


def test_alert_engine_evaluate_non_matching_rule() -> None:
    notifier = AlertNotifier()
    engine = AlertEngine(notifier=notifier)
    engine.add_rule(AlertRule(name="denied", predicate=lambda e: e.result == "denied"))
    event = _event(result="success")
    alerts = engine.evaluate(event)
    assert alerts == []
    assert len(notifier.alerts) == 0


def test_alert_engine_evaluate_multiple_rules_all_match() -> None:
    engine = AlertEngine()
    engine.add_rule(AlertRule(name="r1", predicate=lambda e: True))
    engine.add_rule(AlertRule(name="r2", predicate=lambda e: True))
    alerts = engine.evaluate(_event())
    assert {a.rule_name for a in alerts} == {"r1", "r2"}


def test_alert_engine_evaluate_many() -> None:
    engine = AlertEngine()
    engine.add_rule(AlertRule(name="denied", predicate=lambda e: e.result == "denied"))
    events = [
        _event(result="success"),
        _event(result="denied"),
        _event(result="denied"),
    ]
    alerts = engine.evaluate_many(events)
    assert len(alerts) == 2


def test_alert_engine_default_notifier_created() -> None:
    engine = AlertEngine()
    assert isinstance(engine.notifier, AlertNotifier)


# ── Built-in rule factories ───────────────────────────────────────────────────


def test_make_denied_access_rule_fires_on_denied() -> None:
    rule = make_denied_access_rule()
    assert rule.matches(_event(result="denied"))
    assert not rule.matches(_event(result="success"))


def test_make_destructive_action_rule_fires_on_delete() -> None:
    rule = make_destructive_action_rule()
    assert rule.matches(_event(action="delete"))
    assert rule.matches(_event(action="admin"))
    assert not rule.matches(_event(action="read"))


def test_make_destructive_action_rule_resource_filter() -> None:
    rule = make_destructive_action_rule(resources=["users"])
    assert rule.matches(_event(action="delete", resource="users"))
    assert not rule.matches(_event(action="delete", resource="documents"))


def test_make_audit_access_rule_fires_on_audit_resource() -> None:
    rule = make_audit_access_rule()
    assert rule.matches(_event(resource="audit"))
    assert not rule.matches(_event(resource="documents"))
