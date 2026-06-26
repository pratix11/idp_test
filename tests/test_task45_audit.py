"""Tests for Phase 7 Audit Logs (Task 49).

Tests cover:
- AuditEvent creation with defaults and explicit values
- AuditEvent.to_dict() / from_dict() round-trip
- Default timestamp is timezone-aware UTC
- AuditLogger.log() appends events
- AuditLogger.log_action() convenience method
- AuditLogger with file path writes JSONL
- AuditLog.from_logger() and from_jsonl()
- AuditLog query methods: by_user, by_action, by_resource, denied, since, summary
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from property_intel.enterprise.audit import AuditEvent, AuditLog, AuditLogger


# ── AuditEvent ────────────────────────────────────────────────────────────────


def test_audit_event_defaults() -> None:
    e = AuditEvent(user_id="alice", action="read", resource="documents")
    assert e.user_id == "alice"
    assert e.action == "read"
    assert e.resource == "documents"
    assert e.result == "success"
    assert e.metadata == {}


def test_audit_event_default_timestamp_is_utc() -> None:
    e = AuditEvent(user_id="alice", action="read", resource="documents")
    assert e.timestamp.tzinfo is not None
    assert e.timestamp.tzinfo == timezone.utc


def test_audit_event_explicit_result() -> None:
    e = AuditEvent(user_id="bob", action="delete", resource="users", result="denied")
    assert e.result == "denied"


def test_audit_event_metadata_stored() -> None:
    e = AuditEvent(
        user_id="alice",
        action="execute",
        resource="agents",
        metadata={"agent_type": "research", "query": "housing"},
    )
    assert e.metadata["agent_type"] == "research"
    assert e.metadata["query"] == "housing"


def test_audit_event_to_dict_keys() -> None:
    e = AuditEvent(user_id="alice", action="read", resource="documents")
    d = e.to_dict()
    assert set(d.keys()) >= {"user_id", "action", "resource", "result", "timestamp", "metadata"}


def test_audit_event_to_dict_timestamp_is_iso_string() -> None:
    e = AuditEvent(user_id="alice", action="read", resource="documents")
    d = e.to_dict()
    assert isinstance(d["timestamp"], str)
    # Should be parseable
    datetime.fromisoformat(d["timestamp"])


def test_audit_event_from_dict_round_trip() -> None:
    e = AuditEvent(
        user_id="carol",
        action="write",
        resource="evaluation",
        result="success",
        metadata={"doc_id": "123"},
    )
    d = e.to_dict()
    e2 = AuditEvent.from_dict(d)
    assert e2.user_id == e.user_id
    assert e2.action == e.action
    assert e2.resource == e.resource
    assert e2.result == e.result
    assert e2.metadata == e.metadata
    assert e2.timestamp == e.timestamp


def test_audit_event_from_dict_missing_timestamp_uses_now() -> None:
    d = {"user_id": "dave", "action": "read", "resource": "audit"}
    e = AuditEvent.from_dict(d)
    assert e.user_id == "dave"
    assert e.timestamp is not None


# ── AuditLogger ───────────────────────────────────────────────────────────────


def test_audit_logger_log_appends() -> None:
    logger = AuditLogger()
    e = AuditEvent(user_id="alice", action="read", resource="documents")
    logger.log(e)
    assert len(logger.events) == 1
    assert logger.events[0] is e


def test_audit_logger_log_multiple() -> None:
    logger = AuditLogger()
    for i in range(5):
        logger.log(AuditEvent(user_id=f"user{i}", action="read", resource="documents"))
    assert len(logger.events) == 5


def test_audit_logger_events_returns_copy() -> None:
    logger = AuditLogger()
    logger.log(AuditEvent(user_id="alice", action="read", resource="documents"))
    events = logger.events
    events.clear()
    assert len(logger.events) == 1  # original unaffected


def test_audit_logger_clear() -> None:
    logger = AuditLogger()
    logger.log(AuditEvent(user_id="alice", action="read", resource="documents"))
    logger.clear()
    assert len(logger.events) == 0


def test_audit_logger_log_action_convenience() -> None:
    logger = AuditLogger()
    event = logger.log_action("alice", "execute", "agents", result="success", agent_type="research")
    assert event.user_id == "alice"
    assert event.action == "execute"
    assert event.resource == "agents"
    assert event.result == "success"
    assert event.metadata == {"agent_type": "research"}
    assert len(logger.events) == 1


def test_audit_logger_log_action_denied() -> None:
    logger = AuditLogger()
    event = logger.log_action("guest", "delete", "documents", result="denied")
    assert event.result == "denied"


def test_audit_logger_writes_jsonl_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(path=path)
        logger.log_action("alice", "read", "documents")
        logger.log_action("bob", "write", "search")
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "user_id" in data
            assert "timestamp" in data


def test_audit_logger_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "dir" / "audit.jsonl"
        logger = AuditLogger(path=path)
        logger.log_action("alice", "read", "documents")
        assert path.exists()


# ── AuditLog ──────────────────────────────────────────────────────────────────


def _make_logger() -> AuditLogger:
    logger = AuditLogger()
    logger.log_action("alice", "read", "documents")
    logger.log_action("alice", "write", "documents", result="denied")
    logger.log_action("bob", "execute", "agents")
    logger.log_action("carol", "delete", "users", result="denied")
    logger.log_action("alice", "read", "audit")
    return logger


def test_audit_log_from_logger_length() -> None:
    logger = _make_logger()
    log = AuditLog.from_logger(logger)
    assert len(log) == 5


def test_audit_log_iter() -> None:
    logger = _make_logger()
    log = AuditLog.from_logger(logger)
    events = list(log)
    assert len(events) == 5


def test_audit_log_by_user() -> None:
    log = AuditLog.from_logger(_make_logger())
    alice_events = log.by_user("alice")
    assert len(alice_events) == 3
    assert all(e.user_id == "alice" for e in alice_events)


def test_audit_log_by_action() -> None:
    log = AuditLog.from_logger(_make_logger())
    reads = log.by_action("read")
    assert len(reads) == 2
    assert all(e.action == "read" for e in reads)


def test_audit_log_by_resource() -> None:
    log = AuditLog.from_logger(_make_logger())
    docs = log.by_resource("documents")
    assert len(docs) == 2


def test_audit_log_denied() -> None:
    log = AuditLog.from_logger(_make_logger())
    denied = log.denied()
    assert len(denied) == 2
    assert all(e.result == "denied" for e in denied)


def test_audit_log_since() -> None:
    logger = AuditLogger()
    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    e1 = AuditEvent(user_id="alice", action="read", resource="documents", timestamp=t1)
    e2 = AuditEvent(user_id="bob", action="read", resource="documents", timestamp=t2)
    e3 = AuditEvent(user_id="carol", action="read", resource="documents", timestamp=t3)
    logger.log(e1)
    logger.log(e2)
    logger.log(e3)
    log = AuditLog.from_logger(logger)
    result = log.since(t2)
    assert len(result) == 2
    assert e1 not in result


def test_audit_log_summary() -> None:
    log = AuditLog.from_logger(_make_logger())
    summary = log.summary()
    assert summary["read"] == 2
    assert summary["write"] == 1
    assert summary["execute"] == 1
    assert summary["delete"] == 1


def test_audit_log_from_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(path=path)
        logger.log_action("alice", "read", "documents")
        logger.log_action("bob", "execute", "agents")

        log = AuditLog.from_jsonl(path)
        assert len(log) == 2
        users = {e.user_id for e in log}
        assert users == {"alice", "bob"}
