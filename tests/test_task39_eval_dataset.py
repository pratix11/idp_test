"""Tests for Phase 6 EvalDataset and EvalSample (Task 42).

Tests cover:
- EvalSample creation, to_dict, from_dict round-trip
- EvalDataset add / extend / len / iter / getitem
- subset() returns first n
- from_list() factory
- from_jsonl() / to_jsonl() round-trip
- to_ragas_dict() columnar format
- to_deepeval_test_cases() list format
- Missing optional fields default correctly
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from property_intel.evaluation.dataset import EvalDataset, EvalSample


# ── EvalSample ────────────────────────────────────────────────────────────────


def test_eval_sample_creation() -> None:
    s = EvalSample(
        question="What is the registration fee?",
        answer="Rs. 10,000.",
        contexts=["Fee is Rs. 10,000 per Section 4."],
        ground_truth="Rs. 10,000",
    )
    assert s.question == "What is the registration fee?"
    assert s.answer == "Rs. 10,000."
    assert len(s.contexts) == 1
    assert s.ground_truth == "Rs. 10,000"


def test_eval_sample_defaults() -> None:
    s = EvalSample(question="q", answer="a")
    assert s.contexts == []
    assert s.ground_truth == ""
    assert s.metadata == {}


def test_eval_sample_to_dict() -> None:
    s = EvalSample(question="q", answer="a", contexts=["ctx"], ground_truth="gt")
    d = s.to_dict()
    assert d["question"] == "q"
    assert d["answer"] == "a"
    assert d["contexts"] == ["ctx"]
    assert d["ground_truth"] == "gt"


def test_eval_sample_from_dict_round_trip() -> None:
    original = EvalSample(
        question="What are builder duties?",
        answer="Builders must register.",
        contexts=["Section 3: Registration mandatory."],
        ground_truth="Register within 30 days.",
        metadata={"source": "MahaRERA"},
    )
    restored = EvalSample.from_dict(original.to_dict())
    assert restored.question == original.question
    assert restored.answer == original.answer
    assert restored.contexts == original.contexts
    assert restored.ground_truth == original.ground_truth
    assert restored.metadata == original.metadata


def test_eval_sample_from_dict_missing_optional_fields() -> None:
    s = EvalSample.from_dict({"question": "q", "answer": "a"})
    assert s.contexts == []
    assert s.ground_truth == ""
    assert s.metadata == {}


# ── EvalDataset ───────────────────────────────────────────────────────────────


def _sample(q: str = "q", a: str = "a") -> EvalSample:
    return EvalSample(question=q, answer=a, contexts=[f"ctx for {q}"])


def test_eval_dataset_starts_empty() -> None:
    ds = EvalDataset()
    assert len(ds) == 0


def test_eval_dataset_add() -> None:
    ds = EvalDataset()
    ds.add(_sample("q1"))
    assert len(ds) == 1


def test_eval_dataset_extend() -> None:
    ds = EvalDataset()
    ds.extend([_sample("q1"), _sample("q2")])
    assert len(ds) == 2


def test_eval_dataset_iter() -> None:
    ds = EvalDataset([_sample("q1"), _sample("q2")])
    questions = [s.question for s in ds]
    assert questions == ["q1", "q2"]


def test_eval_dataset_getitem() -> None:
    ds = EvalDataset([_sample("q1"), _sample("q2")])
    assert ds[0].question == "q1"
    assert ds[1].question == "q2"


def test_eval_dataset_subset() -> None:
    ds = EvalDataset([_sample(f"q{i}") for i in range(5)])
    sub = ds.subset(3)
    assert len(sub) == 3
    assert sub[0].question == "q0"


def test_eval_dataset_from_list() -> None:
    records = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"]},
        {"question": "q2", "answer": "a2"},
    ]
    ds = EvalDataset.from_list(records)
    assert len(ds) == 2
    assert ds[0].question == "q1"
    assert ds[1].contexts == []


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def test_jsonl_round_trip() -> None:
    samples = [_sample(f"q{i}") for i in range(3)]
    ds = EvalDataset(samples)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval.jsonl"
        ds.to_jsonl(path)
        loaded = EvalDataset.from_jsonl(path)
    assert len(loaded) == 3
    for i, s in enumerate(loaded):
        assert s.question == f"q{i}"


def test_jsonl_preserves_all_fields() -> None:
    sample = EvalSample(
        question="What is the penalty?",
        answer="Rs. 50,000.",
        contexts=["Penalty is Rs. 50,000."],
        ground_truth="Rs. 50,000",
        metadata={"doc_id": "DOC-001"},
    )
    ds = EvalDataset([sample])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        ds.to_jsonl(path)
        loaded = EvalDataset.from_jsonl(path)
    s = loaded[0]
    assert s.ground_truth == "Rs. 50,000"
    assert s.metadata["doc_id"] == "DOC-001"


# ── Library conversions ───────────────────────────────────────────────────────


def test_to_ragas_dict_shape() -> None:
    ds = EvalDataset([_sample("q1", "a1"), _sample("q2", "a2")])
    d = ds.to_ragas_dict()
    assert set(d.keys()) == {"question", "answer", "contexts", "ground_truth"}
    assert len(d["question"]) == 2
    assert d["question"] == ["q1", "q2"]
    assert d["answer"] == ["a1", "a2"]


def test_to_ragas_dict_empty() -> None:
    ds = EvalDataset()
    d = ds.to_ragas_dict()
    assert all(v == [] for v in d.values())


def test_to_deepeval_test_cases_shape() -> None:
    sample = EvalSample(
        question="q", answer="a", contexts=["ctx"], ground_truth="gt"
    )
    ds = EvalDataset([sample])
    cases = ds.to_deepeval_test_cases()
    assert len(cases) == 1
    assert cases[0]["input"] == "q"
    assert cases[0]["actual_output"] == "a"
    assert cases[0]["retrieval_context"] == ["ctx"]
    assert cases[0]["expected_output"] == "gt"


def test_to_deepeval_test_cases_empty() -> None:
    ds = EvalDataset()
    assert ds.to_deepeval_test_cases() == []
