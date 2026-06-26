"""Tests for Phase 6 EvaluationPipeline and EvaluationReport (Task 46).

Tests cover:
- Pipeline instantiation with/without evaluators
- add_evaluator / evaluator_names
- run() calls each evaluator and merges scores
- run(skip_compute=True) passes through to evaluators
- run() on empty dataset
- EvaluationReport.summary() format
- EvaluationReport.to_dict() / to_json() / from_json() round-trip
- EvaluationReport.merge() combines two reports
- Evaluator protocol: any object with evaluate() + available_metrics() qualifies
- Full integration: pipeline with both RagasEvaluator and DeepEvaluator (skip mode)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from property_intel.evaluation.dataset import EvalDataset, EvalSample
from property_intel.evaluation.deepeval_eval import DeepEvaluator
from property_intel.evaluation.pipeline import Evaluator, EvaluationPipeline, EvaluationReport
from property_intel.evaluation.ragas_eval import RagasEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sample_dataset(n: int = 2) -> EvalDataset:
    ds = EvalDataset()
    for i in range(n):
        ds.add(EvalSample(question=f"q{i}", answer=f"a{i}", contexts=[f"c{i}"]))
    return ds


def _mock_evaluator(scores: dict) -> MagicMock:
    ev = MagicMock()
    ev.evaluate.return_value = scores
    ev.available_metrics.return_value = list(scores.keys())
    return ev


# ── EvaluationPipeline ────────────────────────────────────────────────────────


def test_pipeline_starts_empty() -> None:
    p = EvaluationPipeline()
    assert p.evaluator_names() == []


def test_pipeline_add_evaluator() -> None:
    p = EvaluationPipeline()
    p.add_evaluator("ragas", _mock_evaluator({"faithfulness": 0.9}))
    assert "ragas" in p.evaluator_names()


def test_pipeline_init_with_dict() -> None:
    ev = _mock_evaluator({"x": 0.5})
    p = EvaluationPipeline({"custom": ev})
    assert "custom" in p.evaluator_names()


def test_run_calls_each_evaluator() -> None:
    ev1 = _mock_evaluator({"faithfulness": 0.9})
    ev2 = _mock_evaluator({"correctness": 0.8})
    p = EvaluationPipeline({"ragas": ev1, "deepeval": ev2})
    ds = _sample_dataset()
    p.run(ds)
    ev1.evaluate.assert_called_once()
    ev2.evaluate.assert_called_once()


def test_run_merges_scores_from_all_evaluators() -> None:
    ev1 = _mock_evaluator({"faithfulness": 0.9, "answer_relevancy": 0.85})
    ev2 = _mock_evaluator({"correctness": 0.75})
    p = EvaluationPipeline({"ragas": ev1, "deepeval": ev2})
    report = p.run(_sample_dataset())
    assert "faithfulness" in report.scores
    assert "correctness" in report.scores
    assert len(report.scores) == 3


def test_run_empty_pipeline_returns_empty_report() -> None:
    p = EvaluationPipeline()
    report = p.run(_sample_dataset())
    assert report.scores == {}
    assert report.details == {}


def test_run_skip_compute_passes_kwarg_to_evaluators() -> None:
    ev = _mock_evaluator({"x": float("nan")})
    p = EvaluationPipeline({"ev": ev})
    p.run(_sample_dataset(), skip_compute=True)
    _, kwargs = ev.evaluate.call_args
    assert kwargs.get("_skip_compute") is True


def test_run_details_keyed_by_evaluator_name() -> None:
    ev1 = _mock_evaluator({"f": 0.9})
    ev2 = _mock_evaluator({"c": 0.8})
    p = EvaluationPipeline({"ragas": ev1, "deep": ev2})
    report = p.run(_sample_dataset())
    assert "ragas" in report.details
    assert "deep" in report.details


# ── EvaluationReport ──────────────────────────────────────────────────────────


def test_report_summary_no_scores() -> None:
    r = EvaluationReport()
    assert "No scores computed" in r.summary()


def test_report_summary_with_scores() -> None:
    r = EvaluationReport(scores={"faithfulness": 0.9, "correctness": 0.75})
    summary = r.summary()
    assert "faithfulness" in summary
    assert "0.900" in summary
    assert "correctness" in summary


def test_report_to_dict() -> None:
    r = EvaluationReport(scores={"a": 0.5}, details={"ev": {"a": 0.5}})
    d = r.to_dict()
    assert d["scores"] == {"a": 0.5}
    assert "ev" in d["details"]


def test_report_json_round_trip() -> None:
    r = EvaluationReport(
        scores={"faithfulness": 0.88, "correctness": 0.72},
        details={"ragas": {"faithfulness": 0.88}},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        r.to_json(path)
        loaded = EvaluationReport.from_json(path)
    assert loaded.scores["faithfulness"] == pytest.approx(0.88)
    assert loaded.scores["correctness"] == pytest.approx(0.72)


def test_report_merge_combines_scores() -> None:
    r1 = EvaluationReport(scores={"a": 0.9}, details={"ev1": {}})
    r2 = EvaluationReport(scores={"b": 0.8}, details={"ev2": {}})
    merged = r1.merge(r2)
    assert "a" in merged.scores
    assert "b" in merged.scores
    assert "ev1" in merged.details
    assert "ev2" in merged.details


def test_report_merge_later_score_wins_on_conflict() -> None:
    r1 = EvaluationReport(scores={"x": 0.5})
    r2 = EvaluationReport(scores={"x": 0.9})
    merged = r1.merge(r2)
    assert merged.scores["x"] == pytest.approx(0.9)


# ── Evaluator protocol ────────────────────────────────────────────────────────


def test_evaluator_protocol_satisfied_by_mock() -> None:
    ev = _mock_evaluator({"m": 0.5})
    assert isinstance(ev, Evaluator)


def test_evaluator_protocol_satisfied_by_ragas_evaluator() -> None:
    assert isinstance(RagasEvaluator(), Evaluator)


def test_evaluator_protocol_satisfied_by_deep_evaluator() -> None:
    assert isinstance(DeepEvaluator(), Evaluator)


# ── Full integration: both evaluators in skip_compute mode ───────────────────


def test_pipeline_with_ragas_and_deepeval_skip_compute() -> None:
    p = EvaluationPipeline({
        "ragas": RagasEvaluator(),
        "deepeval": DeepEvaluator(),
    })
    ds = _sample_dataset(3)
    report = p.run(ds, skip_compute=True)
    expected_keys = {"faithfulness", "answer_relevancy", "context_precision", "correctness"}
    assert set(report.scores.keys()) == expected_keys
    for v in report.scores.values():
        assert math.isnan(v)
    assert "ragas" in report.details
    assert "deepeval" in report.details
