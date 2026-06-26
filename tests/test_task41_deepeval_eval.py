"""Tests for Phase 6 DeepEvaluator (Task 44).

Tests cover:
- Instantiation with default and custom model/threshold
- available_metrics() returns correctness + answer_relevancy
- evaluate() on empty dataset returns zeros
- evaluate() with _skip_compute=True returns nan scores (no API calls)
- make_test_case() converts EvalSample to LLMTestCase correctly
- evaluate() with mocked GEval measures produces averaged scores
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
from deepeval.test_case import LLMTestCase

from property_intel.evaluation.dataset import EvalDataset, EvalSample
from property_intel.evaluation.deepeval_eval import DeepEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sample_dataset(n: int = 3) -> EvalDataset:
    ds = EvalDataset()
    for i in range(n):
        ds.add(EvalSample(
            question=f"What is rule {i}?",
            answer=f"Rule {i} states X.",
            contexts=[f"Section {i}: Rule {i}."],
            ground_truth=f"Rule {i} states X.",
        ))
    return ds


# ── Instantiation ─────────────────────────────────────────────────────────────


def test_deepevaluator_instantiates_with_defaults() -> None:
    ev = DeepEvaluator()
    assert ev._model == "gpt-4o-mini"
    assert ev._threshold == 0.5


def test_deepevaluator_accepts_custom_model() -> None:
    ev = DeepEvaluator(model="gpt-4o", threshold=0.7)
    assert ev._model == "gpt-4o"
    assert ev._threshold == 0.7


# ── available_metrics ─────────────────────────────────────────────────────────


def test_available_metrics_returns_two_names() -> None:
    ev = DeepEvaluator()
    assert set(ev.available_metrics()) == {"correctness", "answer_relevancy"}


# ── evaluate() on empty dataset ───────────────────────────────────────────────


def test_evaluate_empty_dataset_returns_zeros() -> None:
    ev = DeepEvaluator()
    scores = ev.evaluate(EvalDataset())
    assert scores == {"correctness": 0.0, "answer_relevancy": 0.0}


# ── evaluate() _skip_compute ─────────────────────────────────────────────────


def test_evaluate_skip_compute_returns_nan() -> None:
    ev = DeepEvaluator()
    scores = ev.evaluate(_sample_dataset(2), _skip_compute=True)
    assert set(scores.keys()) == {"correctness", "answer_relevancy"}
    for v in scores.values():
        assert math.isnan(v)


def test_evaluate_skip_compute_makes_no_metric_calls() -> None:
    ev = DeepEvaluator()
    with patch("property_intel.evaluation.deepeval_eval._make_correctness_metric") as m:
        ev.evaluate(_sample_dataset(2), _skip_compute=True)
        m.assert_not_called()


# ── make_test_case ────────────────────────────────────────────────────────────


def test_make_test_case_returns_llm_test_case() -> None:
    ev = DeepEvaluator()
    sample = EvalSample(
        question="What is the penalty?",
        answer="Rs. 50,000.",
        contexts=["Section 59: penalty is Rs. 50,000."],
        ground_truth="Rs. 50,000 per day.",
    )
    tc = ev.make_test_case(sample)
    assert isinstance(tc, LLMTestCase)
    assert tc.input == "What is the penalty?"
    assert tc.actual_output == "Rs. 50,000."
    assert tc.expected_output == "Rs. 50,000 per day."
    assert tc.retrieval_context == ["Section 59: penalty is Rs. 50,000."]


def test_make_test_case_handles_empty_contexts() -> None:
    ev = DeepEvaluator()
    sample = EvalSample(question="q", answer="a")
    tc = ev.make_test_case(sample)
    assert tc.retrieval_context == []


# ── evaluate() with mocked metrics ───────────────────────────────────────────


def test_evaluate_averages_scores_across_samples() -> None:
    ev = DeepEvaluator()
    ds = _sample_dataset(3)

    mock_correctness = MagicMock()
    mock_relevancy = MagicMock()
    mock_correctness.score = 0.9
    mock_relevancy.score = 0.8

    with patch("property_intel.evaluation.deepeval_eval._make_correctness_metric", return_value=mock_correctness):
        with patch("property_intel.evaluation.deepeval_eval._make_relevancy_metric", return_value=mock_relevancy):
            scores = ev.evaluate(ds)

    assert mock_correctness.measure.call_count == 3
    assert mock_relevancy.measure.call_count == 3
    assert abs(scores["correctness"] - 0.9) < 1e-9
    assert abs(scores["answer_relevancy"] - 0.8) < 1e-9


def test_evaluate_returns_floats() -> None:
    ev = DeepEvaluator()
    ds = _sample_dataset(1)

    mock_m = MagicMock()
    mock_m.score = 0.75

    with patch("property_intel.evaluation.deepeval_eval._make_correctness_metric", return_value=mock_m):
        with patch("property_intel.evaluation.deepeval_eval._make_relevancy_metric", return_value=mock_m):
            scores = ev.evaluate(ds)

    for v in scores.values():
        assert isinstance(v, float)
