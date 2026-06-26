"""DeepEvaluator — Phase 6 DeepEval-based LLM output quality metrics.

Uses DeepEval's G-Eval (LLM-as-judge with chain-of-thought reasoning) to score:
- correctness:        Does the answer match the ground truth?
- answer_relevancy:   Does the answer address the question?

G-Eval works by asking a powerful LLM to reason step-by-step about each
criterion, then assign a numeric score.  Scores are 0–1; higher is better.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams as LLMTestCaseParams

from property_intel.evaluation.dataset import EvalDataset, EvalSample

_CORRECTNESS_CRITERIA = (
    "Determine whether the actual output is factually correct given the "
    "expected output.  The answer should contain the key facts from the "
    "expected output without introducing contradictions."
)

_RELEVANCY_CRITERIA = (
    "Determine whether the actual output directly and completely addresses "
    "the input question.  The answer should be on-topic and not introduce "
    "irrelevant information."
)


def _make_correctness_metric(model: str = "gpt-4o-mini", threshold: float = 0.5) -> GEval:
    return GEval(
        name="Correctness",
        criteria=_CORRECTNESS_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=threshold,
    )


def _make_relevancy_metric(model: str = "gpt-4o-mini", threshold: float = 0.5) -> GEval:
    return GEval(
        name="Answer Relevancy",
        criteria=_RELEVANCY_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
        threshold=threshold,
    )


class DeepEvaluator:
    """Evaluate LLM output quality using DeepEval G-Eval metrics.

    Usage (production):
        evaluator = DeepEvaluator(model="gpt-4o")
        scores = evaluator.evaluate(dataset)

    Usage (testing — no API calls):
        evaluator = DeepEvaluator()
        scores = evaluator.evaluate(dataset, _skip_compute=True)
    """

    METRIC_NAMES: list[str] = ["correctness", "answer_relevancy"]

    def __init__(self, model: str = "gpt-4o-mini", threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold

    def evaluate(
        self,
        dataset: EvalDataset,
        *,
        _skip_compute: bool = False,
    ) -> dict[str, float]:
        """Score *dataset* with DeepEval G-Eval metrics.

        Args:
            dataset:       EvalDataset of question/answer/context triples.
            _skip_compute: Skip actual LLM scoring (for unit tests).

        Returns:
            Dict mapping metric name to mean score across the dataset.
        """
        if len(dataset) == 0:
            return {m: 0.0 for m in self.METRIC_NAMES}

        if _skip_compute:
            return {m: float("nan") for m in self.METRIC_NAMES}

        correctness_metric = _make_correctness_metric(self._model, self._threshold)
        relevancy_metric = _make_relevancy_metric(self._model, self._threshold)

        correctness_scores: list[float] = []
        relevancy_scores: list[float] = []

        for sample in dataset:
            test_case = LLMTestCase(
                input=sample.question,
                actual_output=sample.answer,
                expected_output=sample.ground_truth,
                retrieval_context=list(sample.contexts),  # type: ignore[arg-type]
            )
            correctness_metric.measure(test_case)
            relevancy_metric.measure(test_case)
            correctness_scores.append(correctness_metric.score or 0.0)
            relevancy_scores.append(relevancy_metric.score or 0.0)

        return {
            "correctness": sum(correctness_scores) / len(correctness_scores),
            "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores),
        }

    def available_metrics(self) -> list[str]:
        return list(self.METRIC_NAMES)

    def make_test_case(self, sample: EvalSample) -> LLMTestCase:
        """Convert an EvalSample to a DeepEval LLMTestCase."""
        return LLMTestCase(
            input=sample.question,
            actual_output=sample.answer,
            expected_output=sample.ground_truth,
            retrieval_context=list(sample.contexts),  # type: ignore[arg-type]
        )
