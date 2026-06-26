"""EvaluationPipeline — Phase 6 orchestrator for all evaluators.

Registers any number of evaluators (RAGAS, DeepEval, custom), runs them
over an EvalDataset, and produces a single EvaluationReport with merged scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Evaluator(Protocol):
    """Minimal interface any evaluator must satisfy."""

    def evaluate(self, dataset: Any, **kwargs: Any) -> dict[str, float]: ...

    def available_metrics(self) -> list[str]: ...


@dataclass
class EvaluationReport:
    """Aggregated scores from all evaluators.

    Attributes:
        scores:     Flat dict of metric_name → mean score (0–1).
        details:    Per-evaluator raw output (for debugging).
    """

    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary table."""
        if not self.scores:
            return "No scores computed."
        lines = ["Evaluation Report:", "-" * 36]
        for k, v in sorted(self.scores.items()):
            bar_len = int(v * 20) if 0 <= v <= 1 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {k:<28} {bar} {v:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"scores": self.scores, "details": self.details}

    def to_json(self, path: Path | str) -> None:
        """Write report to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path | str) -> EvaluationReport:
        with open(path) as f:
            data = json.load(f)
        return cls(scores=data.get("scores", {}), details=data.get("details", {}))

    def merge(self, other: EvaluationReport) -> EvaluationReport:
        """Return a new report combining scores from both reports."""
        merged_scores = {**self.scores, **other.scores}
        merged_details = {**self.details, **other.details}
        return EvaluationReport(scores=merged_scores, details=merged_details)


class EvaluationPipeline:
    """Orchestrate multiple evaluators over an EvalDataset.

    Usage:
        pipeline = EvaluationPipeline()
        pipeline.add_evaluator("ragas", RagasEvaluator(llm=..., embeddings=...))
        pipeline.add_evaluator("deepeval", DeepEvaluator(model="gpt-4o-mini"))
        report = pipeline.run(dataset)
        print(report.summary())
    """

    def __init__(self, evaluators: dict[str, Any] | None = None) -> None:
        self._evaluators: dict[str, Any] = dict(evaluators or {})

    def add_evaluator(self, name: str, evaluator: Any) -> None:
        """Register *evaluator* under *name*."""
        self._evaluators[name] = evaluator

    def evaluator_names(self) -> list[str]:
        return list(self._evaluators.keys())

    def run(
        self,
        dataset: Any,
        *,
        skip_compute: bool = False,
    ) -> EvaluationReport:
        """Run all registered evaluators on *dataset*.

        Args:
            dataset:      EvalDataset of question/answer/context triples.
            skip_compute: Pass _skip_compute=True to all evaluators (unit test mode).

        Returns:
            EvaluationReport with merged scores from all evaluators.
        """
        all_scores: dict[str, float] = {}
        all_details: dict[str, Any] = {}

        for name, evaluator in self._evaluators.items():
            try:
                scores = evaluator.evaluate(dataset, _skip_compute=skip_compute)
            except TypeError:
                # evaluator doesn't accept _skip_compute (e.g. custom evaluator)
                scores = evaluator.evaluate(dataset)

            all_scores.update(scores)
            all_details[name] = scores

        return EvaluationReport(scores=all_scores, details=all_details)
