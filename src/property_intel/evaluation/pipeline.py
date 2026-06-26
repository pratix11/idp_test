"""EvaluationPipeline — stub, full implementation in Task 46."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationReport:
    """Aggregated scores from all evaluators."""

    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.scores:
            return "No scores computed."
        lines = [f"  {k}: {v:.3f}" for k, v in sorted(self.scores.items())]
        return "Evaluation Report:\n" + "\n".join(lines)


class EvaluationPipeline:
    """Orchestrates all Phase 6 evaluators. Full implementation in Task 46."""

    def __init__(self, evaluators: list[Any] | None = None) -> None:
        self._evaluators: list[Any] = list(evaluators or [])

    def add_evaluator(self, evaluator: Any) -> None:
        self._evaluators.append(evaluator)

    def run(self, dataset: Any) -> EvaluationReport:
        raise NotImplementedError("Full pipeline implemented in Task 46.")
