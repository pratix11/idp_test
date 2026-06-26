"""EvalDataset — the shared data model for all Phase 6 evaluators.

An EvalSample is a (question, answer, contexts, ground_truth) tuple — the
minimal unit that RAGAS, DeepEval, and our metrics all operate on.

An EvalDataset is an ordered collection of EvalSamples with helper methods
for loading, slicing, and converting to library-specific formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class EvalSample:
    """One question-answer-context triple for evaluation.

    Attributes:
        question:     The user's input question.
        answer:       The system's generated answer (from copilot or agent).
        contexts:     Retrieved text chunks that the answer was based on.
        ground_truth: Optional reference answer for correctness checks.
        metadata:     Arbitrary key/value data (doc_id, agent type, etc.).
    """

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalSample:
        return cls(
            question=data["question"],
            answer=data["answer"],
            contexts=data.get("contexts", []),
            ground_truth=data.get("ground_truth", ""),
            metadata=data.get("metadata", {}),
        )


class EvalDataset:
    """Ordered collection of EvalSamples with I/O and conversion helpers."""

    def __init__(self, samples: list[EvalSample] | None = None) -> None:
        self._samples: list[EvalSample] = list(samples or [])

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, sample: EvalSample) -> None:
        self._samples.append(sample)

    def extend(self, samples: list[EvalSample]) -> None:
        self._samples.extend(samples)

    # ── access ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self._samples[index]

    def subset(self, n: int) -> EvalDataset:
        """Return a new dataset with the first *n* samples."""
        return EvalDataset(self._samples[:n])

    # ── I/O ──────────────────────────────────────────────────────────────

    @classmethod
    def from_list(cls, records: list[dict[str, Any]]) -> EvalDataset:
        """Build a dataset from a list of dicts."""
        return cls([EvalSample.from_dict(r) for r in records])

    @classmethod
    def from_jsonl(cls, path: Path | str) -> EvalDataset:
        """Load samples from a JSONL file (one JSON object per line)."""
        samples: list[EvalSample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(EvalSample.from_dict(json.loads(line)))
        return cls(samples)

    def to_jsonl(self, path: Path | str) -> None:
        """Write all samples to a JSONL file."""
        with open(path, "w") as f:
            for s in self._samples:
                f.write(json.dumps(s.to_dict()) + "\n")

    # ── library conversions ───────────────────────────────────────────────

    def to_ragas_dict(self) -> dict[str, list[Any]]:
        """Convert to the columnar dict format expected by ragas.Dataset."""
        return {
            "question": [s.question for s in self._samples],
            "answer": [s.answer for s in self._samples],
            "contexts": [s.contexts for s in self._samples],
            "ground_truth": [s.ground_truth for s in self._samples],
        }

    def to_deepeval_test_cases(self) -> list[dict[str, Any]]:
        """Convert to list of dicts suitable for building DeepEval LLMTestCase."""
        return [
            {
                "input": s.question,
                "actual_output": s.answer,
                "retrieval_context": s.contexts,
                "expected_output": s.ground_truth,
            }
            for s in self._samples
        ]
