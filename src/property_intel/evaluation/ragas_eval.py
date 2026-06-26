"""RagasEvaluator — Phase 6 RAGAS-based RAG quality metrics.

Metrics:
- faithfulness:       Does the answer only use facts from the retrieved context?
- answer_relevancy:   Does the answer address the question?
- context_precision:  Are the retrieved chunks relevant to the question?

Scores are 0–1; higher is better.

Note: ragas imports langchain_community.chat_models.vertexai which was removed
in langchain-community 0.3.0.  We patch a compatibility stub before importing
ragas so the module loads cleanly regardless of langchain-community version.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

# ── Compatibility shim for ragas → langchain-community 0.3+ ──────────────────
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = ModuleType("langchain_community.chat_models.vertexai")

    class _DummyChatVertexAI:  # minimal stub — never instantiated in practice
        pass

    _stub.ChatVertexAI = _DummyChatVertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# ─────────────────────────────────────────────────────────────────────────────

from ragas import evaluate as ragas_evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    faithfulness,
)

from property_intel.evaluation.dataset import EvalDataset


class RagasEvaluator:
    """Evaluate a RAG pipeline using RAGAS faithfulness, relevancy, and precision.

    Usage (production):
        evaluator = RagasEvaluator(llm=ChatOpenAI(...), embeddings=OpenAIEmbeddings())
        scores = evaluator.evaluate(dataset)

    Usage (testing):
        evaluator = RagasEvaluator()  # no llm/embeddings → metrics not computed
        scores = evaluator.evaluate(dataset, _skip_compute=True)
    """

    METRIC_NAMES: list[str] = ["faithfulness", "answer_relevancy", "context_precision"]

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings

    def evaluate(
        self,
        dataset: EvalDataset,
        *,
        _skip_compute: bool = False,
    ) -> dict[str, float]:
        """Score *dataset* with RAGAS metrics.

        Args:
            dataset:        EvalDataset of question/answer/context triples.
            _skip_compute:  If True, skip the actual LLM-based scoring and
                            return NaN scores. Used in unit tests.

        Returns:
            Dict mapping metric name to mean score across the dataset.
        """
        if len(dataset) == 0:
            return {m: 0.0 for m in self.METRIC_NAMES}

        if _skip_compute:
            return {m: float("nan") for m in self.METRIC_NAMES}

        from datasets import Dataset as HFDataset

        hf_dataset = HFDataset.from_dict(dataset.to_ragas_dict())
        metrics = [faithfulness, answer_relevancy, context_precision]

        kwargs: dict[str, Any] = {"dataset": hf_dataset, "metrics": metrics}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        if self._embeddings is not None:
            kwargs["embeddings"] = self._embeddings

        result: Any = ragas_evaluate(**kwargs)
        return {
            "faithfulness": float(result["faithfulness"]),  # type: ignore[index]
            "answer_relevancy": float(result["answer_relevancy"]),  # type: ignore[index]
            "context_precision": float(result["context_precision"]),  # type: ignore[index]
        }

    def available_metrics(self) -> list[str]:
        return list(self.METRIC_NAMES)
