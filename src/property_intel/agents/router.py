"""AgentRouter — routes user intent to the right Phase 5 agent."""

from __future__ import annotations

from typing import Any


class AgentRouter:
    """Dispatch a user task to the appropriate specialist agent.

    Intent classification is keyword-based in this stub; a future version
    could use an LLM classifier.
    """

    _KEYWORDS: dict[str, list[str]] = {
        "document_analyst": ["summarize", "summary", "extract", "analyse", "analyze"],
        "comparison": ["compare", "comparison", "difference", "changes", "versus", "vs"],
        "compliance": ["comply", "compliance", "violation", "requirement", "valid"],
        "research": ["research", "multi", "multiple", "across", "corpus"],
        "report": ["report", "generate report", "draft report", "write report"],
    }

    def __init__(self, agents: dict[str, Any]) -> None:
        self._agents = agents

    def route(self, task: str) -> str:
        """Classify *task* and run it with the matching agent.

        Returns the agent's text result.
        """
        agent_name = self._classify(task)
        agent = self._agents.get(agent_name)
        if agent is None:
            raise KeyError(f"No agent registered for intent '{agent_name}'")
        return agent.run(task)

    def _classify(self, task: str) -> str:
        """Return agent key whose keywords best match *task*."""
        lower = task.lower()
        scores: dict[str, int] = {name: 0 for name in self._KEYWORDS}
        for name, keywords in self._KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    scores[name] += 1
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "document_analyst"
        return best
