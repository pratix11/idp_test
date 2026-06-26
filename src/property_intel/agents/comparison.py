"""Regulation Comparison Agent — Phase 5.

Capabilities:
- Compare two versions of regulations and highlight differences.
- Detect material changes (new rules, repealed clauses, amended penalties).

The agent retrieves context for both subjects separately, then uses the LLM
to produce a structured comparison.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from property_intel.agents.base import AgentState, BaseAgent
from property_intel.agents.tools import make_retrieve_tool
from property_intel.retrieval.service import RetrievalService

_SYSTEM_PROMPT = """\
You are a Regulation Comparison Specialist for Indian property law.

Your task is to compare two sets of regulatory documents and identify differences.
You have access to retrieve_documents — use it twice: once for each subject.

Structure your output as:
1. Summary of Subject A
2. Summary of Subject B
3. Key Differences (bullet list)
4. Material Changes (new rules, deleted rules, amended penalties)

Be precise: quote or paraphrase specific clauses, cite section numbers where available.
"""


class ComparisonAgent(BaseAgent):
    """Compare two regulatory topics and detect changes between them."""

    def _build_graph(self) -> Any:
        tool_node = self._make_tool_node()
        model_with_tools = self._bind_tools_to_llm()

        def call_agent(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        return self._build_standard_graph(call_agent, tool_node, AgentState)

    def run(self, task: str, **metadata: Any) -> str:
        """Compare the two subjects described in *task*."""
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=task),
            ],
            "task": task,
            "result": None,
            "metadata": dict(metadata),
        }
        final_state = self._graph.invoke(initial_state)
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return str(msg.content)
        return ""

    def compare(self, subject_a: str, subject_b: str) -> str:
        """Convenience method: compare *subject_a* against *subject_b*."""
        task = (
            f"Compare the regulatory documents about '{subject_a}' with those about "
            f"'{subject_b}'. Identify key differences and material changes."
        )
        return self.run(task)

    @classmethod
    def from_retrieval(cls, retrieval: RetrievalService, llm: Any) -> ComparisonAgent:
        """Factory: build comparison agent from a RetrievalService instance."""
        tools = [make_retrieve_tool(retrieval)]
        return cls(tools=tools, llm=llm)
