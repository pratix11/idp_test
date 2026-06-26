"""Research Agent — Phase 5.

Capability: Multi-document research — answer complex questions by running
multiple retrieval queries across the corpus and synthesising findings.

Use this agent when a question spans many documents or requires broad
coverage, not just the best matching chunk.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from property_intel.agents.base import AgentState, BaseAgent
from property_intel.agents.tools import (
    make_extract_tool,
    make_retrieve_tool,
    make_summarize_tool,
)
from property_intel.retrieval.service import RetrievalService

_SYSTEM_PROMPT = """\
You are a Research Agent specialising in Indian property regulation.

Your task is to conduct multi-document research and produce a comprehensive
research brief. You should:
1. Break the research question into sub-topics.
2. Use retrieve_documents or summarize_topic MULTIPLE TIMES — once per
   sub-topic — to gather broad coverage.
3. Use extract_entities to pull out key facts, dates, and amounts.
4. Synthesise all gathered information into a well-structured research brief.

Your output must include:
- Research Question (restate it)
- Key Findings (bullet list, grouped by sub-topic)
- Relevant Rules and Requirements
- Gaps or Uncertainties
- Conclusion

Do not stop after a single retrieval — thorough research requires multiple queries.
"""


class ResearchAgent(BaseAgent):
    """Conduct multi-document research across the regulatory corpus."""

    def _build_graph(self) -> Any:
        tool_node = self._make_tool_node()
        model_with_tools = self._bind_tools_to_llm()

        def call_agent(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        return self._build_standard_graph(call_agent, tool_node, AgentState)

    def run(self, task: str, **metadata: Any) -> str:
        """Research *task* across multiple documents and return a research brief."""
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

    def research(self, question: str) -> str:
        """Convenience method: conduct research on *question* and return brief."""
        task = (
            f"Research the following question across all available regulatory documents:\n\n"
            f"{question}\n\nProvide a comprehensive research brief."
        )
        return self.run(task)

    @classmethod
    def from_retrieval(cls, retrieval: RetrievalService, llm: Any) -> ResearchAgent:
        """Factory: build research agent from a RetrievalService instance."""
        tools = [
            make_retrieve_tool(retrieval),
            make_summarize_tool(retrieval),
            make_extract_tool(retrieval),
        ]
        return cls(tools=tools, llm=llm)
