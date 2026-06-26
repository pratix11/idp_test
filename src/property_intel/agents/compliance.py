"""Compliance Agent — Phase 5.

Capability: Validate whether a described situation complies with regulatory
requirements retrieved from the document corpus.

Output format:
- Verdict: Compliant / Non-Compliant / Needs Review
- Applicable rules (cited from retrieved documents)
- Violations or gaps found
- Recommended actions
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from property_intel.agents.base import AgentState, BaseAgent
from property_intel.agents.tools import make_extract_tool, make_retrieve_tool
from property_intel.retrieval.service import RetrievalService

_SYSTEM_PROMPT = """\
You are a Compliance Verification Agent for Indian property regulations.

Given a description of a situation, action, or document, you must:
1. Retrieve the applicable regulatory requirements using retrieve_documents or
   extract_entities.
2. Check the situation against those requirements.
3. Output a structured compliance verdict.

Your verdict must be one of:
- COMPLIANT: All checked requirements are satisfied.
- NON-COMPLIANT: One or more requirements are violated.
- NEEDS REVIEW: Insufficient information to determine compliance.

Always cite the specific rules or sections you are checking against.
"""


class ComplianceAgent(BaseAgent):
    """Validate regulatory compliance of a described situation or document."""

    def _build_graph(self) -> Any:
        tool_node = self._make_tool_node()
        model_with_tools = self._bind_tools_to_llm()

        def call_agent(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        return self._build_standard_graph(call_agent, tool_node, AgentState)

    def run(self, task: str, **metadata: Any) -> str:
        """Validate *task* (a situation description) for regulatory compliance."""
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

    def validate(self, situation: str) -> str:
        """Check whether *situation* is compliant with applicable regulations."""
        task = (
            f"Validate the following situation for regulatory compliance:\n\n{situation}\n\n"
            "Retrieve the applicable rules, check each requirement, and provide a verdict."
        )
        return self.run(task)

    @classmethod
    def from_retrieval(cls, retrieval: RetrievalService, llm: Any) -> ComplianceAgent:
        """Factory: build compliance agent from a RetrievalService instance."""
        tools = [
            make_retrieve_tool(retrieval),
            make_extract_tool(retrieval),
        ]
        return cls(tools=tools, llm=llm)
