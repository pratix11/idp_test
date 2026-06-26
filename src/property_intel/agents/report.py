"""Report Agent — Phase 5.

Capability: Generate structured professional reports from regulatory documents.

The report includes: Executive Summary, Background, Key Findings,
Applicable Regulations, Risk Assessment, and Recommendations.
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
You are a Report Generation Agent specialising in Indian property regulation.

Your task is to generate a professional, structured regulatory report.

Steps:
1. Use retrieve_documents to gather relevant regulatory passages.
2. Use summarize_topic to build an overview of the regulatory area.
3. Use extract_entities to pull out specific rules, deadlines, penalties, and parties.
4. Draft the report.

Your report must include these sections:
## Executive Summary
## Background
## Key Findings
## Applicable Regulations
## Risk Assessment
## Recommendations

Use precise language. Cite specific sections and rules where available.
Avoid speculation — base all findings on retrieved document content.
"""


class ReportAgent(BaseAgent):
    """Generate structured regulatory reports from the document corpus."""

    def _build_graph(self) -> Any:
        tool_node = self._make_tool_node()
        model_with_tools = self._bind_tools_to_llm()

        def call_agent(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        return self._build_standard_graph(call_agent, tool_node, AgentState)

    def run(self, task: str, **metadata: Any) -> str:
        """Generate a report for *task* and return the report text."""
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

    def generate(self, topic: str, scope: str = "") -> str:
        """Generate a report on *topic* with optional *scope* constraints."""
        scope_clause = f" Focus on: {scope}." if scope else ""
        task = (
            f"Generate a comprehensive regulatory report on the following topic:\n\n"
            f"{topic}{scope_clause}\n\n"
            f"Include all required sections: Executive Summary, Background, "
            f"Key Findings, Applicable Regulations, Risk Assessment, Recommendations."
        )
        return self.run(task)

    @classmethod
    def from_retrieval(cls, retrieval: RetrievalService, llm: Any) -> ReportAgent:
        """Factory: build report agent from a RetrievalService instance."""
        tools = [
            make_retrieve_tool(retrieval),
            make_summarize_tool(retrieval),
            make_extract_tool(retrieval),
        ]
        return cls(tools=tools, llm=llm)
