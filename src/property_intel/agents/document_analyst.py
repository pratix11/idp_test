"""Document Analyst Agent — Phase 5.

Capabilities:
- Summarize: produce a coherent overview of a topic from retrieved chunks.
- Extract: pull out structured facts (rules, deadlines, amounts, parties).

The agent runs a standard ReAct loop: LLM decides to call a tool → tool runs
→ LLM sees the result → continues until it produces a final answer.
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
You are a Document Analyst specialising in Indian property regulation.

Your job is to analyse regulatory documents and produce clear, accurate answers.
You have access to three tools:
- retrieve_documents: search the corpus for relevant passages
- summarize_topic: retrieve and condense information on a topic
- extract_entities: pull out rules, deadlines, amounts, and parties

Always start by retrieving relevant documents before answering.
Be precise: cite specific rules, section numbers, or amounts when available.
When you have enough information, write your final answer — do NOT call any
more tools once you are ready to answer.
"""


class DocumentAnalystAgent(BaseAgent):
    """Analyse, summarise, and extract facts from regulatory documents."""

    def _build_graph(self) -> Any:
        tool_node = self._make_tool_node()
        model_with_tools = self._bind_tools_to_llm()

        def call_agent(state: AgentState) -> dict[str, Any]:
            # Prepend system prompt on the first call (when only the task is there)
            messages = state["messages"]
            if not messages:
                messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=state["task"]),
                ]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        def extract_result(state: AgentState) -> dict[str, Any]:
            last = state["messages"][-1]
            return {"result": getattr(last, "content", "") or ""}

        # Wire standard ReAct loop then extract final answer
        graph = self._build_standard_graph(call_agent, tool_node, AgentState)
        return graph

    def run(self, task: str, **metadata: Any) -> str:
        """Run the analyst agent on *task* and return the final answer."""
        from langchain_core.messages import HumanMessage, SystemMessage

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
        # Extract final AI message content
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return str(msg.content)
        return ""

    @classmethod
    def from_retrieval(cls, retrieval: RetrievalService, llm: Any) -> DocumentAnalystAgent:
        """Factory: build agent from a RetrievalService instance."""
        tools = [
            make_retrieve_tool(retrieval),
            make_summarize_tool(retrieval),
            make_extract_tool(retrieval),
        ]
        return cls(tools=tools, llm=llm)
