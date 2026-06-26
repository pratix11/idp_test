"""Base agent state and abstract base class for all Phase 5 agents."""

from __future__ import annotations

import abc
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through every node in the agent graph.

    messages: Conversation history (tool calls, tool results, AI responses).
              The add_messages reducer appends rather than overwrites.
    task:     The original user request — preserved for context throughout.
    result:   Final output text once the agent concludes.
    metadata: Agent-specific key/value data (doc IDs, compliance flags, etc.).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    result: str | None
    metadata: dict[str, Any]


class BaseAgent(abc.ABC):
    """Abstract base for all Phase 5 LangGraph agents.

    Subclasses implement `_build_graph` which returns a compiled LangGraph
    graph.  The public `run` method feeds the task into the graph and
    extracts the final text result.
    """

    def __init__(self, tools: list[Any], llm: Any) -> None:
        self._tools = tools
        self._llm = llm
        self._graph = self._build_graph()

    @abc.abstractmethod
    def _build_graph(self) -> Any:  # CompiledGraph
        """Build and compile the agent's LangGraph StateGraph."""

    def run(self, task: str, **metadata: Any) -> str:
        """Execute the agent on *task* and return the final text result."""
        initial_state: AgentState = {
            "messages": [],
            "task": task,
            "result": None,
            "metadata": dict(metadata),
        }
        final_state = self._graph.invoke(initial_state)
        return final_state.get("result") or ""

    # ── helpers shared by all agent graph builders ─────────────────────────

    def _make_tool_node(self) -> ToolNode:
        return ToolNode(self._tools)

    def _bind_tools_to_llm(self) -> Any:
        return self._llm.bind_tools(self._tools)

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        """Edge function: route to 'tools' if the LLM made a tool call, else END."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    @staticmethod
    def _build_standard_graph(
        agent_node: Any,
        tool_node: ToolNode,
        state_cls: type,
    ) -> Any:
        """Wire the standard ReAct loop: START → agent ⇄ tools → END."""
        graph: StateGraph = StateGraph(state_cls)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            BaseAgent._should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile()
