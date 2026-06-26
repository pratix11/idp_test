"""Shared LangGraph tools for Phase 5 agents.

Tools are plain Python functions decorated with @tool.  The LLM sees their
docstring and parameter types as its "instruction manual" — that's how it
knows when and how to call them.

All tools that touch retrieval accept an injected RetrievalService so they
can be easily replaced with mocks in tests.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from property_intel.retrieval.service import RetrievalService


def make_retrieve_tool(retrieval: RetrievalService) -> Any:
    """Return a @tool that searches the vector store via *retrieval*."""

    @tool
    def retrieve_documents(query: str, limit: int = 5) -> str:
        """Search the document corpus for chunks relevant to *query*.

        Use this whenever you need factual information from the regulatory
        documents to answer a question or complete a task.

        Args:
            query: Natural language search query.
            limit: Maximum number of chunks to return (default 5).

        Returns:
            Numbered list of relevant text chunks with their source sections.
        """
        chunks = retrieval.search(query, limit=limit)
        if not chunks:
            return "No relevant documents found."
        lines: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            section = chunk.section_title or "—"
            lines.append(f"[{i}] Section: {section}\n{chunk.content}")
        return "\n\n".join(lines)

    return retrieve_documents


def make_summarize_tool(retrieval: RetrievalService) -> Any:
    """Return a @tool that retrieves and summarises a topic."""

    @tool
    def summarize_topic(topic: str) -> str:
        """Retrieve and produce a concise summary of documents about *topic*.

        Use this when you need a high-level overview of a regulatory area
        rather than specific facts.

        Args:
            topic: The subject or regulatory topic to summarise.

        Returns:
            A summary paragraph drawn from retrieved document chunks.
        """
        chunks = retrieval.search(topic, limit=8)
        if not chunks:
            return "No documents found for this topic."
        combined = "\n\n".join(c.content for c in chunks)
        return f"Retrieved context for '{topic}':\n\n{combined}"

    return summarize_topic


def make_extract_tool(retrieval: RetrievalService) -> Any:
    """Return a @tool that retrieves structured facts about a topic."""

    @tool
    def extract_entities(query: str) -> str:
        """Extract key entities (rules, deadlines, amounts, parties) related to *query*.

        Use this when you need to pull out structured facts — numbers,
        dates, named parties, or specific requirements — from the documents.

        Args:
            query: The entity or fact type to search for.

        Returns:
            Bullet list of extracted facts from the document corpus.
        """
        chunks = retrieval.search(query, limit=6)
        if not chunks:
            return "No relevant entities found."
        combined = "\n\n".join(c.content for c in chunks)
        return f"Source passages for entity extraction on '{query}':\n\n{combined}"

    return extract_entities
