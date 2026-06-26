"""Prompt templates for the AI Copilot.

Each function takes structured inputs and returns a list of Message dicts
ready to be passed directly to LLMClient.complete() or stream_complete().

Three modes:
  - qa:        Answer a question from retrieved context with citations.
  - summarize: Summarise a set of retrieved chunks.
  - compare:   Compare two sets of retrieved chunks and highlight differences.
"""

from __future__ import annotations

from property_intel.copilot.context_builder import BuiltContext
from property_intel.copilot.llm_client import Message

_QA_SYSTEM = """\
You are an expert assistant for Indian property and regulatory documents.
Answer the user's question using ONLY the provided document excerpts.
Cite sources inline using [N] notation matching the excerpt numbers.
When a document name is available in the excerpt header (e.g. "Document: Real_Estate_Act_2016"), reference it by name alongside the citation number, e.g. "Real Estate Act 2016 [1]".
If the answer is not found in the excerpts, say "I could not find an answer in the provided documents."
Be concise and precise. Do not speculate beyond what the documents state."""

_SUMMARIZE_SYSTEM = """\
You are an expert assistant for Indian property and regulatory documents.
Produce a clear, structured summary of the provided document excerpts.
Use bullet points for key points. Cite sources inline using [N] notation.
Do not add information not present in the excerpts."""

_COMPARE_SYSTEM = """\
You are an expert assistant for Indian property and regulatory documents.
Compare the two sets of document excerpts provided (Set A and Set B).
Highlight similarities, differences, and any conflicting requirements.
Cite sources inline using [N] for Set A and [N-B] for Set B.
Do not add information not present in the excerpts."""


def build_qa_prompt(question: str, context: BuiltContext) -> list[Message]:
    """Prompt for answering a question from retrieved context."""
    user_content = (
        f"Document excerpts:\n\n{context.context}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": _QA_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_summarize_prompt(context: BuiltContext) -> list[Message]:
    """Prompt for summarising a set of retrieved chunks."""
    user_content = f"Document excerpts to summarise:\n\n{context.context}"
    return [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_compare_prompt(context_a: BuiltContext, context_b: BuiltContext) -> list[Message]:
    """Prompt for comparing two sets of retrieved chunks."""
    user_content = (
        f"Set A excerpts:\n\n{context_a.context}\n\n"
        f"Set B excerpts:\n\n{context_b.context}\n\n"
        "Compare these two sets and highlight key similarities and differences."
    )
    return [
        {"role": "system", "content": _COMPARE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
