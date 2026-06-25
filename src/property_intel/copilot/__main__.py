"""Copilot CLI — ask questions, summarise, or compare from the terminal.

Usage:
    uv run python -m property_intel.copilot ask "What are builder registration rules?"
    uv run python -m property_intel.copilot summarize "builder registration"
    uv run python -m property_intel.copilot compare "2020 regulations" "2023 regulations"
    uv run python -m property_intel.copilot ask --no-stream "..."
"""

from __future__ import annotations

import argparse
import sys

from property_intel.copilot.service import CopilotService
from property_intel.db.session import get_session_factory


def _get_service() -> CopilotService:
    session = get_session_factory()()
    return CopilotService.from_settings(session)


def _print_citations(citations: list[object]) -> None:
    if not citations:
        return
    print("\n--- Citations ---")
    for c in citations:
        from property_intel.copilot.context_builder import Citation
        assert isinstance(c, Citation)
        title = c.section_title or "—"
        print(f"  [{c.index}] doc_id={c.document_id}  section={title}")
        print(f"       {c.content_snippet[:100]}...")


def cmd_ask(args: argparse.Namespace) -> None:
    svc = _get_service()
    if args.stream:
        for chunk in svc.stream_ask(args.question):
            print(chunk, end="", flush=True)
        print()
    else:
        result = svc.ask(args.question)
        print(result.answer)
        _print_citations(result.citations)  # type: ignore[arg-type]


def cmd_summarize(args: argparse.Namespace) -> None:
    svc = _get_service()
    if args.stream:
        for chunk in svc.stream_summarize(args.query):
            print(chunk, end="", flush=True)
        print()
    else:
        result = svc.summarize(args.query)
        print(result.answer)
        _print_citations(result.citations)  # type: ignore[arg-type]


def cmd_compare(args: argparse.Namespace) -> None:
    svc = _get_service()
    if args.stream:
        for chunk in svc.stream_compare(args.query_a, args.query_b):
            print(chunk, end="", flush=True)
        print()
    else:
        result = svc.compare(args.query_a, args.query_b)
        print(result.answer)
        _print_citations(result.citations)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="property-intel-copilot",
        description="AI Copilot for property and regulatory documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask_p = sub.add_parser("ask", help="Answer a question.")
    ask_p.add_argument("question", help="The question to answer.")
    ask_p.add_argument("--no-stream", dest="stream", action="store_false", default=True)
    ask_p.set_defaults(func=cmd_ask)

    sum_p = sub.add_parser("summarize", help="Summarise documents.")
    sum_p.add_argument("query", help="Topic or keywords.")
    sum_p.add_argument("--no-stream", dest="stream", action="store_false", default=True)
    sum_p.set_defaults(func=cmd_summarize)

    cmp_p = sub.add_parser("compare", help="Compare two document sets.")
    cmp_p.add_argument("query_a", help="First topic.")
    cmp_p.add_argument("query_b", help="Second topic.")
    cmp_p.add_argument("--no-stream", dest="stream", action="store_false", default=True)
    cmp_p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
