import argparse

from property_intel.db.session import get_engine, get_session_factory, init_db
from property_intel.retrieval.service import RetrievalService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic / hybrid search over indexed chunks")
    sub = parser.add_subparsers(dest="command", required=True)

    # search sub-command
    search_p = sub.add_parser("search", help="Query the retrieval index")
    search_p.add_argument("query", help="Natural-language query")
    search_p.add_argument("--limit", type=int, default=10)
    search_p.add_argument("--mode", choices=["semantic", "hybrid"], default="hybrid")
    search_p.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")

    # index sub-command
    index_p = sub.add_parser("index", help="Chunk + embed all completed documents")
    index_p.add_argument("--reindex", action="store_true", help="Delete existing chunks and rebuild")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    engine = get_engine()
    init_db(engine)
    session = get_session_factory(engine)()
    svc = RetrievalService.from_settings(session)

    if args.command == "index":
        counts = svc.index_documents(reindex=args.reindex)
        print(f"Indexed {counts['documents']} documents -> {counts['chunks']} chunks")

    elif args.command == "search":
        results = svc.search(
            args.query,
            limit=args.limit,
            mode=args.mode,
            rerank=not args.no_rerank,
        )
        print(f"{len(results)} result(s) for '{args.query}' [{args.mode}]")
        for rank, chunk in enumerate(results, start=1):
            title_hint = f" [{chunk.section_title}]" if chunk.section_title else ""
            print(f"  {rank}. doc={chunk.document_id} chunk={chunk.chunk_index}{title_hint}  score={chunk.score:.4f}")
            print(f"     {chunk.content[:120].replace(chr(10), ' ')}...")

    session.close()


if __name__ == "__main__":
    main()
