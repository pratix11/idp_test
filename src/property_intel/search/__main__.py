import argparse

from property_intel.db.session import get_engine, get_session_factory, init_db
from property_intel.search.schema import SearchFilters, SearchQuery
from property_intel.search.service import SearchMode, SearchService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the document corpus")
    parser.add_argument("text", nargs="?", default=None, help="Free-text query")
    parser.add_argument("--mode", choices=["fulltext", "bm25", "metadata"], default="fulltext")
    parser.add_argument("--category", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    engine = get_engine()
    init_db(engine)
    session = get_session_factory(engine)()

    query = SearchQuery(
        text=args.text,
        filters=SearchFilters(
            category=args.category,
            source=args.source,
            document_type=args.document_type,
        ),
        page=args.page,
        page_size=args.page_size,
    )
    mode: SearchMode = args.mode
    page = SearchService(session).search(query, mode=mode)

    print(f"{page.total} result(s), page {page.page}/{max(page.total_pages, 1)}")
    for item in page.items:
        score = f" (score={item.score:.3f})" if item.score is not None else ""
        print(f"[{item.document_id}] {item.title} - {item.category}/{item.source}{score}")
        if item.snippet:
            print(f"    {item.snippet}")

    session.close()


if __name__ == "__main__":
    main()
