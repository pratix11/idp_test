from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from property_intel.db.models import DocumentModel
from property_intel.search.schema import SearchFilters, SearchQuery, SearchResultItem, SearchResultPage


class FullTextSearch:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search(self, query: SearchQuery) -> SearchResultPage:
        text = (query.text or "").strip()
        if not text:
            return SearchResultPage(items=[], total=0, page=query.page, page_size=query.page_size)

        tsquery = func.websearch_to_tsquery("english", text)
        rank = func.ts_rank(DocumentModel.search_vector, tsquery)
        snippet = func.ts_headline(
            "english",
            func.coalesce(DocumentModel.content, ""),
            tsquery,
            "MaxFragments=1, MaxWords=35, MinWords=15",
        )

        base_stmt = select(DocumentModel).where(DocumentModel.search_vector.op("@@")(tsquery))
        base_stmt = self._apply_filters(base_stmt, query.filters)

        total = self.session.scalar(select(func.count()).select_from(base_stmt.subquery())) or 0

        rows = self.session.execute(
            base_stmt.add_columns(rank.label("score"), snippet.label("snippet"))
            .order_by(rank.desc(), DocumentModel.date.desc())
            .limit(query.page_size)
            .offset(query.offset)
        ).all()

        items = [
            SearchResultItem(
                document_id=doc.id,
                title=doc.title,
                category=doc.category,
                source=doc.source,
                document_type=doc.document_type,
                date=doc.date,
                score=score,
                snippet=snippet_text,
            )
            for doc, score, snippet_text in rows
        ]

        return SearchResultPage(items=items, total=total, page=query.page, page_size=query.page_size)

    @staticmethod
    def _apply_filters(stmt: Any, filters: SearchFilters) -> Any:
        if filters.category is not None:
            stmt = stmt.where(DocumentModel.category == filters.category.value)
        if filters.source is not None:
            stmt = stmt.where(DocumentModel.source == filters.source)
        if filters.document_type is not None:
            stmt = stmt.where(DocumentModel.document_type == filters.document_type)
        if filters.date_from is not None:
            stmt = stmt.where(DocumentModel.date >= filters.date_from)
        if filters.date_to is not None:
            stmt = stmt.where(DocumentModel.date <= filters.date_to)
        return stmt
