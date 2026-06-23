from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from property_intel.db.models import DocumentModel
from property_intel.search.schema import SearchQuery, SearchResultItem, SearchResultPage


class MetadataSearch:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search(self, query: SearchQuery) -> SearchResultPage:
        base_stmt = self._apply_filters(select(DocumentModel), query)

        total = self.session.scalar(select(func.count()).select_from(base_stmt.subquery())) or 0

        rows = self.session.scalars(
            base_stmt.order_by(DocumentModel.date.desc(), DocumentModel.id.desc())
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
            )
            for doc in rows
        ]
        return SearchResultPage(items=items, total=total, page=query.page, page_size=query.page_size)

    @staticmethod
    def _apply_filters(stmt: Any, query: SearchQuery) -> Any:
        filters = query.filters
        title = (query.text or "").strip()
        if title:
            stmt = stmt.where(DocumentModel.title.ilike(f"%{title}%"))
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
