from datetime import date as date_type

from pydantic import BaseModel, Field, model_validator

from property_intel.metadata.schema import DocumentCategory


class SearchFilters(BaseModel):
    category: DocumentCategory | None = None
    source: str | None = None
    document_type: str | None = None
    date_from: date_type | None = None
    date_to: date_type | None = None

    @model_validator(mode="after")
    def date_range_is_ordered(self) -> "SearchFilters":
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class SearchQuery(BaseModel):
    text: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    page: int = 1
    page_size: int = 20

    @model_validator(mode="after")
    def page_is_positive(self) -> "SearchQuery":
        if self.page < 1:
            raise ValueError("page must be >= 1")
        if not (1 <= self.page_size <= 100):
            raise ValueError("page_size must be between 1 and 100")
        return self

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class SearchResultItem(BaseModel):
    document_id: int
    title: str
    category: str
    source: str
    document_type: str
    date: date_type
    score: float | None = None
    snippet: str | None = None


class SearchResultPage(BaseModel):
    items: list[SearchResultItem]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        if self.total == 0:
            return 0
        return -(-self.total // self.page_size)
