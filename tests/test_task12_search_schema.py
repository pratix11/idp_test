from datetime import date

import pytest
from pydantic import ValidationError

from property_intel.search.schema import (
    SearchFilters,
    SearchQuery,
    SearchResultItem,
    SearchResultPage,
)


def test_default_query_is_valid() -> None:
    query = SearchQuery()

    assert query.page == 1
    assert query.page_size == 20
    assert query.offset == 0
    assert query.filters == SearchFilters()


def test_offset_computed_from_page_and_page_size() -> None:
    query = SearchQuery(page=3, page_size=10)

    assert query.offset == 20


def test_page_below_one_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(page=0)


def test_page_size_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(page_size=0)
    with pytest.raises(ValidationError):
        SearchQuery(page_size=101)


def test_filters_date_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(date_from=date(2024, 6, 1), date_to=date(2024, 1, 1))


def test_filters_date_range_equal_is_allowed() -> None:
    filters = SearchFilters(date_from=date(2024, 1, 1), date_to=date(2024, 1, 1))
    assert filters.date_from == filters.date_to


def test_filters_invalid_category_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(category="not-a-real-category")


def test_result_page_total_pages_computed() -> None:
    items = [
        SearchResultItem(
            document_id=1,
            title="Circular 1",
            category="circulars",
            source="maharera",
            document_type="circular",
            date=date(2024, 1, 1),
        )
    ]
    page = SearchResultPage(items=items, total=25, page=1, page_size=10)

    assert page.total_pages == 3


def test_result_page_total_pages_zero_when_empty() -> None:
    page = SearchResultPage(items=[], total=0, page=1, page_size=10)

    assert page.total_pages == 0
