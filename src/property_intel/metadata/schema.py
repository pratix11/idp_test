from datetime import date as date_type
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator


class DocumentCategory(str, Enum):
    ACTS = "acts"
    CIRCULARS = "circulars"
    ORDERS = "orders"
    REGULATIONS = "regulations"
    REPORTS = "reports"
    RULES = "rules"


class DocumentMetadata(BaseModel):
    title: str
    source: str
    category: DocumentCategory
    document_type: str
    date: date_type
    pages: int
    file_path: Path
    markdown_path: Path | None = None

    @field_validator("title", "source", "document_type")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("date")
    @classmethod
    def not_in_future(cls, value: date_type) -> date_type:
        if value > datetime.now().date():
            raise ValueError("date must not be in the future")
        return value

    @field_validator("pages")
    @classmethod
    def positive_page_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("pages must be a positive integer")
        return value

    @field_validator("file_path")
    @classmethod
    def file_path_must_exist_and_be_pdf(cls, value: Path) -> Path:
        if value.suffix.lower() != ".pdf":
            raise ValueError(f"file_path must point to a .pdf file, got: {value}")
        if not value.exists():
            raise ValueError(f"file_path does not exist: {value}")
        return value

    @field_validator("markdown_path")
    @classmethod
    def markdown_path_must_be_md(cls, value: Path | None) -> Path | None:
        if value is not None and value.suffix.lower() != ".md":
            raise ValueError(f"markdown_path must point to a .md file, got: {value}")
        return value
