from datetime import date as date_type
from datetime import datetime

from sqlalchemy import Date, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from property_intel.db.base import Base
from property_intel.registry.state_machine import DocumentState


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(64))
    document_type: Mapped[str] = mapped_column(String(128))
    date: Mapped[date_type] = mapped_column(Date)
    pages: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True)
    markdown_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(32), default=DocumentState.UPLOADED.value)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
