from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from property_intel.db.models import ChunkModel, DocumentModel
from property_intel.registry.state_machine import DocumentState, DuplicateDocumentError, validate_transition
from property_intel.retrieval.models import DocumentChunk


class DocumentNotFoundError(Exception):
    pass


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **fields: object) -> DocumentModel:
        document = DocumentModel(**fields)
        self.session.add(document)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise DuplicateDocumentError(
                f"Document with this file_path or content_hash already exists: {exc}"
            ) from exc
        self.session.refresh(document)
        return document

    def get_by_id(self, document_id: int) -> DocumentModel:
        document = self.session.get(DocumentModel, document_id)
        if document is None:
            raise DocumentNotFoundError(f"No document with id={document_id}")
        return document

    def get_by_file_path(self, file_path: str) -> DocumentModel | None:
        stmt = select(DocumentModel).where(DocumentModel.file_path == file_path)
        return self.session.scalars(stmt).first()

    def get_by_content_hash(self, content_hash: str) -> DocumentModel | None:
        stmt = select(DocumentModel).where(DocumentModel.content_hash == content_hash)
        return self.session.scalars(stmt).first()

    def update(self, document_id: int, **fields: object) -> DocumentModel:
        document = self.get_by_id(document_id)
        for key, value in fields.items():
            setattr(document, key, value)
        self.session.commit()
        self.session.refresh(document)
        return document

    def update_state(
        self, document_id: int, target: DocumentState, error_message: str | None = None
    ) -> DocumentModel:
        document = self.get_by_id(document_id)
        current = DocumentState(document.state)
        validate_transition(current, target)
        document.state = target.value
        if error_message is not None:
            document.error_message = error_message
        self.session.commit()
        self.session.refresh(document)
        return document

    def delete(self, document_id: int) -> None:
        document = self.get_by_id(document_id)
        self.session.delete(document)
        self.session.commit()

    def list_all(self) -> list[DocumentModel]:
        return list(self.session.scalars(select(DocumentModel)).all())

    def list_searchable(self) -> list[DocumentModel]:
        stmt = select(DocumentModel).where(
            DocumentModel.state == DocumentState.COMPLETED.value,
            DocumentModel.content.is_not(None),
        )
        return list(self.session.scalars(stmt).all())


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_create(self, chunks: list[DocumentChunk]) -> list[ChunkModel]:
        models = [
            ChunkModel(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                content=c.content,
                token_count=c.token_count,
                section_title=c.section_title,
            )
            for c in chunks
        ]
        self._session.add_all(models)
        self._session.commit()
        for m in models:
            self._session.refresh(m)
        return models

    def get_by_document_id(self, document_id: int) -> list[ChunkModel]:
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return list(self._session.scalars(stmt).all())

    def delete_by_document_id(self, document_id: int) -> None:
        self._session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        self._session.commit()

    def count_by_document_id(self, document_id: int) -> int:
        stmt = select(func.count()).select_from(ChunkModel).where(
            ChunkModel.document_id == document_id
        )
        return self._session.scalar(stmt) or 0
