from enum import Enum


class DocumentState(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InvalidTransitionError(Exception):
    pass


class DuplicateDocumentError(Exception):
    pass


VALID_TRANSITIONS: dict[DocumentState, set[DocumentState]] = {
    DocumentState.UPLOADED: {DocumentState.PROCESSING},
    DocumentState.PROCESSING: {DocumentState.COMPLETED, DocumentState.FAILED},
    DocumentState.FAILED: {DocumentState.PROCESSING},
    DocumentState.COMPLETED: set(),
}


def validate_transition(current: DocumentState, target: DocumentState) -> None:
    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition from {current.value!r} to {target.value!r}"
        )


class DocumentRegistry:
    """In-memory tracker of document lifecycle state, keyed by a caller-supplied
    identifier (e.g. content hash or file path). Persistent storage (Task 8) reuses
    the same `validate_transition` rules against a database-backed state column.
    """

    def __init__(self) -> None:
        self._states: dict[str, DocumentState] = {}

    def register(self, identifier: str) -> DocumentState:
        if identifier in self._states:
            raise DuplicateDocumentError(f"Document {identifier!r} is already registered")
        self._states[identifier] = DocumentState.UPLOADED
        return DocumentState.UPLOADED

    def get_state(self, identifier: str) -> DocumentState:
        if identifier not in self._states:
            raise KeyError(identifier)
        return self._states[identifier]

    def transition(self, identifier: str, target: DocumentState) -> DocumentState:
        current = self.get_state(identifier)
        validate_transition(current, target)
        self._states[identifier] = target
        return target
