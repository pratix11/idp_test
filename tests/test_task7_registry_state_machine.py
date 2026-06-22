import pytest

from property_intel.registry.state_machine import (
    DocumentRegistry,
    DocumentState,
    DuplicateDocumentError,
    InvalidTransitionError,
    validate_transition,
)


@pytest.mark.parametrize(
    "current,target",
    [
        (DocumentState.UPLOADED, DocumentState.PROCESSING),
        (DocumentState.PROCESSING, DocumentState.COMPLETED),
        (DocumentState.PROCESSING, DocumentState.FAILED),
        (DocumentState.FAILED, DocumentState.PROCESSING),
    ],
)
def test_valid_transitions_do_not_raise(current, target) -> None:
    validate_transition(current, target)


@pytest.mark.parametrize(
    "current,target",
    [
        (DocumentState.UPLOADED, DocumentState.COMPLETED),
        (DocumentState.UPLOADED, DocumentState.FAILED),
        (DocumentState.COMPLETED, DocumentState.PROCESSING),
        (DocumentState.COMPLETED, DocumentState.UPLOADED),
        (DocumentState.FAILED, DocumentState.COMPLETED),
        (DocumentState.PROCESSING, DocumentState.UPLOADED),
    ],
)
def test_invalid_transitions_raise(current, target) -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(current, target)


def test_registry_register_sets_uploaded_state() -> None:
    registry = DocumentRegistry()
    state = registry.register("doc-1")
    assert state == DocumentState.UPLOADED
    assert registry.get_state("doc-1") == DocumentState.UPLOADED


def test_registry_duplicate_registration_raises() -> None:
    registry = DocumentRegistry()
    registry.register("doc-1")
    with pytest.raises(DuplicateDocumentError):
        registry.register("doc-1")


def test_registry_full_lifecycle_transition() -> None:
    registry = DocumentRegistry()
    registry.register("doc-1")
    registry.transition("doc-1", DocumentState.PROCESSING)
    registry.transition("doc-1", DocumentState.COMPLETED)
    assert registry.get_state("doc-1") == DocumentState.COMPLETED


def test_registry_failure_then_retry() -> None:
    registry = DocumentRegistry()
    registry.register("doc-1")
    registry.transition("doc-1", DocumentState.PROCESSING)
    registry.transition("doc-1", DocumentState.FAILED)
    assert registry.get_state("doc-1") == DocumentState.FAILED

    registry.transition("doc-1", DocumentState.PROCESSING)
    registry.transition("doc-1", DocumentState.COMPLETED)
    assert registry.get_state("doc-1") == DocumentState.COMPLETED


def test_registry_invalid_transition_raises_and_state_unchanged() -> None:
    registry = DocumentRegistry()
    registry.register("doc-1")
    with pytest.raises(InvalidTransitionError):
        registry.transition("doc-1", DocumentState.COMPLETED)
    assert registry.get_state("doc-1") == DocumentState.UPLOADED


def test_registry_unknown_identifier_raises_key_error() -> None:
    registry = DocumentRegistry()
    with pytest.raises(KeyError):
        registry.get_state("does-not-exist")
