import logging

from property_intel.logging_setup import configure_logging, get_logger


def test_log_file_created(tmp_path) -> None:
    logger = configure_logging(tmp_path, logger_name="test_creation")
    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()

    assert (tmp_path / "test_creation.log").exists()
    assert (tmp_path / "test_creation.error.log").exists()


def test_error_logging_goes_to_error_file(tmp_path) -> None:
    logger = configure_logging(tmp_path, logger_name="test_errors")
    logger.info("just info, should not appear in error log")
    logger.error("boom, should appear in error log")
    for handler in logger.handlers:
        handler.flush()

    error_log = (tmp_path / "test_errors.error.log").read_text()
    app_log = (tmp_path / "test_errors.log").read_text()

    assert "boom" in error_log
    assert "just info" not in error_log
    assert "boom" in app_log
    assert "just info" in app_log


def test_rotation_behavior(tmp_path) -> None:
    logger = configure_logging(
        tmp_path,
        logger_name="test_rotation",
        max_bytes=200,
        backup_count=3,
    )
    for i in range(200):
        logger.info("padding message number %d to force rotation", i)
    for handler in logger.handlers:
        handler.flush()

    rotated = list(tmp_path.glob("test_rotation.log.*"))
    assert rotated, "expected at least one rotated backup file"


def test_idempotent_configuration_does_not_duplicate_handlers(tmp_path) -> None:
    logger1 = configure_logging(tmp_path, logger_name="test_idempotent")
    handler_count_after_first = len(logger1.handlers)

    logger2 = configure_logging(tmp_path, logger_name="test_idempotent", level="DEBUG")

    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count_after_first
    assert logger2.level == logging.DEBUG


def test_get_logger_returns_child_of_base() -> None:
    child = get_logger("ingestion.organizer")
    assert child.name == "property_intel.ingestion.organizer"


def test_get_logger_no_name_returns_base() -> None:
    base = get_logger()
    assert base.name == "property_intel"
