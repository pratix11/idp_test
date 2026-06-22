import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    log_dir: Path,
    level: str = "INFO",
    logger_name: str = "property_intel",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    if logger_name in _CONFIGURED_LOGGERS:
        logger.setLevel(level)
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    app_handler = RotatingFileHandler(
        log_dir / f"{logger_name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    app_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        log_dir / f"{logger_name}.error.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(logger_name)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger("property_intel")
    if name is None:
        return base
    return base.getChild(name)
