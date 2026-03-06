from __future__ import annotations

import logging
from pathlib import Path

from config import LoggingConfig


def setup_logger(logging_config: LoggingConfig) -> logging.Logger:
    """Configure application logger with stream and optional file handlers."""
    logger = logging.getLogger("elephant_behavior")
    logger.setLevel(logging_config.level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logging_config.log_to_file:
        log_path = Path(logging_config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
