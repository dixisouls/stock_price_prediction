"""
Logging configuration for the Bitcoin price prediction model.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

from bitcoin_prediction.config import get_config


def setup_logger(
    name: str = "bitcoin_prediction", log_to_file: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Name of the logger.
        log_to_file: Whether to log to a file.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    # Set logging level
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if needed
    if log_to_file:
        config = get_config()
        log_dir = Path(config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create a default logger for the package
logger = setup_logger()
