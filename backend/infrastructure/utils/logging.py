import logging
from datetime import datetime
import os


def setup_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Setup logger with console and optional file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clean formatter with date/time but no microseconds
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_to_file:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # File handler
        fh = logging.FileHandler(f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
