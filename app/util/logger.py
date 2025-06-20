import logging
import os
from logging.handlers import RotatingFileHandler
import sys

from app.config.constants import DEFAULT_LOG_LEVEL, LOG_DIR, LOG_FILE

LOG_LEVEL = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

def get_logger(name: str) -> logging.Logger:
    """
    Initializes and returns a logger with specified handlers and format.

    The logger is configured with a console handler and a rotating file handler.
    Log messages are directed to both the console and a log file (`harupyquant.log`)
    in the `logs` directory. The log level can be set via the `LOG_LEVEL`
    environment variable.

    Args:
        name (str): The name of the logger, typically `__name__`.

    Returns:
        logging.Logger: The configured logger instance.
    """
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=10*1024*1024, backupCount=5
    )  # 10 MB per file, 5 backups
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger 