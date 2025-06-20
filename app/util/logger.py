import logging
import os
from logging.handlers import RotatingFileHandler
import sys

LOG_DIR = "logs"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

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
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

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
    log_file = os.path.join(LOG_DIR, "harupyquant.log")
    fh = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )  # 10 MB per file, 5 backups
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger 