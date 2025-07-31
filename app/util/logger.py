import os
import sys
from pathlib import Path            
import logging
from logging.handlers import RotatingFileHandler
import io

# Add project root to the Python path   
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOG_LEVEL = "INFO"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "harupyquant.log"


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
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    logger.setLevel(LOG_LEVEL)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler with Unicode support
    if os.name == 'nt':  # Windows
        # Use a custom stream handler for Windows to handle Unicode properly
        class UnicodeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Encode to UTF-8 and decode to handle Unicode properly
                    encoded_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
                    stream = self.stream
                    stream.write(encoded_msg)
                    stream.write(self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        ch = UnicodeStreamHandler(sys.stdout)
    else:
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