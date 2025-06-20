from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories
APP_DIR = PROJECT_ROOT / "app"
CONFIG_DIR = APP_DIR / "config"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = APP_DIR / "data"
TESTS_DIR = PROJECT_ROOT / "tests"

# Log settings
LOG_FILE = LOG_DIR / "harupyquant.log"
DEFAULT_LOG_LEVEL = "INFO"

# Trading constants (examples)
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = "H1" 