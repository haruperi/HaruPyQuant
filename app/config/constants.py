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

# Default path for the configuration file
DEFAULT_CONFIG_PATH = 'config.ini'

# Trading parameters
MAX_DEVIATION = 5  # Maximum allowed deviation in points
MAX_SLIPPAGE = 3  # Maximum allowed slippage in points
MAGIC_NUMBER = 123456  # Unique identifier for trades placed by this EA
INITIAL_CAPITAL = 10000  # Initial balance for backtest
LOT_SIZE = 0.01  # Lot size for backtest
RISK_PER_TRADE = 1.0  # Risk per trade in percentage
MAX_POSITIONS = 5  # Maximum number of concurrent positions
MARGIN = 0.0025
SPREAD = 0.00007

# Other system configurations (Add more as needed)
LOG_LEVEL = "INFO" 

# Trading symbols by asset class
FOREX_SYMBOLS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCHF", "USDCAD", "USDJPY"
]

COMMODITY_SYMBOLS = [
    "XAUUSD", "XAUEUR", "XAUGBP", "XAUJPY", "XAUAUD", "XAUCHF", "XAGUSD"
]

INDEX_SYMBOLS = [
    "US500", "US30", "UK100", "GER40", "NAS100", "USDX", "EURX"
]

# Combine all symbols
ALL_SYMBOLS = FOREX_SYMBOLS + COMMODITY_SYMBOLS + INDEX_SYMBOLS 

# Backtest settings
DEFAULT_SYMBOL = "GBPUSD"  # Symbol to backtest
DEFAULT_TIMEFRAME = "M5"  # Timeframe to backtest
DEFAULT_CORE_TIMEFRAME = "D1"  # Timeframe to calculate core functions 
DEFAULT_START_DATE = "2025-01-01"  # Start date for backtest
DEFAULT_END_DATE = "2025-03-31"  # End date for backtest
DEFAULT_START_CANDLE = 0
DEFAULT_END_CANDLE = 1000 