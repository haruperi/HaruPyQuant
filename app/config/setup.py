import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from configparser import ConfigParser

# Add project root to the Python path   
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(PROJECT_ROOT))

# Import logger after path setup
from app.util.logger import *

# Common directories
APP_DIR = PROJECT_ROOT / "app"
CONFIG_DIR = APP_DIR / "config"
TESTS_DIR = PROJECT_ROOT / "tests"

# Default path for the configuration file
DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'config.ini'))

# Load configuration
config = ConfigParser(interpolation=None)

if DEFAULT_CONFIG_PATH is None:
    raise ValueError("No configuration file path provided. Please provide a valid configuration file path.")
        
if not os.path.exists(DEFAULT_CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {DEFAULT_CONFIG_PATH}. Please provide a valid configuration file path.")
        
config.read(DEFAULT_CONFIG_PATH)
        
MT5_LOGIN = int(config['MT5']['Login'])
MT5_PASSWORD = config['MT5']['Password']
MT5_SERVER = config['MT5']['Server']
MT5_PATH = config['MT5']['Path']

JBLANKED_API_KEY = config['JBLANKED']['API_KEY']


# Trading parameters
MAX_DEVIATION = 5  # Maximum allowed deviation in points
MAX_SLIPPAGE = 3  # Maximum allowed slippage in points
MAGIC_NUMBER = 123456  # Unique identifier for trades placed by this EA
INITIAL_CAPITAL = 1000  # Initial balance for backtest
LOT_SIZE = 0.01  # Lot size for backtest
RISK_PER_TRADE = 0.01  # Risk per trade in percentage
MAX_POSITIONS = 5  # Maximum number of concurrent positions
MARGIN = 0.0025
SPREAD = 0.00007

# Indicator parameters
ADR_PERIOD = 10 # Number of days over which to calculate the ADR
ATR_PERIOD = 12 # ATR period
STOP_ADR_RATIO = 10 # Stop loss level as a multiple of the ADR
FAST_MA_PERIOD = 12 # Fast moving average period
SLOW_MA_PERIOD = 48 # Slow moving average period
BIAS_MA_PERIOD = 144 # Bias moving average period
RSI_PERIOD = 12 # RSI period
WILLIAMS_R_PERIOD = 6 # Williams %R period

# Risk management variables 
CORRELATION_PERIOD = 10       # Correlation period (rolling window 48 for intraday and 10 for daily)
VOLATILITY_PERIOD = 5        # Volatility period (rolling window 24 for intraday and 5 for daily)
CONFIDENCE_LEVEL = 0.95       # Percent to be covered in statistics
RISK_THRESHOLD = 50         # Risk threshold for accepting new positions (10%)

# Data Settings
DATA_DIR = PROJECT_ROOT / "app/data/files"
TESTING_RESULTS_FILE = DATA_DIR / "testing_results.csv"
from app.data.mt5_client import MT5Client

INTERVAL_MINUTES = 5          # Trading timeframe minutes
TIME_SHIFT=-3                 # Broker time shift from GMT 0
DEFAULT_TIMEFRAME = f'M{INTERVAL_MINUTES}'  # Default timeframe for data retrieval
CORE_TIMEFRAME = "D1"         # Timeframe to calculate core functions (H1 for intraday and D1 for daily)
START_POS=0                   # Data retrieval index starting point
END_POS=1000                   # Data retrieval index ending point
END_POS_HTF=200               # Data retrieval index ending point for a higher timeframe (if any)
END_POS_D1=ADR_PERIOD*3                 # Data retrieval index ending point for daily timeframe (whole last month)
RANGE_START = datetime.now().strftime("%Y-%m-%d")         # Data retrieval range starting point
RANGE_END = (datetime.now() - timedelta(days=END_POS_D1)).strftime("%Y-%m-%d")  # Data retrieval index starting point
START_DATE = "2024-12-15"     # Data retrieval date starting point
END_DATE = "2025-04-01"       # Data retrieval date ending point
TEST_SYMBOL = "GBPUSD"        # Random symbol for testing purposes
DEFAULT_SYMBOL = TEST_SYMBOL  # Default symbol for testing and examples
DEFAULT_START_CANDLE = START_POS  # Default start position for data retrieval
DEFAULT_END_CANDLE = END_POS      # Default end position for data retrieval


# Chart Colors
CHART_BACKGROUND_COLOR = "#161A25"
BULLISH_CANDLE_COLOR = "#26A69A"
BEARISH_CANDLE_COLOR = "#EF5350"

# Other system configurations (Add more as needed)
LOG_LEVEL = "INFO" 

# MongoDB settings
MONGO_DB_NAME = "haru_pyquant"
MONGO_COLLECTION_NAME = "forex_sample"

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

# Dukascopy Instruments
INSTRUMENT_IDX_PRT_IDX_EUR = "PRT.IDX/EUR"
INSTRUMENT_BND_CFD_BUND_TR_EUR = "BUND.TR/EUR"
INSTRUMENT_BND_CFD_UKGILT_TR_GBP = "UKGILT.TR/GBP"
INSTRUMENT_BND_CFD_USTBOND_TR_USD = "USTBOND.TR/USD"
INSTRUMENT_ETF_CFD_ARKI_US_USD = "ARKI.US/USD"
INSTRUMENT_ETF_CFD_BUGG_GB_GBP = "BUGG.GB/GBP"
INSTRUMENT_ETF_CFD_CSH2_FR_EUR = "CSH2.FR/EUR"
INSTRUMENT_ETF_CFD_CSH2_GB_GBX = "CSH2.GB/GBX"
INSTRUMENT_ETF_CFD_CYSE_GB_GBX = "CYSE.GB/GBX"
INSTRUMENT_ETF_CFD_ESIH_GB_GBP = "ESIH.GB/GBP"
INSTRUMENT_ETF_CFD_IGLN_US_USD = "IGLN.US/USD"
INSTRUMENT_ETF_CFD_IUFS_US_USD = "IUFS.US/USD"
INSTRUMENT_ETF_CFD_SEMI_GB_GBP = "SEMI.GB/GBP"
INSTRUMENT_ETF_CFD_SGLD_US_USD = "SGLD.US/USD"
INSTRUMENT_ETF_CFD_SMH_US_USD = "SMH.US/USD"
INSTRUMENT_ETF_CFD_SMTC_US_USD = "SMTC.US/USD"
INSTRUMENT_ETF_CFD_WTAI_US_USD = "WTAI.US/USD"
INSTRUMENT_ETF_CFD_XDER_GB_GBX = "XDER.GB/GBX"
INSTRUMENT_ETF_CFD_XDWH_US_USD = "XDWH.US/USD"
INSTRUMENT_ETF_CFD_XDWT_US_USD = "XDWT.US/USD"
INSTRUMENT_VCCY_ADA_USD = "ADA/USD"
INSTRUMENT_VCCY_AVE_USD = "AVE/USD"
INSTRUMENT_VCCY_BAT_USD = "BAT/USD"
INSTRUMENT_VCCY_BCH_CHF = "BCH/CHF"
INSTRUMENT_VCCY_BCH_EUR = "BCH/EUR"
INSTRUMENT_VCCY_BCH_GBP = "BCH/GBP"
INSTRUMENT_VCCY_BCH_USD = "BCH/USD"
INSTRUMENT_VCCY_BTC_CHF = "BTC/CHF"
INSTRUMENT_VCCY_BTC_EUR = "BTC/EUR"
INSTRUMENT_VCCY_BTC_GBP = "BTC/GBP"
INSTRUMENT_VCCY_BTC_USD = "BTC/USD"
INSTRUMENT_VCCY_CMP_USD = "CMP/USD"
INSTRUMENT_VCCY_DSH_USD = "DSH/USD"
INSTRUMENT_VCCY_ENJ_USD = "ENJ/USD"
INSTRUMENT_VCCY_EOS_USD = "EOS/USD"
INSTRUMENT_VCCY_ETH_CHF = "ETH/CHF"
INSTRUMENT_VCCY_ETH_EUR = "ETH/EUR"
INSTRUMENT_VCCY_ETH_GBP = "ETH/GBP"
INSTRUMENT_VCCY_ETH_USD = "ETH/USD"
INSTRUMENT_VCCY_LNK_USD = "LNK/USD"
INSTRUMENT_VCCY_LTC_CHF = "LTC/CHF"
INSTRUMENT_VCCY_LTC_EUR = "LTC/EUR"
INSTRUMENT_VCCY_LTC_GBP = "LTC/GBP"
INSTRUMENT_VCCY_LTC_USD = "LTC/USD"
INSTRUMENT_VCCY_MAT_USD = "MAT/USD"
INSTRUMENT_VCCY_MKR_USD = "MKR/USD"
INSTRUMENT_VCCY_TRX_USD = "TRX/USD"
INSTRUMENT_VCCY_UNI_USD = "UNI/USD"
INSTRUMENT_VCCY_UST_USD = "UST/USD"
INSTRUMENT_VCCY_XLM_CHF = "XLM/CHF"
INSTRUMENT_VCCY_XLM_EUR = "XLM/EUR"
INSTRUMENT_VCCY_XLM_GBP = "XLM/GBP"
INSTRUMENT_VCCY_XLM_USD = "XLM/USD"
INSTRUMENT_VCCY_XMR_USD = "XMR/USD"
INSTRUMENT_VCCY_XRP_USD = "XRP/USD"
INSTRUMENT_VCCY_YFI_USD = "YFI/USD"

# Trading symbols by asset class (Dukascopy)
Dukascopy_FOREX_SYMBOLS = [
    "AUD/CAD", "AUD/CHF", "AUD/JPY", "AUD/NZD", "AUD/USD",
    "CAD/CHF", "CAD/JPY", "CHF/JPY",
    "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/GBP", "EUR/JPY", "EUR/NZD", "EUR/USD",
    "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/JPY", "GBP/NZD", "GBP/USD",
    "NZD/CAD", "NZD/CHF", "NZD/JPY", "NZD/USD",
    "USD/CHF", "USD/CAD", "USD/JPY"
]

Dukascopy_COMMODITY_SYMBOLS = [
    "XAU/USD", "XAU/EUR", "XAU/GBP", "XAU/JPY", "XAU/AUD", "XAU/CHF", "XAG/USD"
]

Dukascopy_INDEX_SYMBOLS = [
    "US500", "US30", "UK100", "GER40", "NAS100", "USD/X", "EUR/X"
]

# Combine all symbols (Dukascopy)
Dukascopy_ALL_SYMBOLS = Dukascopy_FOREX_SYMBOLS + Dukascopy_COMMODITY_SYMBOLS + Dukascopy_INDEX_SYMBOLS 