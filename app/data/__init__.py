import pandas as pd
import requests

# Import the custom logger function
from app.util.logger import get_logger

# Import streaming functionality
try:
    from .streaming import (
        StreamingManager,
        StreamConfig,
        StreamType,
        TickData,
        OHLCVData,
        DataStreamHandler,
        MT5TickStreamHandler,
        MT5OHLCVStreamHandler,
        WebSocketStreamHandler,
        create_tick_callback,
        create_ohlcv_callback,
        create_data_logger_callback
    )
except ImportError as e:
    logger = get_logger(__name__)
    logger.warning(f"Could not import streaming modules: {e}")

# Get logger instance using custom configuration
logger = get_logger(__name__)

# Define what gets imported with "from app.data import *"
__all__ = [
    'validate_ohlcv_data', 
    'fetch_fundamental_data',
    'StreamingManager',
    'StreamConfig', 
    'StreamType',
    'TickData',
    'OHLCVData',
    'DataStreamHandler',
    'MT5TickStreamHandler',
    'MT5OHLCVStreamHandler',
    'WebSocketStreamHandler',
    'create_tick_callback',
    'create_ohlcv_callback',
    'create_data_logger_callback'
]

# Common Data Functions

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Basic validation for OHLCV data."""
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logger.error("DataFrame does not contain required columns")
        return False
    
    # Check for negative prices
    if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        logger.error("DataFrame contains negative prices")
        return False
    
    # Check for logical OHLC relationships
    high_valid = (df['High'] >= df['Open']) & (df['High'] >= df['Close'])
    low_valid = (df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])
    
    if not (high_valid & low_valid).all():
        logger.error("DataFrame contains invalid OHLC relationships")
        return False
    
    return True


def fetch_fundamental_data(api_key):
    url: str = "https://www.jblanked.com/news/api/forex-factory/calendar/today/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
        # for news in data:
        #    print(f"{news['Date']} {news['Currency']} {news['Name']} STRENGTH : {news['Strength']} OUTCOME : {news['Outcome']} ACTUAL : {news['Actual']} FORECAST : {news['Forecast']} PREVIOUS : {news['Previous']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None