"""
Manages the connection to the MetaTrader 5 terminal.
"""

import MetaTrader5 as mt5  # type: ignore
from typing import Optional, Dict, List, Union, Any
from datetime import datetime, timedelta, UTC
import pandas as pd
import os
import time # Import time module for sleep
from configparser import ConfigParser

# Suppress type checking for MetaTrader5 module


# Constants for messages
_MSG_NOT_CONNECTED = "Not connected to MT5"

# Import the custom logger function
from app.util.logger import get_logger

# Get logger instance using custom configuration
logger = get_logger(__name__)

class MT5Client:
    """Handles the connection and authentication to the MT5 terminal."""

    def __init__(self, config_path=None, symbols=None, broker=1):
        """
        Initializes the MT5 connection using credentials from the config file.

        Args:
            config_path (str): Path to the configuration file. Defaults to DEFAULT_CONFIG_PATH.
            symbols (list): List of symbols to initialize
            broker (int): Broker number (1, 2, 3, or 4)
        """

        self.config = ConfigParser(interpolation=None)
        if config_path is None:
            logger.error("No configuration file path provided. Please provide a valid configuration file path.")
            raise ValueError("No configuration file path provided. Please provide a valid configuration file path.")
        
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}. Please provide a valid configuration file path.")
        
        self.config.read(config_path)
        
        # Store broker number for symbol mapping
        self.broker = broker
        
        # Initialize symbol mappings based on broker
        self._init_symbol_mappings()
        
        if broker == 1:
            self.login = int(self.config['MT5']['Login'])
            self.password = self.config['MT5']['Password']
            self.server = self.config['MT5']['Server']
            self.path = self.config['MT5']['Path']
        elif broker == 2:
            self.login = int(self.config['DEMO_MT5']['Login'])
            self.password = self.config['DEMO_MT5']['Password']
            self.server = self.config['DEMO_MT5']['Server']
            self.path = self.config['DEMO_MT5']['Path']
        elif broker == 3:
            self.login = int(self.config['MT5_PURPLE']['Login'])
            self.password = self.config['MT5_PURPLE']['Password']
            self.server = self.config['MT5_PURPLE']['Server']
            self.path = self.config['MT5_PURPLE']['Path']
        else:
            logger.error(f"Invalid broker specified: {broker}")
            raise ValueError(f"Invalid broker specified: {broker}")


        self.symbols = symbols or []

        # Connection attempt parameters
        self.max_retries = 3
        self.retry_delay = 5 # seconds
        
        self._connected = False
        self._initialized = False
        self.connect()

    def _init_symbol_mappings(self):
        """Initialize symbol mappings based on broker number."""
        if self.broker == 3:
            # Purple Trader symbol mappings (adds _ecn suffix)
            self.symbol_mapping = {
                "AUDCAD": "AUDCAD_ecn",
                "AUDCHF": "AUDCHF_ecn",
                "AUDJPY": "AUDJPY_ecn",
                "AUDNZD": "AUDNZD_ecn",
                "AUDUSD": "AUDUSD_ecn",
                "CADCHF": "CADCHF_ecn",
                "CADJPY": "CADJPY_ecn",
                "CHFJPY": "CHFJPY_ecn",
                "EURAUD": "EURAUD_ecn",
                "EURCAD": "EURCAD_ecn",
                "EURCHF": "EURCHF_ecn",
                "EURGBP": "EURGBP_ecn",
                "EURJPY": "EURJPY_ecn",
                "EURNZD": "EURNZD_ecn",
                "EURUSD": "EURUSD_ecn",
                "GBPAUD": "GBPAUD_ecn",
                "GBPCAD": "GBPCAD_ecn",
                "GBPCHF": "GBPCHF_ecn",
                "GBPJPY": "GBPJPY_ecn",
                "GBPNZD": "GBPNZD_ecn",
                "GBPUSD": "GBPUSD_ecn",
                "NZDCAD": "NZDCAD_ecn",
                "NZDCHF": "NZDCHF_ecn",
                "NZDJPY": "NZDJPY_ecn",
                "NZDUSD": "NZDUSD_ecn",
                "USDCHF": "USDCHF_ecn",
                "USDCAD": "USDCAD_ecn",
                "USDJPY": "USDJPY_ecn",
                "XAUUSD": "XAUUSD_ecn",
                "XAGUSD": "XAGUSD_ecn"
            }
        else:
            # For other brokers, no mapping needed
            self.symbol_mapping = {}

    def map_symbol(self, symbol: str) -> str:
        """
        Map a standard symbol to broker-specific symbol format.
        
        Args:
            symbol (str): Standard symbol (e.g., "EURUSD")
            
        Returns:
            str: Broker-specific symbol (e.g., "EURUSD_ecn" for broker 3)
        """
        return self.symbol_mapping.get(symbol, symbol)

    def connect(self):
        """Establishes connection to the MetaTrader 5 terminal with retry logic."""
        if self._connected:
            logger.info("Already connected to MT5.")
            return True

        attempts = 0
        while attempts < self.max_retries:
            attempts += 1
            logger.info(f"Initializing MT5 connection (Attempt {attempts}/{self.max_retries})...")
            
            if mt5.initialize(login=self.login, password=self.password, server=self.server, path=self.path):    # type: ignore
                logger.info("MT5 initialized successfully. Verifying account info...")
                account_info = mt5.account_info()  # type: ignore
                if account_info:
                    logger.info(f"Connected to account: {account_info.login} on server {account_info.server}")
                    self._connected = True
                    self.initialize_symbols()
                    return True # Successfully connected
                else:
                    error_code = mt5.last_error()  # type: ignore
                    logger.error(f"Failed to get account info after successful initialization, error code = {error_code}")
                    mt5.shutdown()  # type: ignore
                    # Treat this as a connection failure for retry purposes
            else:
                error_code = mt5.last_error()  # type: ignore
                logger.error(f"MT5 initialize() failed on attempt {attempts}, error code = {error_code}")
                # mt5.shutdown() might be implicitly called or not needed if initialize failed,
                # but calling it ensures cleanup.
                mt5.shutdown()  # type: ignore
            
            # Wait before the next retry, unless it's the last attempt
            if attempts < self.max_retries:
                logger.info(f"Waiting {self.retry_delay} seconds before next connection attempt...")
                time.sleep(self.retry_delay)

        # If loop finishes without returning True, connection failed
        logger.error(f"Failed to connect to MT5 after {self.max_retries} attempts.")
        self._connected = False
        return False

    def shutdown(self):
        """Shuts down the MetaTrader 5 connection."""
        if self._connected:
            logger.info("Shutting down MT5 connection...")
            mt5.shutdown()  # type: ignore
            self._connected = False
            logger.info("MT5 connection shut down.")
        else:
            logger.info("MT5 connection already shut down.")

    def is_connected(self):
        """Checks if the MT5 terminal is connected."""
        # Perform a light check, e.g., getting terminal info
        terminal_info = mt5.terminal_info()  # type: ignore
        if terminal_info is None:
            logger.warning(f"MT5 connection lost (terminal_info failed), error code: {mt5.last_error()}")  # type: ignore
            self._connected = False
            # Attempt to reconnect automatically or signal failure
            # self.connect() # Optional: uncomment for auto-reconnect attempt
            return False
            
        # Update internal state based on a successful check
        if not self._connected:
             logger.info("MT5 connection re-established.")
             self._connected = True
             
        return self._connected

    def get_connection_status(self):
        """Returns the internal connection status flag."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        # Connection is established in __init__
        if not self._connected:
            raise ConnectionError("Failed to establish MT5 connection.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def initialize_symbols(self) -> None:
        """
        Initialize required symbols in the market watch.
        This ensures all symbols are available for data retrieval.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return
        
        try:
            # Get all available symbols
            available_symbols = self.symbols_get()
            if available_symbols is None:
                logger.error("Failed to get available symbols")
                return
                
            available_symbol_names = {symbol.name for symbol in available_symbols}
            
            for symbol in self.symbols:
                # Map the standard symbol to the broker-specific symbol
                broker_symbol = self.map_symbol(symbol)
                
                if broker_symbol not in available_symbol_names:
                    logger.warning(f"Symbol {symbol} not available in broker's symbol list. Attempting to add.")
                    if not mt5.symbol_select(broker_symbol, True): # type: ignore
                        logger.error(f"Failed to add symbol {broker_symbol} to market watch: {mt5.last_error()}") # type: ignore
                        continue
                    
                    # Verify symbol was actually added
                    if not self.is_symbol_visible(broker_symbol):
                        logger.error(f"Symbol {broker_symbol} not visible in market watch after adding")
                        continue
                        
                logger.debug(f"Symbol {symbol} initialized successfully")
                self._initialized = True
            
            # Log summary
            visible_symbols = [s for s in self.symbols if self.is_symbol_visible(s)]
            logger.info(f"Symbol initialization completed. {len(visible_symbols)}/{len(self.symbols)} symbols visible in market watch")
            
        except Exception as e:
            logger.error(f"Error initializing symbols: {e}")
            raise

    def symbols_get(self):
        """
        Get all available symbols from MT5.
        
        Returns:
            List[mt5.SymbolInfo]: List of available symbols or None if error occurs
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        try:
            # Get symbols
            symbols = mt5.symbols_get() # type: ignore
            if symbols is None:
                error = mt5.last_error() # type: ignore
                logger.error(f"Failed to get symbols from MT5: {error}")
                return None
            
            # Log success
            logger.info(f"Successfully retrieved {len(symbols)} symbols from MT5")
            return symbols
            
        except Exception as e:
            logger.exception(f"Error getting symbols from MT5: {e}")
            return None
        
    def is_symbol_visible(self, symbol: str) -> bool:
        """
        Check if a symbol is visible in the market watch.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            
        Returns:
            bool: True if symbol is visible in market watch, False otherwise
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol) # type: ignore
            if symbol_info is None:
                return False
            return bool(symbol_info.visible)
        except Exception as e:
            logger.error(f"Error checking symbol visibility for {symbol}: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information including balance, equity, etc.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            raise RuntimeError(_MSG_NOT_CONNECTED)
        
        try:
            account_info = mt5.account_info() # type: ignore
            if account_info is None:
                raise RuntimeError(f"Failed to get account info: {mt5.last_error()}") # type: ignore
            
            return {
                "login": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "profit": account_info.profit,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "leverage": account_info.leverage,
                "currency": account_info.currency,
                "server": account_info.server,
                "trade_mode": account_info.trade_mode,
                "trade_allowed": account_info.trade_allowed,
                "trade_expert": account_info.trade_expert,
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            
        Returns:
            Dict[str, Any]: Symbol information including price, spread, etc.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            raise RuntimeError(_MSG_NOT_CONNECTED)
        
        try:
            # Map the symbol to broker-specific format
            broker_symbol = self.map_symbol(symbol)
            
            symbol_info = mt5.symbol_info(broker_symbol) # type: ignore
            if symbol_info is None:
                raise RuntimeError(f"Failed to get symbol info for {symbol} (mapped to {broker_symbol}): {mt5.last_error()}") # type: ignore
            
            return symbol_info
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol} (mapped to {broker_symbol}): {e}")
            raise

    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets the most recent tick for a symbol."""
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        # Map the symbol to broker-specific format
        broker_symbol = self.map_symbol(symbol)
        
        tick = mt5.symbol_info_tick(broker_symbol)  # type: ignore
        if tick:
            # Convert named tuple to dictionary
            return tick._asdict()
        logger.error(f"Failed to get tick for {symbol} (mapped to {broker_symbol}): {mt5.last_error()}")  # type: ignore
        return None

    def get_orders(self) -> List[Dict[str, Any]]:
        """Gets all active orders."""
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return []
        orders = mt5.orders_get()  # type: ignore
        if orders is None:
            logger.error(f"Failed to get orders: {mt5.last_error()}")  # type: ignore
            return []
        return [order._asdict() for order in orders]

    def get_positions(self) -> List[Dict[str, Any]]:
        """Gets all open positions."""
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return []
        positions = mt5.positions_get()  # type: ignore
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")  # type: ignore
            return []
        return [pos._asdict() for pos in positions]

    def order_send(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends a trade request to the MT5 terminal."""
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None

        logger.info(f"Sending order request: {request}")
        result = mt5.order_send(request)  # type: ignore

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")  # type: ignore
            return None

        logger.info(f"Order send result: {result}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:  # type: ignore
            logger.error(f"Order send rejected: retcode={result.retcode}, comment={result.comment}")
            return None

        return result._asdict()

    def get_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to MT5 timeframe enum."""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe.upper())
    
    def _get_rates_from_mt5(self, symbol: str, tf: int, start_pos: Optional[int] = None, end_pos: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[List[Dict[str, Any]]]:
        """Internal helper to fetch raw rates from MT5."""
        if start_date is not None or end_date is not None:
            # Use UTC timestamps consistently
            now = datetime.now(UTC)
            default_start = now - timedelta(days=30)
            default_end = now

            # Ensure start_date and end_date are timezone-aware (UTC)
            if start_date is not None and start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=UTC)
            if end_date is not None and end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=UTC)

            _start = start_date or default_start
            _end = end_date or default_end
            logger.info(f"Fetching {symbol} data from {_start} to {_end}")
            rates = mt5.copy_rates_range(symbol, tf, _start, _end) # type: ignore
        else:
            _start_pos = start_pos or 0
            _count = int((end_pos or 1000) - _start_pos) # MT5 uses count, not end position, ensure integer
            logger.info(f"Fetching {symbol} data from position {_start_pos}, count {_count}")
            # Corrected call to use copy_rates_from_pos with count
            try:
                rates = mt5.copy_rates_from_pos(symbol, tf, _start_pos, _count) # type: ignore
            except SystemError as e:
                error = mt5.last_error() # type: ignore
                logger.error(f"SystemError getting rates for {symbol}: {error}")
                return None

        if rates is None:
            error = mt5.last_error() # type: ignore
            logger.error(f"Failed to get rates for {symbol}: {error}")
            return None
            
        logger.info(f"Fetched {len(rates)} bars for {symbol}")
        return rates

    def _format_rates_dataframe(self, rates, symbol: str) -> Optional[pd.DataFrame]:
        """Internal helper to format raw rates into a DataFrame."""
        if rates is None or len(rates) == 0:
            logger.warning(f"No rate data returned for {symbol} to format.")
            return None
            
        try:
            df = pd.DataFrame(rates)
            
            # Log the actual columns we received for debugging
            logger.debug(f"Original columns for {symbol}: {list(df.columns)}")
            
            # Check if we have the expected columns before renaming
            expected_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                logger.error(f"Available columns: {list(df.columns)}")
                return None
            
            # Rename columns
            df.rename(columns={
                'time': 'datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)

            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
            
            # Select only the columns we need
            required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_columns].copy()  # Use .copy() to avoid SettingWithCopyWarning
            
            # Set index
            df.set_index('datetime', inplace=True)
            
            logger.debug(f"Successfully formatted DataFrame for {symbol} with shape: {df.shape}")
            return df  # type: ignore
            
        except Exception as e:
            logger.exception(f"Error formatting DataFrame for {symbol}: {e}")
            return None

    def fetch_data(self, symbol: str, timeframe: str = "D1", start_pos: Optional[int] = None, end_pos: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:

        """
        Fetch OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g. "M1", "M5", "H1", "D1")
            start_pos: Start position (0 means most recent)
            end_pos: End position (used to calculate count for position-based fetching)
            start_date: Optional start date for data filtering (UTC recommended)
            end_date: Optional end date for data filtering (UTC recommended)
            
        Returns:
            Optional[pd.DataFrame]: OHLCV data with datetime index, or None on failure.
        """
        if not self.is_connected():
            logger.error(f"Cannot fetch data: {_MSG_NOT_CONNECTED}")
            return None

        try:
            # Map the symbol to broker-specific format
            broker_symbol = self.map_symbol(symbol)
            if broker_symbol != symbol:
                logger.debug(f"Mapped symbol {symbol} to {broker_symbol} for broker {self.broker}")
            
            tf = self.get_timeframe(timeframe)
            if tf is None:
                logger.error(f"Invalid timeframe specified: {timeframe}")
                return None # Return None for invalid timeframe
            
            # Fetch raw rates using the helper method
            rates = self._get_rates_from_mt5(
                broker_symbol, tf, start_pos, end_pos, start_date, end_date
            )
            
            # Format rates into DataFrame using the helper method
            return self._format_rates_dataframe(rates, symbol)
            
        except Exception as e:
            # Catch unexpected errors during timeframe conversion or helper calls
            logger.exception(f"Unexpected error during fetch_data for {symbol}: {e}")
            return None 


# TODO: Write documentation for the MT5Client class