import unittest
import os
import sys
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, UTC

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.data.mt5_client import MT5Client
from app.config.constants import DEFAULT_CONFIG_PATH, ALL_SYMBOLS


class TestMT5ClientIntegration(unittest.TestCase):
    """
    Integration tests for the MT5Client class.
    These tests require a live connection to a MetaTrader 5 terminal
    and a valid 'config.ini' file in the project root.
    """

    @classmethod
    def setUpClass(cls):
        """
        Check for the configuration file before running any tests.
        """
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            raise FileNotFoundError(f"Configuration file not found at {DEFAULT_CONFIG_PATH}. Cannot run integration tests.")
        
        # Initialize a client once for the class to check connection, then shut it down.
        # Each test will create its own instance to ensure isolation.
        client = None  # Initialize client to None
        try:
            client = MT5Client(config_path=DEFAULT_CONFIG_PATH)
            if not client.get_connection_status():
                raise ConnectionError("Failed to connect to MT5 for integration tests.")
        finally:
            if client and client.get_connection_status():
                client.shutdown()

    def setUp(self):
        """Set up a new MT5Client instance for each test."""
        self.client = MT5Client(config_path=DEFAULT_CONFIG_PATH)
        self.assertTrue(self.client.get_connection_status(), "Failed to connect to MT5 for a test.")

    def tearDown(self):
        """Shut down the connection after each test."""
        self.client.shutdown()

    def test_connection_and_shutdown(self):
        """Test connection status methods and shutdown."""
        self.assertTrue(self.client.get_connection_status())
        self.assertTrue(self.client.is_connected())
        self.client.shutdown()
        self.assertFalse(self.client.get_connection_status())
        self.assertFalse(self.client.is_connected())

    def test_context_manager(self):
        """Test the client as a context manager."""
        with MT5Client(config_path=DEFAULT_CONFIG_PATH) as client:
            self.assertTrue(client.is_connected())
        self.assertFalse(client.is_connected(), "Connection should be closed after exiting context.")

    def test_get_account_info(self):
        """Test fetching account information."""
        account_info = self.client.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertIn('login', account_info)
        self.assertIn('balance', account_info)
        self.assertIn('equity', account_info)
        self.assertIsInstance(account_info['login'], int)
        self.assertIsInstance(account_info['balance'], float)

    def test_symbols_get(self):
        """Test retrieving all symbols."""
        symbols = self.client.symbols_get()
        self.assertIsNotNone(symbols)
        self.assertIsInstance(symbols, (list, tuple))
        self.assertGreater(len(symbols), 0)
        self.assertTrue(hasattr(symbols[0], 'name'))

    def test_initialize_symbols_and_visibility(self):
        """Test symbol initialization and visibility checks."""
        # initialize_symbols is called in __init__.
        # We test a common symbol from the ALL_SYMBOLS list.
        test_symbol = "EURUSD"
        if test_symbol in ALL_SYMBOLS:
            is_visible = self.client.is_symbol_visible(test_symbol)
            self.assertTrue(is_visible, f"{test_symbol} should be visible after initialization.")
        
        self.assertFalse(self.client.is_symbol_visible('NONEXISTENT_SYMBOL_XYZ'))

    def test_get_symbol_info(self):
        """Test fetching symbol information for a valid symbol."""
        symbol = "GBPUSD"
        symbol_info = self.client.get_symbol_info(symbol)
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info['name'], symbol)
        self.assertIn('bid', symbol_info)
        self.assertIn('ask', symbol_info)
        self.assertIn('contract_size', symbol_info)
        self.assertIsInstance(symbol_info['bid'], float)
        self.assertGreater(symbol_info['bid'], 0)

    def test_get_symbol_info_invalid_symbol(self):
        """Test fetching info for an invalid symbol raises an error."""
        with self.assertRaises(RuntimeError):
            self.client.get_symbol_info("INVALIDSYMBOL123")

    def test_get_timeframe(self):
        """Test timeframe string to enum conversion."""
        self.assertEqual(self.client.get_timeframe("M1"), mt5.TIMEFRAME_M1)
        self.assertEqual(self.client.get_timeframe("H4"), mt5.TIMEFRAME_H4)
        self.assertEqual(self.client.get_timeframe("D1"), mt5.TIMEFRAME_D1)
        self.assertIsNone(self.client.get_timeframe("INVALID_TF"))

    def test_fetch_data_by_position(self):
        """Test fetching data by position."""
        symbol = "EURUSD"
        df = self.client.fetch_data(symbol, timeframe="M1", start_pos=0, end_pos=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertEqual(list(df.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_fetch_data_by_date(self):
        """Test fetching data by date range."""
        symbol = "EURUSD"
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=1)
        df = self.client.fetch_data(symbol, timeframe="H1", start_date=start_date, end_date=end_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(list(df.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertGreaterEqual(df.index[0], start_date)
        self.assertLessEqual(df.index[-1], end_date)

    def test_fetch_data_invalid_symbol(self):
        """Test fetching data for an invalid symbol returns None."""
        df = self.client.fetch_data("INVALIDSYMBOL123", timeframe="M1")
        self.assertIsNone(df)

    def test_fetch_data_invalid_timeframe(self):
        """Test fetching data with an invalid timeframe returns None."""
        df = self.client.fetch_data("EURUSD", timeframe="INVALID_TF")
        self.assertIsNone(df)


class RealTestCheck:

    def test_connect(self):
        client = MT5Client()
        if client.connect():
            print("Connection successful")
        else:
            print("Connection failed")

    def test_is_connected(self):
        client = MT5Client()
        if client.is_connected():
            print("Connection successful")
        else:
            print("Connection failed")
        client.shutdown()
        if client.get_connection_status():
            print("Shutdown failed")
        else:
            print("Shutdown successful")

    def test_get_account_info(self):
        client = MT5Client()
        account_info = client.get_account_info()
        print(account_info)

    def test_get_symbol_info(self):
        client = MT5Client()
        symbol_info = client.get_symbol_info("EURUSD")
        print(symbol_info)

    def test_get_timeframe(self):
        client = MT5Client()
        timeframe = client.get_timeframe("M1")
        print(timeframe)

    def test_fetch_data_by_position(self):
        client = MT5Client()
        df = client.fetch_data("EURUSD", timeframe="M1", start_pos=0, end_pos=10)
        print(df)

    def test_fetch_data_by_date(self):
        client = MT5Client()
        df = client.fetch_data("EURUSD", timeframe="M1", start_date=datetime.now(UTC) - timedelta(days=1), end_date=datetime.now(UTC))
        print(df)


if __name__ == '__main__':
    #unittest.main() 
    real_test = RealTestCheck()
    real_test.test_fetch_data_by_date()