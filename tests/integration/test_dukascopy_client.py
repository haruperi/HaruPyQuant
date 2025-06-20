import unittest
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.data import dukascopy_client as dkc
from app.config.constants import INSTRUMENT_FX_MAJORS_EUR_USD, INSTRUMENT_FX_MAJORS_GBP_USD

class TestDukascopyClientIntegration(unittest.TestCase):
    """
    Integration tests for the dukascopy_client module.
    These tests connect to the live Dukascopy API to fetch data.
    """

    def test_fetch_ohlc_data(self):
        """Test fetching historical OHLC data."""
        instrument = "EUR/USD"
        start_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(microsecond=0)
        end_date = start_date + timedelta(hours=1)
        
        df = dkc.fetch(
            instrument=instrument,
            interval=dkc.INTERVAL_MIN_5,
            offer_side=dkc.OFFER_SIDE_BID,
            start=start_date,
            end=end_date,
        )
        
        self.assertIsInstance(df, pd.DataFrame, "Should return a pandas DataFrame")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn("open", df.columns, "DataFrame should have an 'open' column")
        self.assertIn("high", df.columns, "DataFrame should have a 'high' column")
        self.assertIn("low", df.columns, "DataFrame should have a 'low' column")
        self.assertIn("close", df.columns, "DataFrame should have a 'close' column")
        self.assertIn("volume", df.columns, "DataFrame should have a 'volume' column")
        
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex")
        self.assertTrue(df.index.is_monotonic_increasing, "Index should be sorted")
        
        # Allow for a bit of leeway in the start time check due to API behavior
        interval_minutes = int(''.join(filter(str.isdigit, dkc.INTERVAL_MIN_5)))
        self.assertGreaterEqual(df.index.min().to_pydatetime(), start_date - timedelta(minutes=interval_minutes), "Data should start after or at the start_date")
        self.assertLessEqual(df.index.max().to_pydatetime(), end_date, "Data should end before or at the end_date")

    def test_fetch_tick_data(self):
        """Test fetching historical tick data."""
        instrument = "GBP/JPY"
        start_date = datetime.now(timezone.utc) - timedelta(minutes=5)
        end_date = datetime.now(timezone.utc)

        df = dkc.fetch(
            instrument=instrument,
            interval=dkc.INTERVAL_TICK,
            offer_side=dkc.OFFER_SIDE_ASK,
            start=start_date,
            end=end_date,
        )
        
        self.assertIsInstance(df, pd.DataFrame, "Should return a pandas DataFrame")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn("bidPrice", df.columns)
        self.assertIn("askPrice", df.columns)
        self.assertIn("bidVolume", df.columns)
        self.assertIn("askVolume", df.columns)
        
        self.assertIsInstance(df.index, pd.DatetimeIndex, "Index should be a DatetimeIndex")

    def test_live_fetch_completed_bars(self):
        """Test live fetching of completed OHLC bars."""
        instrument = "AUD/USD"
        start_date = datetime.now(timezone.utc) - timedelta(minutes=10)
        end_date = start_date + timedelta(minutes=1) # Fetch a short period
        
        data_generator = dkc.live_fetch(
            instrument=instrument,
            interval_value=1,
            time_unit=dkc.TIME_UNIT_MIN,
            offer_side=dkc.OFFER_SIDE_BID,
            start=start_date,
            end=end_date,
            completed_bars_only=True,
        )
        
        bars = list(data_generator)
        
        # This test can be flaky if there's no trade activity. 
        # We'll just ensure it runs without errors and yields DataFrames if data is available.
        if bars:
            self.assertIsInstance(bars[0], pd.DataFrame)
            self.assertEqual(len(bars[0]), 1)

    def test_live_fetch_incomplete_bars(self):
        """Test live fetching of incomplete (live) OHLC bars."""
        instrument = "USD/CAD"
        start_date = datetime.now(timezone.utc) - timedelta(seconds=15)
        end_date = datetime.now(timezone.utc)
        
        data_generator = dkc.live_fetch(
            instrument=instrument,
            interval_value=5,
            time_unit=dkc.TIME_UNIT_SEC,
            offer_side=dkc.OFFER_SIDE_BID,
            start=start_date,
            end=end_date,
            completed_bars_only=False,
        )
        
        # We are just testing that it yields data frames correctly
        results = list(data_generator)
        if results:
            self.assertIsInstance(results[-1], pd.DataFrame)
            if not results[-1].empty:
                self.assertIn("close", results[-1].columns)

    def test_fetch_invalid_instrument(self):
        """Test that fetching an invalid instrument fails gracefully."""
        # Dukascopy API seems to return empty list for invalid instruments, not an error.
        instrument = "INVALID/SYMBOL"
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 1, 1, tzinfo=timezone.utc)
        
        df = dkc.fetch(
            instrument=instrument,
            interval=dkc.INTERVAL_MIN_5,
            offer_side=dkc.OFFER_SIDE_BID,
            start=start_date,
            end=end_date,
        )
        
        self.assertTrue(df.empty, "Fetching an invalid instrument should return an empty DataFrame")

class RealTestCheck:
    def test_historical_data(self):
        
        start = datetime(2025, 6, 1)
        end = datetime(2025, 6, 18)
        instrument = INSTRUMENT_FX_MAJORS_EUR_USD
        interval = dkc.INTERVAL_HOUR_1
        offer_side = dkc.OFFER_SIDE_BID

        df = dkc.fetch(
            instrument,
            interval,
            offer_side,
            start,
            end,
        )

        print(df)

    def test_live_data(self):

        now = datetime.now()
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(hours=24)
        instrument = INSTRUMENT_FX_MAJORS_GBP_USD
        offer_side = dkc.OFFER_SIDE_BID

        iterator = dkc.live_fetch(
            instrument,
            1,
            dkc.TIME_UNIT_MIN,
            offer_side,
            start,
            end,
        )

        for df in iterator:
            print(df)


if __name__ == '__main__':
    #unittest.main() 
    real_test = RealTestCheck()
    #real_test.test_historical_data()
    real_test.test_live_data()