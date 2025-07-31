import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy


logger = get_logger(__name__)

class HarrietStrategy(BaseStrategy):
    """
    The Harriet strategy identifies Higher Lows for buy signals and Lower Highs for sell
    signals on both the lower and higher timeframes simultaneously.
    """

    def __init__(self, mt5_client: MT5Client, parameters: Dict[str, Any]):
        """
        Initializes the strategy with the given parameters.

        Args:
            mt5_client (MT5Client): The MT5 client for data access.
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
        """
        super().__init__(mt5_client, parameters)
        logger.info("Initializing Harriet strategy")

    def get_features(self, lt_data: pd.DataFrame, ht_data: pd.DataFrame, ht_min_dist: int = 5, lt_min_dist: int = 2, symbol_info: dict = None) -> pd.DataFrame:  
        """
        Generates Harriet signals for the strategy.
        Override this method to define strategy logic and signal generation.
        Signals should be stored in data['Signal'] as +1 (buy), -1 (sell), 0 (hold).

        Args:
            data (pd.DataFrame): The historical price data.

        Returns:
            pd.DataFrame: A DataFrame with Signal column.
        """
        logger.info("Starting Harriet strategy calculations")

        if lt_data.empty:
            logger.error(f"No data found for {self.symbol} on {self.lt_interval} timeframe.")
            return None
        
        if ht_data.empty:
            logger.error(f"No data found for {self.symbol} on {self.ht_interval} timeframe.")
            return None
        
        # Rename higher timeframe columns to avoid conflicts
        ht_data.rename(columns={
            'Open': 'HT_Open', 'High': 'HT_High',
            'Low': 'HT_Low', 'Close': 'HT_Close',
            'Volume': 'HT_Volume'
        }, inplace=True)

        # Resample higher timeframe data to match the lower timeframe index
        # and forward-fill the missing values. This simulates having the last known
        # HT bar data available for each LT bar.
        merged_data = pd.concat([lt_data, ht_data], axis=1)
        merged_data[['HT_Open', 'HT_High', 'HT_Low', 'HT_Close', 'HT_Volume']] = \
            merged_data[['HT_Open', 'HT_High', 'HT_Low', 'HT_Close', 'HT_Volume']].ffill()

        # Drop rows with any remaining NaN values (usually at the beginning)
        merged_data.dropna(inplace=True)

        # Generate signals
        # Calculate the minimum distance in price terms
        ht_min_distance_pips = ht_min_dist * 10 * symbol_info.point
        lt_min_distance_pips = lt_min_dist * 10 * symbol_info.point

        # --- Lower Timeframe Signals ---
        # These can be calculated directly on the main DataFrame.
        lt_higher_low = (
            (merged_data['Low'] > merged_data['Low'].shift(1)) &
            (merged_data['Close'] > merged_data['Open']) &
            ((merged_data['Low'] - merged_data['Low'].shift(1)) > lt_min_distance_pips)
        )
        lt_lower_high = (
            (merged_data['High'] < merged_data['High'].shift(1)) &
            (merged_data['Close'] < merged_data['Open']) &
            ((merged_data['High'].shift(1) - merged_data['High']) > lt_min_distance_pips)
        )

        # --- Higher Timeframe Signals ---
        # To correctly compare previous HT bars, we must first isolate the unique HT bars
        # to avoid comparing against forward-filled data from the same bar.
        ht_cols = ['HT_Open', 'HT_High', 'HT_Low', 'HT_Close']
        ht_data = merged_data[ht_cols].drop_duplicates().copy()

        # Calculate HT signals on the unique HT data.
        # Higher Low condition for Buy on HT
        ht_buy_condition = (
            (ht_data['HT_Low'] > ht_data['HT_Low'].shift(1)) &
            (ht_data['HT_Close'] > ht_data['HT_Open']) &
            ((ht_data['HT_Low'] - ht_data['HT_Low'].shift(1)) > ht_min_distance_pips)
        )
        # Lower High condition for Sell on HT
        ht_sell_condition = (
            (ht_data['HT_High'] < ht_data['HT_High'].shift(1)) &
            (ht_data['HT_Close'] < ht_data['HT_Open']) &
            ((ht_data['HT_High'].shift(1) - ht_data['HT_High']) > ht_min_distance_pips)
        )

        # Add the boolean signals as new columns to the unique HT data.
        ht_data['ht_higher_low'] = ht_buy_condition
        ht_data['ht_lower_high'] = ht_sell_condition
        
        # Now, map these hourly signals back to the main (e.g., 5-minute) dataframe.
        # We can do this by setting the HT signal columns on the main df, which will
        # create NaNs where the index doesn't match, and then forward-filling them.
        merged_data['ht_higher_low'] = ht_data['ht_higher_low']
        merged_data['ht_lower_high'] = ht_data['ht_lower_high']
        merged_data[['ht_higher_low', 'ht_lower_high']] = merged_data[['ht_higher_low', 'ht_lower_high']].ffill().infer_objects(copy=False).fillna(False)

        # --- Combine Signals ---
        # A final signal is generated only if both timeframes agree.
        buy_signal = lt_higher_low & merged_data['ht_higher_low']
        sell_signal = lt_lower_high & merged_data['ht_lower_high']

        # Create the 'signal' column.
        merged_data['Signal'] = 0
        merged_data.loc[buy_signal, 'Signal'] = 1
        merged_data.loc[sell_signal, 'Signal'] = -1

        # Clean up the helper columns.
        merged_data.drop(columns=['ht_higher_low', 'ht_lower_high'], inplace=True)

        logger.info("Harriet strategy signal generation complete.")
        
        return merged_data

    
