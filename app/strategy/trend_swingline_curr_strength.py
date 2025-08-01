import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy


logger = get_logger(__name__)

class TrendSwinglineCurrStrength(BaseStrategy):
    """
    This class is the Trend Swingline Curr Strength.
    """

    def __init__(self, mt5_client: MT5Client, symbol_info: dict, parameters: Dict[str, Any]):
        """
        Initializes the strategy with the given parameters.

        Args:
            mt5_client (MT5Client): The MT5 client for data access.
            symbol_info (dict): The symbol information.
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
        """
        super().__init__(mt5_client, parameters, symbol_info)

    def get_features(self, data: pd.DataFrame, df_base: pd.DataFrame, df_quote: pd.DataFrame) -> pd.DataFrame:  
        """
        Generates Trend Swingline Curr Strength signals for the strategy.
        Override this method to define strategy logic and signal generation.
        Signals should be stored in data['Signal'] as +1 (buy), -1 (sell), 0 (hold).

        Args:
            data (pd.DataFrame): The historical price data. 
            df_base (pd.DataFrame): The historical price data of the base currency. 
            df_quote (pd.DataFrame): The historical price data of the quote currency. 
        Returns:
            pd.DataFrame: A DataFrame with Signal column.
        """
    
        # First, calculate the swing values using SMC
        df_base = self.smc.calculate_swingline(df_base)
        df_quote = self.smc.calculate_swingline(df_quote)

        # Keep only the following columns
        df_base = df_base[["Close", "swingline", "swingvalue"]].copy()
        df_quote = df_quote[["Close", "swingline", "swingvalue"]].copy()


        df_base = df_base.dropna()
        df_quote = df_quote.dropna()

        df_base = df_base.rename(columns={'Close': 'base_Close','swingline': 'base_swingline', 'swingvalue': 'base_swingvalue'})
        df_quote = df_quote.rename(columns={'Close': 'quote_Close','swingline': 'quote_swingline', 'swingvalue': 'quote_swingvalue'})

        # Merge The df_base and the df_quote
        curr_index_df =  df_base.merge(df_quote, left_index=True, right_index=True, how='left')

        # Calculate HT signals on the unique HT data.
        # Higher Low condition for Buy on HT
        
        # Currency signals
        strong_buy = (curr_index_df['base_Close'] > curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == 1) & (curr_index_df['quote_Close'] < curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == -1)
        mid_buy = (curr_index_df['base_Close'] > curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == 1) & (curr_index_df['quote_Close'] < curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == 1)
        weak_buy = (curr_index_df['base_Close'] > curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == -1) & (curr_index_df['quote_Close'] < curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == 1)

        strong_sell = (curr_index_df['base_Close'] < curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == -1) & (curr_index_df['quote_Close'] > curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == 1)
        mid_sell = (curr_index_df['base_Close'] < curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == -1) & (curr_index_df['quote_Close'] > curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == -1)
        weak_sell = (curr_index_df['base_Close'] < curr_index_df['base_swingvalue']) & (curr_index_df['base_swingline'] == 1) & (curr_index_df['quote_Close'] > curr_index_df['quote_swingvalue']) & (curr_index_df['quote_swingline'] == -1)

        curr_index_df['Strength'] = 0
        curr_index_df.loc[strong_buy, 'Strength'] = 3
        curr_index_df.loc[mid_buy, 'Strength'] = 2
        curr_index_df.loc[weak_buy, 'Strength'] = 1
        curr_index_df.loc[strong_sell, 'Strength'] = -3
        curr_index_df.loc[mid_sell, 'Strength'] = -2
        curr_index_df.loc[weak_sell, 'Strength'] = -1
        curr_index_df['Strength'] = curr_index_df['Strength'].shift(1)   # Shift by 1 row to avoid look ahead bias

        curr_index_df = curr_index_df[["Strength"]]
        data = data.merge(curr_index_df, left_index=True, right_index=True, how='left')
        data["Strength"] = data["Strength"].ffill()

        long_cond = (
            (data["Trigger"] == 1) &
            (data["Strength"] > 0)
        )

        short_cond = (
            (data["Trigger"] == -1) &
            (data["Strength"] < 0)
        )

        data["Signal"] = 0
        data.loc[long_cond, "Signal"] = 1
        data.loc[short_cond, "Signal"] = -1

        # Drop these columns
        data.drop(columns=["highest_low", "lowest_high"], inplace=True)
        
        return data

    
