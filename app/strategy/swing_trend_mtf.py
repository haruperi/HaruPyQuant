import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy
from .indicators import SmartMoneyConcepts


logger = get_logger(__name__)

class SwingTrendMTF(BaseStrategy):
    """
    This class is the Swing Trend MTF.
    """

    def __init__(self, mt5_client: MT5Client, symbol_info: dict, parameters: Dict[str, Any]):
        """
        Initializes the strategy with the given parameters.

        Args:
            mt5_client (MT5Client): The MT5 client for data access.
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
        """
        self.mt5_client = mt5_client
        self.symbol_info = symbol_info
        self.symbol = symbol_info.name
        self.smc = SmartMoneyConcepts(self.mt5_client, self.symbol)
        super().__init__(mt5_client, parameters)

    def get_features(self, dataM1: pd.DataFrame, dataM5: pd.DataFrame, dataH1: pd.DataFrame) -> pd.DataFrame:  
        """
        Generates Swing Trend MTF signals for the strategy.
        Override this method to define strategy logic and signal generation.
        Signals should be stored in data['Signal'] as +1 (buy), -1 (sell), 0 (hold).

        Args:
            dataM1 (pd.DataFrame): The historical price data. 
            dataM5 (pd.DataFrame): The historical price data. 
            dataH1 (pd.DataFrame): The historical price data. 
        Returns:
            pd.DataFrame: A DataFrame with Signal column.
        """
    
        # First, calculate the swing values using SMC
        dataM5 = self.smc.calculate_swingline(dataM5)
        dataH1 = self.smc.calculate_swingline(dataH1)


        # Shift by 1 row to avoid look ahead bias
        dataM5 = dataM5.shift(1)
        dataH1 = dataH1.shift(1)

        if dataM5 is None or "swingline" not in dataM5.columns or "swingvalue" not in dataM5.columns:
            logger.error(f"Error: Failed to calculate swingline for {self.symbol}. Skipping...")
            return False, dataM1

        if dataH1 is None or "swingline" not in dataH1.columns or "swingvalue" not in dataH1.columns:
            logger.error(f"Error: Failed to calculate swingline for {self.symbol}. Skipping...")
            return False, dataM1

        dataM5 = dataM5[["swingline", "swingvalue"]]
        dataM5 = dataM5.rename(columns={"swingline": "swingline_M5", "swingvalue": "swingvalue_M5"})

        dataH1 = dataH1[["swingline", "swingvalue"]]
        dataH1 = dataH1.rename(columns={"swingline": "swingline_H1", "swingvalue": "swingvalue_H1"})

        dataM5 = dataM5.merge(dataH1, left_index=True, right_index=True, how='left')
        dataM1 = dataM1.merge(dataM5, left_index=True, right_index=True, how='left')

        dataM1["swingline_M5"] = dataM1["swingline_M5"].ffill()
        dataM1["swingvalue_M5"] = dataM1["swingvalue_M5"].ffill()
        dataM1["swingline_H1"] = dataM1["swingline_H1"].ffill()
        dataM1["swingvalue_H1"] = dataM1["swingvalue_H1"].ffill()

        long_cond = (
            (dataM1["Close"].shift(1) < dataM1["swingvalue_M5"].shift(1)) &
            (dataM1["Close"] > dataM1["swingvalue_M5"]) &
            (dataM1["swingline_M5"] == -1) &
            (dataM1["swingline_H1"] == 1)
        )

        short_cond = (
            (dataM1["Close"].shift(1) > dataM1["swingvalue_M5"].shift(1)) &
            (dataM1["Close"] < dataM1["swingvalue_M5"]) &
            (dataM1["swingline_M5"] == 1) &
            (dataM1["swingline_H1"] == -1)
        )

        dataM1["Signal"] = 0
        dataM1.loc[long_cond, "Signal"] = 1
        dataM1.loc[short_cond, "Signal"] = -1
        
        return dataM1

    
