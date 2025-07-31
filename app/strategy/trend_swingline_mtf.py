import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy


logger = get_logger(__name__)

class TrendSwinglineMTF(BaseStrategy):
    """
    This class is the Trend Swingline MTF.
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

    def get_features(self, data: pd.DataFrame, dataH1: pd.DataFrame) -> pd.DataFrame:  
        """
        Generates Trend Swingline MTF signals for the strategy.
        Override this method to define strategy logic and signal generation.
        Signals should be stored in data['Signal'] as +1 (buy), -1 (sell), 0 (hold).

        Args:
            data (pd.DataFrame): The historical price data. 
            dataH1 (pd.DataFrame): The historical price data. 
        Returns:
            pd.DataFrame: A DataFrame with Signal column.
        """
    
        # First, calculate the swing values using SMC
        dataH1 = self.smc.calculate_swingline(dataH1)


        # Shift by 1 row to avoid look ahead bias
        dataH1 = dataH1.shift(1)

        if data is None or "Trigger" not in data.columns:
            logger.error(f"Error: Failed to calculate LTF trigger signal for {self.symbol}. Skipping...")
            return data

        if dataH1 is None or "swingline" not in dataH1.columns or "swingvalue" not in dataH1.columns:
            logger.error(f"Error: Failed to calculate HTF swingline for {self.symbol}. Skipping...")
            return data
        
        dataH1 = dataH1[["swingline", "swingvalue"]]
        dataH1 = dataH1.rename(columns={"swingline": "swingline_H1", "swingvalue": "swingvalue_H1"})

        data = data.merge(dataH1, left_index=True, right_index=True, how='left')

        data["swingline_H1"] = data["swingline_H1"].ffill()
        data["swingvalue_H1"] = data["swingvalue_H1"].ffill()

        long_cond = (
            (data["Trigger"] == 1) &
            (data["swingline_H1"] == 1)
        )

        short_cond = (
            (data["Trigger"] == -1) &
            (data["swingline_H1"] == -1)
        )

        data["Signal"] = 0
        data.loc[long_cond, "Signal"] = 1
        data.loc[short_cond, "Signal"] = -1
        
        return data

    
