import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy


logger = get_logger(__name__)

class MeanReversionHODLOD(BaseStrategy):
    """
    This class is the Mean Reversion, we sell at the Highest Of Day - HOD and buy at the Lowest Of Day - LOD.
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

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:  
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
        data = self.smc.calculate_swingline(data)

        long_cond = (
            (data["Trigger"] == 1) & (data["Reversal_Trigger"] == 1)
        )

        short_cond = (
            (data["Trigger"] == -1) & (data["Reversal_Trigger"] == -1)
        )

        data["Signal"] = 0
        data.loc[long_cond, "Signal"] = 1
        data.loc[short_cond, "Signal"] = -1
        
        return data

    
