import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.config.setup import *
from .base import BaseStrategy


logger = get_logger(__name__)

class HoldGoldStrategy(BaseStrategy):
    """
    This class is a benchmark strategy that holds gold.
    """

    def __init__(self, mt5_client: MT5Client, parameters: Dict[str, Any]):
        """
        Initializes the strategy with the given parameters.

        Args:
            mt5_client (MT5Client): The MT5 client for data access.
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
        """
        super().__init__(mt5_client, parameters)

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:  
        """
        Generates signals for the strategy.
        Override this method to define strategy logic and signal generation.
        Signals should be stored in data['Signal'] as +1 (buy), -1 (sell), 0 (hold).

        Args:
            data (pd.DataFrame): The historical price data.

        Returns:
            pd.DataFrame: A DataFrame with Signal column.
        """
        # Generate signals
        data['Signal'] = 1
        return data

    
