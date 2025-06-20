from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    All strategies should inherit from this class and implement the `calculate_signals` method.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the strategy with a set of parameters.

        Args:
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
        """
        self.parameters = parameters

    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trading signals for the given market data.

        This method must be implemented by all subclasses.

        Args:
            data (pd.DataFrame): A DataFrame with market data (OHLCV).

        Returns:
            pd.DataFrame: A DataFrame with calculated signals. The signals could be
                          represented in one or more columns (e.g., 'signal', 'buy', 'sell').
                          A common convention is:
                          - 1 for a buy signal
                          - -1 for a sell signal
                          - 0 for no signal
        """
        pass 