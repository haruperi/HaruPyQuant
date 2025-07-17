from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from app.strategy.exit_strategy import ExitStrategy

class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    All strategies should inherit from this class and implement the `calculate_signals` method.
    """

    def __init__(self, parameters: Dict[str, Any], exit_strategy: Optional[ExitStrategy] = None):
        """
        Initializes the strategy with a set of parameters and optional exit strategy.

        Args:
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
            exit_strategy (Optional[ExitStrategy]): Optional exit strategy for SL/TP management.
        """
        self.parameters = parameters
        self.exit_strategy = exit_strategy

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
    
    def get_exit_levels(
        self, 
        data: pd.DataFrame, 
        entry_price: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Gets exit levels (SL/TP) from the exit strategy if available.
        
        Args:
            data (pd.DataFrame): Market data
            entry_price (float): Entry price
            direction (str): 'BUY' or 'SELL'
            symbol_info (Dict[str, Any]): Symbol information
            
        Returns:
            tuple[Optional[float], Optional[float]]: (stop_loss, take_profit) levels
        """
        if self.exit_strategy is None:
            return None, None
        
        return self.exit_strategy.calculate_exit_levels(data, entry_price, direction, symbol_info)
    
    def should_exit_position(
        self, 
        data: pd.DataFrame, 
        position_info: Dict[str, Any]
    ) -> bool:
        """
        Checks if a position should be exited based on the exit strategy.
        
        Args:
            data (pd.DataFrame): Current market data
            position_info (Dict[str, Any]): Current position information
            
        Returns:
            bool: True if position should be exited
        """
        if self.exit_strategy is None:
            return False
        
        return self.exit_strategy.should_exit(data, position_info) 