"""
Example strategy demonstrating how to use BaseStrategy with ExitStrategy.
"""

import pandas as pd
from app.strategy.base import BaseStrategy
from app.strategy.exit_strategy import FixedPipExitStrategy, ATRExitStrategy, TrailingStopExitStrategy
from app.strategy.indicators import calculate_sma, calculate_rsi
from app.util.logger import get_logger

logger = get_logger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Example strategy using moving average crossover with various exit strategies.
    """
    
    def __init__(self, parameters: dict, exit_strategy_type: str = "fixed_pip"):
        """
        Initialize the strategy with parameters and exit strategy.
        
        Args:
            parameters (dict): Strategy parameters
            exit_strategy_type (str): Type of exit strategy ('fixed_pip', 'atr', 'trailing')
        """
        # Extract exit strategy parameters
        exit_params = self._extract_exit_parameters(parameters, exit_strategy_type)
        
        # Create exit strategy
        exit_strategy = self._create_exit_strategy(exit_strategy_type, exit_params)
        
        # Initialize base strategy
        super().__init__(parameters, exit_strategy)
    
    def _extract_exit_parameters(self, parameters: dict, exit_strategy_type: str) -> dict:
        """Extract exit strategy parameters from main parameters."""
        exit_params = {}
        
        if exit_strategy_type == "fixed_pip":
            exit_params = {
                'stop_loss_pips': parameters.get('stop_loss_pips', 50),
                'take_profit_pips': parameters.get('take_profit_pips', 100)
            }
        elif exit_strategy_type == "atr":
            exit_params = {
                'atr_period': parameters.get('atr_period', 14),
                'sl_atr_multiplier': parameters.get('sl_atr_multiplier', 2.0),
                'tp_atr_multiplier': parameters.get('tp_atr_multiplier', 3.0)
            }
        elif exit_strategy_type == "trailing":
            exit_params = {
                'trailing_distance_pips': parameters.get('trailing_distance_pips', 30),
                'trailing_step_pips': parameters.get('trailing_step_pips', 10)
            }
        
        return exit_params
    
    def _create_exit_strategy(self, exit_strategy_type: str, exit_params: dict):
        """Create the appropriate exit strategy instance."""
        if exit_strategy_type == "fixed_pip":
            return FixedPipExitStrategy(exit_params)
        elif exit_strategy_type == "atr":
            return ATRExitStrategy(exit_params)
        elif exit_strategy_type == "trailing":
            return TrailingStopExitStrategy(exit_params)
        else:
            logger.warning(f"Unknown exit strategy type: {exit_strategy_type}")
            return None
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using moving average crossover.
        """
        # Strategy parameters
        fast_period = self.parameters.get('fast_period', 10)
        slow_period = self.parameters.get('slow_period', 20)
        rsi_period = self.parameters.get('rsi_period', 14)
        rsi_oversold = self.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.parameters.get('rsi_overbought', 70)
        
        # Calculate indicators
        fast_ma = calculate_sma(data['Close'], fast_period)
        slow_ma = calculate_sma(data['Close'], slow_period)
        rsi = calculate_rsi(data['Close'], rsi_period)
        
        # Initialize signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Generate signals
        # Buy signal: fast MA crosses above slow MA and RSI is oversold
        buy_condition = (
            (fast_ma > slow_ma) & 
            (fast_ma.shift(1) <= slow_ma.shift(1)) & 
            (rsi < rsi_oversold)
        )
        
        # Sell signal: fast MA crosses below slow MA and RSI is overbought
        sell_condition = (
            (fast_ma < slow_ma) & 
            (fast_ma.shift(1) >= slow_ma.shift(1)) & 
            (rsi > rsi_overbought)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add indicator values for reference
        signals['fast_ma'] = fast_ma
        signals['slow_ma'] = slow_ma
        signals['rsi'] = rsi
        
        logger.info(f"Generated {signals['signal'].abs().sum()} signals")
        return signals


# Example usage functions
def create_strategy_with_fixed_exits():
    """Create strategy with fixed pip exits."""
    parameters = {
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'stop_loss_pips': 50,
        'take_profit_pips': 100
    }
    
    return MovingAverageCrossoverStrategy(parameters, "fixed_pip")

def create_strategy_with_atr_exits():
    """Create strategy with ATR-based exits."""
    parameters = {
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'atr_period': 14,
        'sl_atr_multiplier': 2.0,
        'tp_atr_multiplier': 3.0
    }
    
    return MovingAverageCrossoverStrategy(parameters, "atr")

def create_strategy_with_trailing_stops():
    """Create strategy with trailing stops."""
    parameters = {
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trailing_distance_pips': 30,
        'trailing_step_pips': 10
    }
    
    return MovingAverageCrossoverStrategy(parameters, "trailing") 