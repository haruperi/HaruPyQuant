import pandas as pd
from typing import Dict, Any
from app.util.logger import get_logger
from .base import BaseStrategy
from .indicators import calculate_ma

logger = get_logger(__name__)

class NaiveTrendStrategy(BaseStrategy):
    """
    A naive trend-following strategy based on three Exponential Moving Averages (EMAs).
    - Fast EMA
    - Slow EMA
    - Bias EMA (to determine the overall trend direction)
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the naive trend strategy.

        Args:
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
                Expected keys: 'fast_ema_period', 'slow_ema_period', 'bias_ema_period'.
        """
        default_params = {
            'fast_ema_period': 12,
            'slow_ema_period': 48,
            'bias_ema_period': 144
        }
        default_params.update(parameters)
        super().__init__(default_params)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trading signals based on the 3-EMA strategy.

        A buy signal is generated when the fast EMA crosses above the slow EMA,
        and both are above the bias EMA.

        A sell signal is generated when the fast EMA crosses below the slow EMA,
        and both are below the bias EMA.

        Args:
            data (pd.DataFrame): A DataFrame with market data, must include a 'Close' column.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns for EMAs and 'signal'.
                          'signal' column: 1 for buy, -1 for sell, 0 for hold.
        """
        if 'Close' not in data.columns:
            logger.error("Input DataFrame must contain a 'Close' column.")
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        df = data.copy()

        fast_period = self.parameters['fast_ema_period']
        slow_period = self.parameters['slow_ema_period']
        bias_period = self.parameters['bias_ema_period']

        df = calculate_ma(df, fast_period, "EMA", 'Close')
        df = calculate_ma(df, slow_period, "EMA", 'Close')
        df = calculate_ma(df, bias_period, "EMA", 'Close')
        
        # Get the column names that were created
        fast_ema_col = f'ema_{fast_period}'
        slow_ema_col = f'ema_{slow_period}'
        bias_ema_col = f'ema_{bias_period}'

        buy_crossover = (df[fast_ema_col].shift(1) <= df[slow_ema_col].shift(1)) & (df[fast_ema_col] > df[slow_ema_col])
        buy_bias = (df[fast_ema_col] > df[bias_ema_col]) & (df[slow_ema_col] > df[bias_ema_col])
        buy_condition = buy_crossover & buy_bias

        sell_crossover = (df[fast_ema_col].shift(1) >= df[slow_ema_col].shift(1)) & (df[fast_ema_col] < df[slow_ema_col])
        sell_bias = (df[fast_ema_col] < df[bias_ema_col]) & (df[slow_ema_col] < df[bias_ema_col])
        sell_condition = sell_crossover & sell_bias

        df['Signal'] = 0
        df.loc[buy_condition, 'Signal'] = 1
        df.loc[sell_condition, 'Signal'] = -1
        logger.info(f"Completed getting signals for {self.__class__.__name__}. Number of buy signals: {buy_condition.sum()}, Number of sell signals: {sell_condition.sum()}")

        return df 