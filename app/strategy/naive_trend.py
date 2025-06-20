import pandas as pd
from typing import Dict, Any

from .base import BaseStrategy
from .indicators import exponential_moving_average

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
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        df = data.copy()

        fast_period = self.parameters['fast_ema_period']
        slow_period = self.parameters['slow_ema_period']
        bias_period = self.parameters['bias_ema_period']

        df['fast_ema'] = exponential_moving_average(df['Close'], fast_period)
        df['slow_ema'] = exponential_moving_average(df['Close'], slow_period)
        df['bias_ema'] = exponential_moving_average(df['Close'], bias_period)

        buy_crossover = (df['fast_ema'].shift(1) <= df['slow_ema'].shift(1)) & (df['fast_ema'] > df['slow_ema'])
        buy_bias = (df['fast_ema'] > df['bias_ema']) & (df['slow_ema'] > df['bias_ema'])
        buy_condition = buy_crossover & buy_bias

        sell_crossover = (df['fast_ema'].shift(1) >= df['slow_ema'].shift(1)) & (df['fast_ema'] < df['slow_ema'])
        sell_bias = (df['fast_ema'] < df['bias_ema']) & (df['slow_ema'] < df['bias_ema'])
        sell_condition = sell_crossover & sell_bias

        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1

        return df 