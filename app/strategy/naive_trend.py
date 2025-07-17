import pandas as pd
from typing import Dict, Any
from app.config.setup import *
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


# Backtesting the strategy
if __name__ == "__main__":
    # 1. Importing here to avoid circular imports and unnecessary imports during normal execution
    from app.backtesting import Backtest, Strategy

    # 2. Backtesting functions
    def calculate_position_size(equity, risk_percent, price_distance):
        risk_amount = equity * risk_percent / 100             
        position_size = max(1000, (int(risk_amount / price_distance) // 1000) * 1000)
        #print(f"Position size: {position_size}, Equity: {equity}, Risk Amount: {risk_amount}, Price distance: {price_distance}")
        return position_size

    class NaiveTrendFollowing(Strategy):

        take_profit = 20
        stop_loss = 10

        risk_percent = RISK_PER_TRADE
        commission = 0.00005
        leverage = 10

        def init(self):
            pass


        def next(self):
            
            symbol_info = mt5_client.get_symbol_info(DEFAULT_SYMBOL)
            # Access point value directly from the symbol info object
            point_value = getattr(symbol_info, 'point', 0.00001)  # Default fallback
            price = self.data.Close[-1]

            print(f"Current Trades: {len(self.trades)}")
            
            if (self.data.Signal[-1] == 1 and not self.position):

                sl_price = price - self.stop_loss * point_value * 10
                tp_price = price + self.take_profit * point_value * 10

                position_size = calculate_position_size(self.equity, self.risk_percent, self.stop_loss* point_value * 10)
                self.buy(size=position_size, sl = sl_price, tp = tp_price)
        

            elif (self.data.Signal[-1] == -1 and not self.position):
                
                sl_price = price + self.stop_loss * point_value * 10
                tp_price = price - self.take_profit * point_value * 10

                position_size = calculate_position_size(self.equity, self.risk_percent, self.stop_loss* point_value * 10)
                self.sell(size=position_size, sl = sl_price, tp = tp_price)

    # 3. Fetching data
    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, demo=True)
    df = mt5_client.fetch_data("EURUSD", "M5", start_pos=0, end_pos=300)

    # 4. Getting strategy signals
    strategy_name = NaiveTrendFollowing
    strategy = NaiveTrendStrategy(parameters={"fast_ema_period": 12, "slow_ema_period": 24, "bias_ema_period": 72})
    if df is not None:
        signals = strategy.get_signals(df)
        print(signals)
    else:
        print("Failed to fetch data")

    # 5. Backtest
    backtest = Backtest(signals, strategy_name, cash=INITIAL_CAPITAL, margin=MARGIN, spread=SPREAD)
    stats = backtest.run()
    print(f"Backtest results: {stats}")

    # 6. Plot results
    backtest.plot(filename=os.path.join(BACKTESTS_DIR, strategy_name.__name__))
