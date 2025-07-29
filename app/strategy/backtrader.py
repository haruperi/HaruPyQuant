import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timezone, timedelta
from app.config.setup import *
from app.backtesting import Backtest, Strategy
from .base import BaseStrategy
from .indicators import *
import plotly.graph_objects as go
from app.trading.risk_manager import *
from .random import RandomStrategy
from .naive_trend import NaiveTrendStrategy
from .harriet import HarrietStrategy

class BackTradeStrategy(Strategy):
    """
    BackTraderStrategy class is my custom generic class to backtest strategies.
    It executes trades based on the pre-calculated 'Signal' column.
    """
    # --- Strategy Parameters ---
    # These can be optimized by the backtesting framework.
    
    # Class variables to store MT5 client and symbol info
    mt5_client = None
    symbol_info = None
    risk_manager = None
    
    def init(self):
        """
        Initialize the strategy.
        This method is called by the Backtest framework.
        """

        
        # Point value for a 5-digit broker
        self.point = self.symbol_info.point
        
        # Pre-calculate TP and SL in price terms
        # self.tp = self.take_profit_pips * 10 * self.point
        # self.sl = self.stop_loss_pips * 10 * self.point

    def next(self):
        """
        This method is called for each bar of the data.
        It checks the signal and places trades accordingly.
        """
        # --- Risk Management ---
        # def calculate_position_size(equity, risk_percent, price_distance):
        #   risk_amount = equity * risk_percent / 100             
        #   position_size = max(1000, (int(risk_amount / price_distance) // 1000) * 1000)
        #   print(f"Position size: {position_size}, Equity: {equity}, Risk Amount: {risk_amount}, Price distance: {price_distance}")
        #   return position_size
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(self.data.SL[-1], self.symbol_info)
        # Convert position size to volume units
        position_size_volume = position_size / self.point
        sl = self.data.SL[-1] * self.point * 10
        tp = self.data.SL[-1] * 2 * self.point * 10
      

        # Close any open position if the signal reverses
        # This part is optional but can be a valid approach
        # if self.position:
        #     if self.data.Signal[-1] == -1 and self.position.is_long:
        #         self.position.close()
        #     elif self.data.Signal[-1] == 1 and self.position.is_short:
        #         self.position.close()
                
        # --- Entry Logic ---
        # If there is no open position, check for a new signal
        if not self.position:
            # Buy signal
            if self.data.Signal[-1] == 1:
                # Calculate SL and TP based on the current price
                sl_price = self.data.Close[-1] - sl
                tp_price = self.data.Close[-1] + tp
                print(f"Buying at {self.data.Close[-1]} with SL: {sl_price} and TP: {tp_price}")
                self.buy( sl=sl_price, tp=tp_price)
            
            # Sell signal
            elif self.data.Signal[-1] == -1:
                # Calculate SL and TP based on the current price
                sl_price = self.data.Close[-1] + sl
                tp_price = self.data.Close[-1] - tp
                print(f"Selling at {self.data.Close[-1]} with SL: {sl_price} and TP: {tp_price}")
                self.sell( sl=sl_price, tp=tp_price)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Get Data ---
    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)
    symbol_info = mt5_client.get_symbol_info(TEST_SYMBOL)

    data = mt5_client.fetch_data(TEST_SYMBOL, DEFAULT_TIMEFRAME, START_POS, END_POS)
    data_htf = mt5_client.fetch_data(TEST_SYMBOL, HIGHER_TIMEFRAME, START_POS, END_POS_HTF)
    data_core = mt5_client.fetch_data(TEST_SYMBOL, CORE_TIMEFRAME, START_POS, END_POS_D1)
    

    # -------------- Get Entry Signal for live trading ---dada
    # random_strategy.data = data  # Set the data attribute
    # entry_signal, entry_time = random_strategy.get_entry_signal(data.index[-1])

    

    # if entry_signal == 1:
    #     buy_price = data.loc[entry_time]["Open"]
    #     print(f"Buy at {buy_price}, entry time: {entry_time}")
    # elif entry_signal == -1:
    #     sell_price = data.loc[entry_time]["Open"]
    #     print(f"Sell at {sell_price}, entry time: {entry_time}")
    # else:
    #     print("No signal")

    # -----------------------------------------------------------
    



    # 2. Generate the trading signals
    if data is not None and not data.empty:
        strategy = 3
        if strategy == 1:
            random_strategy = RandomStrategy(mt5_client, parameters={})
            data = random_strategy.get_features(data)
            str_message, data = random_strategy.get_trade_parameters(data, data_core, 1, symbol_info)
            print("SIGNAL DATA:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)
        elif strategy == 2:
            naive_trend_strategy = NaiveTrendStrategy(mt5_client, parameters={"fast_ema_period": 12, "slow_ema_period": 24, "bias_ema_period": 72})
            data = naive_trend_strategy.get_features(data)
            str_message, data = naive_trend_strategy.get_trade_parameters(data, data_core, 1, symbol_info)
            print("SIGNAL DATA:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)
        elif strategy == 3:
            harriet_strategy = HarrietStrategy(mt5_client, parameters={"ht_min_dist": 5, "lt_min_dist": 2})
            data = harriet_strategy.get_features(data, data_htf, ht_min_dist=5, lt_min_dist=2, symbol_info=symbol_info)
            str_message, data = harriet_strategy.get_trade_parameters(data, data_core, 1, symbol_info)
            print("SIGNAL DATA:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)

        # 3. Run the backtest
        print("Starting backtest...")
        
        # Set class variables for the strategy
        BackTradeStrategy.mt5_client = mt5_client
        BackTradeStrategy.symbol_info = symbol_info
        BackTradeStrategy.risk_manager = RiskManager(mt5_client)
        
        bt = Backtest(
            data,
            BackTradeStrategy,
            cash=10000,
            spread=SPREAD,
            commission=.0002, # Example commission
            margin=MARGIN,
            exclusive_orders=True
        )

        stats = bt.run()
        print("\n--- Backtest Results ---")
        print(stats)
        print(stats._trades)

        # print("\nPlotting results...")
        bt.plot(filename=os.path.join(BACKTESTS_DIR, "Backtrader"))
        # bt.plot()

        # # Optimize the strategy with heatmap to see all results
        # print("Running optimization...")
        # optimized_bt, heatmap = bt.optimize(
        #     take_profit_pips=range(10, 50, 5), 
        #     stop_loss_pips=range(10, 50, 5),
        #     return_heatmap=True,
        #     maximize='Return [%]'  # Maximize return percentage instead of SQN
        # )
        
        # print("\n=== OPTIMIZATION RESULTS (PROFIT MAXIMIZATION) ===")
        # print("Best Parameters:")
        # print(f"  Take Profit Pips: {optimized_bt['_strategy'].take_profit_pips}")
        # print(f"  Stop Loss Pips: {optimized_bt['_strategy'].stop_loss_pips}")
        # print("\nBest Performance:")
        # print(optimized_bt)
        
        # print("\n=== TOP 10 PROFITABLE PARAMETER COMBINATIONS ===")
        # # Sort heatmap by return percentage and show top 10
        # sorted_results = heatmap.sort_values(ascending=False)
        # print("Rank | Take Profit | Stop Loss | Return [%]")
        # print("-" * 45)
        # for i, (params, score) in enumerate(sorted_results.head(10).items(), 1):
        #     tp, sl = params
        #     print(f"{i:4d} | {tp:11d} | {sl:9d} | {score:.4f}")
        
        # print(f"\nTotal parameter combinations tested: {len(heatmap)}")
        
        # # Also show the best combination by absolute profit
        # print("\n=== BEST ABSOLUTE PROFIT ===")
        # # Run optimization with absolute profit maximization
        # optimized_bt_profit, heatmap_profit = bt.optimize(
        #     take_profit_pips=[10, 15, 20, 25, 30, 35, 40, 45, 50], 
        #     stop_loss_pips=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        #     return_heatmap=True,
        #     maximize='Equity Final [$]'  # Maximize absolute profit
        # )
        # print("Best Parameters for Absolute Profit:")
        # print(f"  Take Profit Pips: {optimized_bt_profit['_strategy'].take_profit_pips}")
        # print(f"  Stop Loss Pips: {optimized_bt_profit['_strategy'].stop_loss_pips}")
        # print(f"  Final Equity: ${optimized_bt_profit['Equity Final [$]']:.2f}")
        # print(f"  Return: {optimized_bt_profit['Return [%]']:.2f}%")