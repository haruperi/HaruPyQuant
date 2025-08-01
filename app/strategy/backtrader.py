import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timezone, timedelta
import sys
from app.config.setup import *
from app.backtesting import Backtest, Strategy
from .base import BaseStrategy
from .indicators import *
import plotly.graph_objects as go
from app.trading.risk_manager import *
from .bench_random import RandomStrategy
from .bench_hold_gold import HoldGoldStrategy
from .bench_naive_trend import NaiveTrendStrategy
from .harriet import HarrietStrategy
from .trend_swingline_mtf import TrendSwinglineMTF
from .trend_swingline_curr_strength import TrendSwinglineCurrStrength
from app.util.helper import rearrange_trades_columns

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

        # --------------------------------------- Risk Management ---------------------------------------
        # Filter to trade only during trading hours which is 09:00 to 18:00, else skip the trade
        if self.data.index.hour[-1] < 9 or self.data.index.hour[-1] > 18:
            return

        position_size = self.risk_manager.calculate_position_size(self.data.SL[-1], self.symbol_info)
        position_size_volume = round(position_size / self.point)    # Convert position size to volume units
        sl = self.data.SL[-1] * self.point * 10
        tp = self.data.SL[-1] * 2 * self.point * 10

      
        # --------------------------------------- Exit Logic (Optional) ---------------------------------------
        # Close any open position if the signal reverses
        # This part is optional but can be a valid approach
        # if self.position:
        #     if self.data.Signal[-1] == -1 and self.position.is_long:
        #         self.position.close()
        #     elif self.data.Signal[-1] == 1 and self.position.is_short:
        #         self.position.close()



        # --------------------------------------- Entry Logic ---------------------------------------
        # If there is no open position, check for a new signal
        if not self.position:
            # Buy signal
            if self.data.Signal[-1] == 1:
                # Calculate SL and TP based on the current price
                sl_price = round(self.data.Close[-1] - sl, self.symbol_info.digits)
                tp_price = round(self.data.Close[-1] + tp, self.symbol_info.digits)
                print(f"Buying at {self.data.Close[-1]} with SL: {sl_price} and TP: {tp_price} and size: {position_size_volume}")
                self.buy(size=position_size_volume, sl=sl_price, tp=tp_price)
            
            # Sell signal
            elif self.data.Signal[-1] == -1:
                # Calculate SL and TP based on the current price
                sl_price = round(self.data.Close[-1] + sl, self.symbol_info.digits)
                tp_price = round(self.data.Close[-1] - tp, self.symbol_info.digits)
                print(f"Selling at {self.data.Close[-1]} with SL: {sl_price} and TP: {tp_price} and size: {position_size_volume}")
                self.sell(size=position_size_volume, sl=sl_price, tp=tp_price)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Get Data ---
    # Initialize secondary MT5 client for index data (broker 3 - Purple Trading)
    mt5_client_indices = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=INDEX_SYMBOLS, broker=3)
    
    # Check secondary MT5 connection for indices
    indices_account_info = mt5_client_indices.get_account_info()
    if indices_account_info is None:
        logger.error(f"Failed to connect to indices MT5 terminal. Error code: {mt5_client_indices.mt5.last_error()}")
        index_dataframes = {}
    else:
        logger.info("Indices MT5 initialized successfully")
        logger.info(f"Indices Login: {indices_account_info['login']} \tserver: {indices_account_info['server']}")

        # Store index dataframes in a dictionary
        index_dataframes = {}
        for index in INDEX_SYMBOLS:
            index_dataframes[index] = mt5_client_indices.fetch_data(index, "H1", start_pos=START_POS, end_pos=END_POS_HTF)
            logger.info(f"Fetched {index} data: {index_dataframes[index].shape if index_dataframes[index] is not None else 'None'}")
        
    mt5_client_indices.shutdown()


    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)
    symbol = TEST_SYMBOL
    try:
        
        symbol_info = mt5_client.get_symbol_info(symbol)
        
        # Fetch data using keyword arguments
        # data = mt5_client.fetch_data(symbol=TEST_SYMBOL, timeframe=DEFAULT_TIMEFRAME, start_date=START_DATE, end_date=END_DATE)
        # data_htf = mt5_client.fetch_data(symbol=TEST_SYMBOL, timeframe=HIGHER_TIMEFRAME, start_date=START_DATE, end_date=END_DATE)
        # data_core = mt5_client.fetch_data(symbol=TEST_SYMBOL, timeframe=CORE_TIMEFRAME, start_date=START_DATE_CORE, end_date=END_DATE)

        data = mt5_client.fetch_data(symbol=symbol, timeframe=DEFAULT_TIMEFRAME, start_pos=START_POS, end_pos=END_POS)
        data_htf = mt5_client.fetch_data(symbol=symbol, timeframe=HIGHER_TIMEFRAME, start_pos=START_POS, end_pos=END_POS_HTF)
        data_core = mt5_client.fetch_data(symbol=symbol, timeframe=CORE_TIMEFRAME, start_pos=START_POS, end_pos=END_POS_D1)

        if data is None or data_htf is None or data_core is None:
            logger.error("Failed to fetch data for one or more timeframes.")


    except Exception as e:
        logger.error(f"Error fetching data: {e}")


    strategy = 4

    # 2. Generate the trading signals
    if data is not None and not data.empty and data_core is not None and not data_core.empty:
        # Random Strategy
        if strategy == 1:
            random_strategy = RandomStrategy(mt5_client, parameters={"take_profit_pips": 80, "stop_loss_pips": 40})
            data = random_strategy.get_features(data)
            str_message, data = random_strategy.get_trade_parameters(data, data_core, 1, symbol_info)

            print(f"SIGNAL DATA for {symbol}:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)


        # Naive Trend Strategy
        elif strategy == 2:
            naive_trend_strategy = NaiveTrendStrategy(mt5_client, parameters={"fast_ema_period": 12, "slow_ema_period": 24, "bias_ema_period": 72, "take_profit_pips": 40, "stop_loss_pips": 20})
            data = naive_trend_strategy.get_features(data)
            str_message, data = naive_trend_strategy.get_trade_parameters(data, data_core, 1, symbol_info)
         
            print(f"SIGNAL DATA for {symbol}:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)


        # Harriet Strategy
        elif strategy == 3:
            harriet_strategy = HarrietStrategy(mt5_client, parameters={"ht_min_dist": 5, "lt_min_dist": 2})
            data = harriet_strategy.get_features(data, data_htf, ht_min_dist=5, lt_min_dist=2, symbol_info=symbol_info)
            str_message, data = harriet_strategy.get_trade_parameters(data, data_core, 1, symbol_info)

            print("SIGNAL DATA:")
            print(data[data["Signal"] != 0])
            print("TRADING PARAMETERS:")
            print(str_message)
          

        # Trend Swingline MTF Strategy
        elif strategy == 4:
            trend_swingline_mtf_strategy = TrendSwinglineMTF(mt5_client, symbol_info, parameters={})
            trigger_signal, data = trend_swingline_mtf_strategy.get_trigger_signal(data)
            data = trend_swingline_mtf_strategy.get_features(data, data_htf)
            entry_signal, entry_time = trend_swingline_mtf_strategy.get_entry_signal(data)
            str_message, data = trend_swingline_mtf_strategy.get_trade_parameters(data, data_core, trigger_signal, symbol_info)
                
        # Trend Swingline Curr Strength Strategy
        elif strategy == 5:
            trend_swingline_curr_strength_strategy = TrendSwinglineCurrStrength(mt5_client, symbol_info, parameters={})
            trigger_signal, data = trend_swingline_curr_strength_strategy.get_trigger_signal(data)

            # Get base and quote currency symbols
            base_currency_symbol = f"{symbol[:3]}X"
            quote_currency_symbol = f"{symbol[3:]}X"
            
            # Get the corresponding dataframes from the index_dataframes dictionary
            df_base = index_dataframes[base_currency_symbol]
            df_quote = index_dataframes[quote_currency_symbol]
            
            # Get the features and signals and trade parameters
            data = trend_swingline_curr_strength_strategy.get_features(data, df_base, df_quote)
            entry_signal, entry_time = trend_swingline_curr_strength_strategy.get_entry_signal(data)
            str_message, data = trend_swingline_curr_strength_strategy.get_trade_parameters(data, data_core, trigger_signal, symbol_info)

    
        print(f"SIGNAL DATA for {symbol}:")
        print("LAST TRADING PARAMETERS:")
        print(str_message)
        # Print andSave the signal data to a CSV file
        signal_data = data[data["Signal"] != 0]
        print(signal_data)
        signal_data.to_csv(f"{DATA_DIR}/{symbol}-signals.csv")





    # 3. Run the backtest
    # print("Starting backtest...")
    
    # # Check if data is valid before running backtest
    # if data is None or data.empty:
    #     logger.error("Error: No valid data for backtesting. Please check your strategy parameters and data.")
    #     sys.exit()
    
    # # Check if data has required columns for backtesting
    # required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
    # missing_columns = [col for col in required_columns if col not in data.columns]
    # if missing_columns:
    #     logger.error(f"Error: Missing required columns for backtesting: {missing_columns}")
    #     sys.exit()
    
    # # Set class variables for the strategy
    # BackTradeStrategy.mt5_client = mt5_client
    # BackTradeStrategy.symbol_info = symbol_info
    # BackTradeStrategy.risk_manager = RiskManager(mt5_client)
    
    # bt = Backtest(
    #     data,
    #     BackTradeStrategy,
    #     cash=INITIAL_CAPITAL,
    #     spread=SPREAD,
    #     commission=.0002, # Example commission
    #     margin=MARGIN,
    #     exclusive_orders=True
    # )

    # stats = bt.run()
    # print("\n--- Backtest Results ---")
    # print(stats)
    
    # # Rearrange and print trades with desired column order
    # rearranged_trades = rearrange_trades_columns(stats._trades)
    # print("\n--- Trades (Rearranged Columns) ---")
    # print(rearranged_trades)


    # # print("\nPlotting results and saving to CSV...")
    # bt.plot(filename=os.path.join(BACKTESTS_DIR, "Backtrader"))
    #data.to_csv(f"{DATA_DIR}/combined_signals.csv")

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