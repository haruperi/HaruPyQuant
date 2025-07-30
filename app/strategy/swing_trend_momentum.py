import pandas as pd
import numpy as np
import sys
from datetime import datetime
from typing import Dict, Any
from app.config.setup import *
from .base import BaseStrategy
from .indicators import *
from app.trading.risk_manager import *

logger = get_logger(__name__)

class SwingTrendMomentumStrategy(BaseStrategy):
    def __init__(self, mt5_client: MT5Client, symbol_info):
        self.mt5_client = mt5_client
        self.symbol_info = symbol_info
        self.symbol = symbol_info.name
        self.smc = SmartMoneyConcepts(self.mt5_client, self.symbol)

    def signal_trigger(self, data: pd.DataFrame, dataH1: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
        # First, calculate the swing values using SMC
        data = self.smc.calculate_swingline(data)
        dataH1 = self.smc.calculate_swingline(dataH1)

        if data is None or "swingline" not in data.columns or "swingvalue" not in data.columns:
            logger.error(f"Error: Failed to calculate swingline for {self.symbol}. Skipping...")
            return False, data

        if dataH1 is None or "swingline" not in dataH1.columns or "swingvalue" not in dataH1.columns:
            logger.error(f"Error: Failed to calculate swingline for {self.symbol}. Skipping...")
            return False, data
            
        dataH1 = dataH1[["swingline", "swingvalue"]]
        dataH1 = dataH1.rename(columns={"swingline": "swingline_H1", "swingvalue": "swingvalue_H1"})

        data = data.merge(dataH1, left_index=True, right_index=True, how='left')

        data["swingline_H1"] = data["swingline_H1"].ffill()
        data["swingvalue_H1"] = data["swingvalue_H1"].ffill()

        data["Trigger"] = 0
        Signal_Trigger = False

        long_cond = (
            (data["Close"].shift(1) < data["swingvalue"].shift(1)) &
            (data["Close"] > data["swingvalue"]) &
            (data["swingline"] == -1) &
            (data["swingvalue"] > data["swingvalue_H1"])
        )

        short_cond = (
            (data["Close"].shift(1) > data["swingvalue"].shift(1)) &
            (data["Close"] < data["swingvalue"]) &
            (data["swingline"] == 1) &
            (data["swingvalue"] < data["swingvalue_H1"])
        )

        data.loc[long_cond, "Trigger"] = 1
        data.loc[short_cond, "Trigger"] = -1

        # Check if the previous completed candle has a signal
        if data["Trigger"].iloc[-2] != 0:
            Signal_Trigger = True

        return Signal_Trigger, data

    def get_signals(self, data: pd.DataFrame, df_base: pd.DataFrame, df_quote: pd.DataFrame) -> pd.DataFrame:

        df_base = self.smc.calculate_swingline(df_base)
        df_quote = self.smc.calculate_swingline(df_quote)

        

         # Initialize trend columns
        Signal_Trigger = False
        action = 0  # Initialize action variable
        data["Signal"] = 0
        data["Trend_Strength"] = 0
        df_base["base_trend"] = 0
        df_quote["quote_trend"] = 0

        # Strong Bullish = 2: swingline bullish AND price above swing value (strong momentum)
        # Weak Bullish = 1: swingline bullish BUT price below swing value (weak momentum)
        # Strong Bearish = -2: swingline bearish AND price below swing value (strong momentum)
        # Weak Bearish = -1: swingline bearish BUT price above swing value (weak momentum)
        base_strong_bullish = (df_base["swingline"] == 1) & (df_base["Close"] > df_base["swingvalue"])
        base_weak_bullish = (df_base["swingline"] == 1) & (df_base["Close"] <= df_base["swingvalue"])
        base_strong_bearish = (df_base["swingline"] == -1) & (df_base["Close"] < df_base["swingvalue"])
        base_weak_bearish = (df_base["swingline"] == -1) & (df_base["Close"] >= df_base["swingvalue"])

        df_base.loc[base_strong_bullish, "base_trend"] = 2
        df_base.loc[base_weak_bullish, "base_trend"] = 1
        df_base.loc[base_strong_bearish, "base_trend"] = -2
        df_base.loc[base_weak_bearish, "base_trend"] = -1

        quote_strong_bullish = (df_quote["swingline"] == 1) & (df_quote["Close"] > df_quote["swingvalue"])
        quote_weak_bullish = (df_quote["swingline"] == 1) & (df_quote["Close"] <= df_quote["swingvalue"])   
        quote_strong_bearish = (df_quote["swingline"] == -1) & (df_quote["Close"] < df_quote["swingvalue"])
        quote_weak_bearish = (df_quote["swingline"] == -1) & (df_quote["Close"] >= df_quote["swingvalue"])

        df_quote.loc[quote_strong_bullish, "quote_trend"] = 2
        df_quote.loc[quote_weak_bullish, "quote_trend"] = 1
        df_quote.loc[quote_strong_bearish, "quote_trend"] = -2
        df_quote.loc[quote_weak_bearish, "quote_trend"] = -1

        df_base = df_base[["base_trend", "swingline"]]
        df_quote = df_quote[["quote_trend", "swingline"]]

        df_base = df_base.rename(columns={"swingline": "swingline_base"})
        df_quote = df_quote.rename(columns={"swingline": "swingline_quote"})

        df_base['base_trend'] = df_base['base_trend'].shift(1)
        df_quote['quote_trend'] = df_quote['quote_trend'].shift(1)
        df_base['swingline_base'] = df_base['swingline_base'].shift(1)
        df_quote['swingline_quote'] = df_quote['swingline_quote'].shift(1)

        

        data = data.merge(df_base, left_index=True, right_index=True, how='left')
        data = data.merge(df_quote, left_index=True, right_index=True, how='left')

        data["base_trend"] = data["base_trend"].ffill()
        data["quote_trend"] = data["quote_trend"].ffill()
        data["swingline_base"] = data["swingline_base"].ffill()
        data["swingline_quote"] = data["swingline_quote"].ffill()

        data["Trend_Strength"] = data["base_trend"] + data["quote_trend"]

        buy_cond = (data["Trigger"] == 1) & (data["base_trend"] == 2) & (data["quote_trend"] < 0)
        sell_cond = (data["Trigger"] == -1) & (data["base_trend"] == -2) & (data["quote_trend"] > 0)    

        data.loc[buy_cond, "Signal"] = 1
        data.loc[sell_cond, "Signal"] = -1

        # Check if the previous completed candle has a signal
        if data["Signal"].iloc[-2] != 0:
            Signal_Trigger = True
            action = data["Signal"].iloc[-2]

        return Signal_Trigger, action, data


    def get_parameters(self, df: pd.DataFrame, df_core: pd.DataFrame, mt5_client: MT5Client, action: int, symbol_info: dict) -> pd.DataFrame:
        risk_manager = RiskManager(mt5_client)

        df_core = adr(df_core, symbol_info)
        if df_core is None or "SL" not in df_core.columns:
            print(f"Failed to calculate ADR for {symbol_info.name}, skipping...")
            return None
        
        stop_loss = df_core["SL"].iloc[-1]
        lots = risk_manager.calculate_position_size(stop_loss, symbol_info)
        open_positions = 0
        curr_value_at_risk = 0

        # Get the current open positions
        positions = mt5_client.get_positions()

        if positions is not None and len(positions) > 0:
            # Iterate through the positions and save them to the dictionary
            for position in positions:
                # Handle dictionary positions (MT5 returns dict via _asdict())
                vol_lots = -position['volume'] if position['type'] == 1 else position['volume']
                risk_manager.add_position(position['symbol'], vol_lots)
                open_positions = open_positions + 1

            curr_value_at_risk = risk_manager.run()

        lots = lots if action == 1 else -lots
        risk_manager.add_position(symbol_info.name, lots)
        proposed_value_at_risk = risk_manager.run()

        if open_positions == 0 or curr_value_at_risk == 0:
            incr_var = 100
        else:
            incr_var = ((proposed_value_at_risk - curr_value_at_risk) / curr_value_at_risk) * 100

        adr = df_core["ADR"].iloc[-1]
        range = df_core["daily_range"].iloc[-1]
        range_percentage = range / adr * 100

        str_message = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Strategy Name" : "Swing Trend Momentum",
                "Strength": df["Trend_Strength"].iloc[-1],
                "Symbol": symbol_info.name,
                "Signal": action,
                "ADR": round(adr),
                "Range": round(range_percentage),
                "SL Pips": stop_loss,
                "TP Pips": stop_loss * 2,
                "Lots": lots,
                "CurrVAR": f"${round(curr_value_at_risk):,.2f}",
                "PropVAR": f"${round(proposed_value_at_risk):,.2f}",
                "DiffVAR": f"{round(incr_var)}%",
            }

        if action == 1:
            str_message["Price"] = symbol_info.ask
            str_message["SL Price"] = round(symbol_info.bid - stop_loss * 10 * symbol_info.trade_tick_size, symbol_info.digits)
            str_message["TP Price"] = round(symbol_info.ask + stop_loss * 2 * 10 * symbol_info.trade_tick_size, symbol_info.digits)
        else:
            str_message["Price"] = symbol_info.bid
            str_message["SL Price"] = round(symbol_info.ask + stop_loss * 10 * symbol_info.trade_tick_size, symbol_info.digits)
            str_message["TP Price"] = round(symbol_info.bid - stop_loss * 2 * 10 * symbol_info.trade_tick_size, symbol_info.digits)
       


        return str_message, df




if __name__ == "__main__":
    print("Swing Trend Momentum Strategy Testing")
    ############################################################################################################################################

    # For Testing Single Symbol

    # 1. Get data

     # Initialize secondary MT5 client for index data (broker 3 - Purple Trading)
    mt5_client_indices = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=INDEX_SYMBOLS, broker=3)
    
    # Check secondary MT5 connection for indices
    indices_account_info = mt5_client_indices.get_account_info()
    if indices_account_info is None:
        logger.warning(f"Failed to connect to indices MT5 terminal. Error code: {mt5_client_indices.mt5.last_error()}")
        logger.warning("Index data will not be available. Using main MT5 for all data.")
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

    # Initialize main MT5 client for trading data (broker 2 - Demo account)
    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=2)
    
    # Check main MT5 connection
    current_account_info = mt5_client.get_account_info()
    if current_account_info is None:
        logger.error(f"Failed to connect to main MT5 terminal. Error code: {mt5_client.mt5.last_error()}")
        logger.error(f"Please ensure MT5 is running and you are logged in to a trading account.")
        df = None
        df_H1 = None
        symbol_info = None
    else:
        logger.info("Main MT5 initialized successfully")
        logger.info(f"Login: {current_account_info['login']} \tserver: {current_account_info['server']}")
        logger.info(f"Balance: {current_account_info['balance']} USD, \t Equity: {current_account_info['equity']} USD, \t Profit: {current_account_info['profit']} USD")
        
        symbol_info = mt5_client.get_symbol_info(TEST_SYMBOL)
        if not symbol_info or not hasattr(symbol_info, "trade_tick_size"):
            logger.error(f"Error: Invalid symbol_info for {TEST_SYMBOL}. Skipping...")

        df = mt5_client.fetch_data(TEST_SYMBOL, "M5", start_pos=START_POS, end_pos=END_POS)
        df_H1 = mt5_client.fetch_data(TEST_SYMBOL, "H1", start_pos=START_POS, end_pos=END_POS_HTF)
        
        logger.info(f"Fetched {TEST_SYMBOL} M5 data: {df.shape if df is not None else 'None'}")
        logger.info(f"Fetched {TEST_SYMBOL} H1 data: {df_H1.shape if df_H1 is not None else 'None'}")
        
    print(f"Main symbol data:")
    print(f"  M5: {df if df is not None else 'None'}")
    print(f"  H1: {df_H1 if df_H1 is not None else 'None'}")

    # Get base and quote currency symbols
    base_currency_symbol = f"{TEST_SYMBOL[:3]}X"
    quote_currency_symbol = f"{TEST_SYMBOL[3:]}X"

    print(f"Currency symbols:")
    print(f"  Base: {base_currency_symbol}")
    print(f"  Quote: {quote_currency_symbol}")
    
    # Get the corresponding dataframes from the index_dataframes dictionary
    if base_currency_symbol in index_dataframes:
        df_base = index_dataframes[base_currency_symbol]
        print(f"Base currency data: {df_base if df_base is not None else 'None'}")
    else:
        print(f"Base currency {base_currency_symbol} not found in index data")
        df_base = None
        
    if quote_currency_symbol in index_dataframes:
        df_quote = index_dataframes[quote_currency_symbol]
        print(f"Quote currency data: {df_quote if df_quote is not None else 'None'}")
    else:
        print(f"Quote currency {quote_currency_symbol} not found in index data")
        df_quote = None

    # Check if we have all required data before proceeding
    if df is None or df_H1 is None:
        logger.error("Failed to fetch required data for main symbol. Cannot proceed with strategy testing.")
        mt5_client.shutdown()
        sys.exit(1)
    
    if df_base is None or df_quote is None:
        logger.warning("Some currency index data is missing. Strategy may not work optimally.")
        logger.warning("Consider checking broker 3 connection for index data.")



    # strategy = SwingTrendMomentumStrategy(mt5_client, symbol_info)
    # trigger, df = strategy.signal_trigger(df, df_H1)
   
    

    # signal, df = strategy.get_signals(df, df_base, df_quote)
    # action = df["Signal"].iloc[-2]
    

    # df_core = mt5_client.fetch_data(TEST_SYMBOL, CORE_TIMEFRAME, start_pos=START_POS, end_pos=END_POS_D1)

                        
    # str_message, df = strategy.get_parameters(df, df_core, mt5_client, action, symbol_info)
    # print(df.tail(36))
    # print(str_message)



    ############################################################################################################################################

    # For Testing All Symbols

    # for symbol in FOREX_SYMBOLS:

    #     symbol_info = mt5_client.get_symbol_info(symbol)
    #     if not symbol_info or not hasattr(symbol_info, "trade_tick_size"):
    #         logger.error(f"Error: Invalid symbol_info for {symbol}. Skipping...")
    #         continue


    #     df = mt5_client.fetch_data(symbol, "M5", start_pos=START_POS, end_pos=END_POS)
    #     df_H1 = mt5_client.fetch_data(symbol, "H1", start_pos=START_POS, end_pos=END_POS_HTF)
        
    #     if df is not None and df_H1 is not None:
    #         strategy = SwingTrendMomentumStrategy(mt5_client, symbol_info)
    #         trigger, df = strategy.signal_trigger(df, df_H1)
    #         if not trigger:
    #             logger.info(f"No trigger for {symbol}, skipping...")
    #             continue
    #         else:
    #             base_currency = symbol[:3]
    #             quote_currency = symbol[3:]

    #             df_base = mt5_client.fetch_data(f"{base_currency}X", "H1", start_pos=START_POS, end_pos=END_POS_HTF)
    #             df_quote = mt5_client.fetch_data(f"{quote_currency}X", "H1", start_pos=START_POS, end_pos=END_POS_HTF)

    #             if df_base is not None and df_quote is not None:
                    
    #                 logger.info(f"Processing signals for {symbol}")
    #                 logger.info(f"df_base: {base_currency}X")
    #                 logger.info(f"df_quote: {quote_currency}X")
    #                 signal, df = strategy.get_signals(df, df_base, df_quote)
    #                 if not signal:
    #                     logger.info(f"No signal for {symbol}, skipping...")
    #                     continue
    #                 else:
    #                     df_core = mt5_client.fetch_data(symbol, CORE_TIMEFRAME, start_pos=START_POS, end_pos=END_POS_D1)
    #                     if df_core is None:
    #                         print(f"Failed to fetch {CORE_TIMEFRAME} data for {symbol}, skipping...")
    #                         continue
                        
    #                     df = strategy.get_parameters(df, df_core)
    #                     print(df)
    #             else:
    #                 if df_base is None:
    #                     logger.error(f"Error: Failed to fetch base currency data for {base_currency}X")
    #                 if df_quote is None:
    #                     logger.error(f"Error: Failed to fetch quote currency data for {quote_currency}X")
    #                 continue
    #     else:
    #         if df is None:
    #             logger.error(f"Error: Failed to fetch M5 data for {symbol}. Skipping...")
    #         if df_H1 is None:
    #             logger.error(f"Error: Failed to fetch H1 data for {symbol}. Skipping...")
    #         continue