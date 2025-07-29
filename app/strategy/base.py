from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from app.strategy.exit_strategy import ExitStrategy
from app.trading.risk_manager import *
from app.strategy.indicators import adr

class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    All strategies should inherit from this class and implement the `calculate_signals` method.
    """

    def __init__(self, mt5_client: MT5Client, parameters: Dict[str, Any], exit_strategy: Optional[ExitStrategy] = None):
        """
        Initializes the strategy with a set of parameters and optional exit strategy.

        Args:
            parameters (Dict[str, Any]): A dictionary of parameters for the strategy.
            exit_strategy (Optional[ExitStrategy]): Optional exit strategy for SL/TP management.
        """

        # Set parameters
        self.parameters = parameters
        self.mt5_client = mt5_client

        # Get Entry parameters
        self.buy, self.sell = False, False
        self.open_buy_price, self.open_sell_price = None, None
        self.entry_time, self.exit_time = None, None

        # Get exit parameters
        self.var_buy_high, self.var_sell_high = None, None
        self.var_buy_low, self.var_sell_low = None, None
        self.exit_strategy = exit_strategy

        # Set output dictionary
        self.output_dictionary = parameters.copy()

    @abstractmethod
    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def get_trade_parameters(self, df: pd.DataFrame, df_core: pd.DataFrame, action: int, symbol_info: dict) -> pd.DataFrame:
        """
        Generates trade parameters for the strategy.
        These parameters include: "Time", "Strategy Name", "Strength", 
        "Symbol", "Signal", "ADR", "Range", "SL Pips", "TP Pips", "Lots", 
        "CurrVAR", "PropVAR", "DiffVAR".         
        Override this method to define trade parameters for the strategy.

        Args:
            df (pd.DataFrame): The historical price data.
            df_core (pd.DataFrame): The core historical price data.
            mt5_client (MT5Client): The MT5 client.
            action (int): The action to take.
            symbol_info (dict): The symbol information.

        Returns:
            Dict[str, Any]: A dictionary of trade parameters.
            pd.DataFrame: A DataFrame with trade parameters as columns.
        """
        risk_manager = RiskManager(self.mt5_client)

        df_core = adr(df_core, symbol_info)
        if df_core is None or "SL" not in df_core.columns:
            print(f"Failed to calculate ADR for {symbol_info.name}, skipping...")
            return None      
        
        stop_loss = df_core["SL"].iloc[-1]
        lots = risk_manager.calculate_position_size(stop_loss, symbol_info)
        open_positions = 0
        curr_value_at_risk = 0

        # Get the current open positions
        positions = self.mt5_client.get_positions()

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

        adr_value = df_core["ADR"].iloc[-1]
        range = df_core["daily_range"].iloc[-1]
        range_percentage = range / adr_value * 100

        str_message = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Strategy Name" : "Swing Trend Momentum",
                #"Strength": df["Trend_Strength"].iloc[-1],
                "Symbol": symbol_info.name,
                "Signal": action,
                "ADR": round(adr_value) if not pd.isna(adr_value) else 0,
                "Range": round(range_percentage) if not pd.isna(range_percentage) else 0,
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
       
     
        df_core = df_core[["daily_range", "ADR", "SL"]]
        
        # Create date-only index for merging
        df_core_date = df_core.copy()
        df_core_date.index = df_core_date.index.date
        
        df_date = df.copy()
        df_date.index = df_date.index.date
        
        # Merge on date only
        df_merged = df_date.merge(df_core_date, left_index=True, right_index=True, how='left')
        
        # Restore original datetime index
        df_merged.index = df.index
        
        # Forward fill the merged data
        df_merged["daily_range"] = df_merged["daily_range"].ffill()
        df_merged["ADR"] = df_merged["ADR"].ffill()
        df_merged["SL"] = df_merged["SL"].ffill()
        
        # Update the original df with merged data
        df["daily_range"] = df_merged["daily_range"]
        df["ADR"] = df_merged["ADR"]
        df["SL"] = df_merged["SL"]
        return str_message, df

    def get_entry_signal(self, time):
        """
        Entry signal
        :param time: TimeStamp of the row
        :return: Entry signal of the row and entry time
        """
        # If we are in the first or second columns, we do nothing
        if len(self.data.loc[:time]) < 2:
            return 0, self.entry_time

        # Create entry signal --> -1,0,1
        entry_signal = 0
        if self.data.loc[:time]["Signal"].iloc[-2] == 1:
            entry_signal = 1
        elif self.data.loc[:time]["Signal"].iloc[-2] == -1:
            entry_signal = -1

        # Enter in buy position only if we want to, and we aren't already
        if entry_signal == 1: # and not self.buy and not self.sell:
            self.buy = True
            self.open_buy_price = self.data.loc[time]["Open"]
            self.entry_time = time

        # Enter in buy position only if we want to, and we aren't already
        elif entry_signal == -1: # and not self.sell and not self.buy:
            self.sell = True
            self.open_sell_price = self.data.loc[time]["Open"]
            self.entry_time = time

        else:
            entry_signal = 0

        return entry_signal, self.entry_time

    def get_exit_signal(self, time):
        """
        Take-profit & Stop-loss exit signal
        :param time: TimeStamp of the row
        :return: P&L of the position IF we close it

        **ATTENTION**: If you allow your bot to take a buy and a sell position in the same time,
        you need to return 2 values position_return_buy AND position_return_sell (and sum both for each day)
        """
        # Verify if we need to close a position and update the variations IF we are in a buy position
        if self.buy:
            self.var_buy_high = (self.data.loc[time]["High"] - self.open_buy_price) / self.open_buy_price
            self.var_buy_low = (self.data.loc[time]["Low"] - self.open_buy_price) / self.open_buy_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_buy_high) and (self.var_buy_low < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["High_time"] < self.data.loc[time]["Low_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["Low_time"] < self.data.loc[time]["High_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                else:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = 0
                    self.exit_time = time
                    return position_return_buy, self.exit_time

            elif self.tp < self.var_buy_high:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_buy_low < self.sl:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

        # Verify if we need to close a position and update the variations IF we are in a sell position
        if self.sell:
            self.var_sell_high = -(self.data.loc[time]["High"] - self.open_sell_price) / self.open_sell_price
            self.var_sell_low = -(self.data.loc[time]["Low"] - self.open_sell_price) / self.open_sell_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_sell_low) and (self.var_sell_high < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["Low_time"] < self.data.loc[time]["High_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["High_time"] < self.data.loc[time]["Low_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                else:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = 0
                    self.exit_time = time
                    return position_return_sell, self.exit_time

            # Close with a positive P&L if high_time is before low_time
            elif self.tp < self.var_sell_low:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_sell_high < self.sl:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

        return 0, None
    
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