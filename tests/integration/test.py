import os
import sys
import pandas as pd
from datetime import datetime, timedelta, UTC
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to the Python path   
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.data.mt5_client import MT5Client
from app.config.constants import DEFAULT_CONFIG_PATH, ALL_SYMBOLS
from app.strategy.indicators import SmartMoneyConcepts

class TrendLevel:
    """
    Represents a support or resistance trend line, storing its properties.
    This corresponds to the 'TrendLevel' structure in the pseudo-code.
    """
    def __init__(self, name, index_a, index_b, price):
        self.name = name
        self.index_a = index_a
        self.index_b = index_b
        self.price = price

    def __repr__(self):
        """Provides a clean string representation for printing."""
        return (f"TrendLevel(Name: {self.name}, "
                f"Start Index: {self.index_a}, "
                f"End Index: {self.index_b}, "
                f"Price: {self.price:.2f})")


class RealBreakOut:
    """
    Implements the breakout indicator logic based on swing analysis,
    as defined in the pseudo-code.
    """

    def __init__(self, bullish_color="DodgerBlue", bearish_color="Red"):
        """
        Initializes the indicator's parameters and state variables.
        Corresponds to Section 1: Initialization and Parameters.
        """
        # --- User Parameters ---
        self.bullish_color = bullish_color
        self.bearish_color = bearish_color

        # --- Internal State Properties ---
        self.last_resistance_level = None
        self.last_support_level = None
        
        # This will simulate chart drawings for demonstration
        self.active_lines = {} 

        # --- Swing Tracking Properties ---
        self.swing_direction = 0  # 0 for uninitialized
        self.highest_high = 0
        self.lowest_low = float('inf')
        self.lowest_high = float('inf')
        self.highest_low = 0
        self.index_highest_high = -1
        self.index_lowest_low = -1
        
        # --- Data store ---
        self.data = None

    def _initialize_swing_variables(self, first_bar, index):
        """
        Initializes all swing-related variables using the first data point.
        Corresponds to 'When the Indicator Starts (Initialize)'.
        """
        print("--- Indicator Initializing with First Bar ---")
        self.swing_direction = -1  # Assume initial downswing
        self.highest_high = self.lowest_high = first_bar['High']
        self.lowest_low = self.highest_low = first_bar['Low']
        self.index_highest_high = self.index_lowest_low = index

    def calculate(self, bars_data):
        """
        Processes a DataFrame of historical bar data, adds indicator columns,
        and returns the modified DataFrame.
        """
        self.data = bars_data.copy()  # Keep datetime index
        
        # Lists to hold the calculated values for each bar
        support_prices = []
        resistance_prices = []
        swing_directions = []
        breakout_signals = []

        for index, current_bar in self.data.iterrows():
            # Run main calculation for the current bar
            breakout_signal = self._calculate_single_bar(current_bar, index)
            
            # Append the current state to our lists
            support_prices.append(self.last_support_level.price if self.last_support_level else np.nan)
            resistance_prices.append(self.last_resistance_level.price if self.last_resistance_level else np.nan)
            swing_directions.append(self.swing_direction if self.swing_direction != 0 else np.nan)
            breakout_signals.append(breakout_signal)

        # Add the lists as new columns to the DataFrame
        self.data['Support'] = support_prices
        self.data['Resistance'] = resistance_prices
        self.data['Swing'] = swing_directions
        self.data['Breakout'] = breakout_signals
        
        return self.data

    def _calculate_single_bar(self, current_bar, index):
        """
        The main calculation logic executed for each individual bar.
        Now returns a breakout signal if one occurs.
        """
        # Use index (datetime) for logging
        bar_time = index
        if self.swing_direction == 0:
            self._initialize_swing_variables(current_bar, index)
            return None # No breakout on the first bar

        if self.swing_direction == 1:  # Currently in an UPSWING
            if current_bar['High'] > self.highest_high:
                self.highest_high = current_bar['High']
                self.index_highest_high = index
            if current_bar['Low'] > self.highest_low:
                self.highest_low = current_bar['Low']
            
            # Check for reversal to DOWNSWING
            if current_bar['High'] < self.highest_low:
                self.swing_direction = -1
                name = f"Resistance-{self.index_highest_high}"
                self.last_resistance_level = TrendLevel(name, self.index_highest_high, index, self.highest_high)
                self.active_lines[name] = self.last_resistance_level
                print(f"{bar_time}: Swing Reversal DOWN. New {self.last_resistance_level}")
                self.lowest_low = current_bar['Low']
                self.lowest_high = current_bar['High']
                self.index_lowest_low = index
        
        else:  # Currently in a DOWNSWING (-1)
            if current_bar['Low'] < self.lowest_low:
                self.lowest_low = current_bar['Low']
                self.index_lowest_low = index
            if current_bar['High'] < self.lowest_high:
                self.lowest_high = current_bar['High']

            # Check for reversal to UPSWING
            if current_bar['Low'] > self.lowest_high:
                self.swing_direction = 1
                name = f"Support-{self.index_lowest_low}"
                self.last_support_level = TrendLevel(name, self.index_lowest_low, index, self.lowest_low)
                self.active_lines[name] = self.last_support_level
                print(f"{bar_time}: Swing Reversal UP. New {self.last_support_level}")
                self.highest_high = current_bar['High']
                self.highest_low = current_bar['Low']
                self.index_highest_high = index

        if self.last_resistance_level:
            self.last_resistance_level.index_b = index
        if self.last_support_level:
            self.last_support_level.index_b = index

        return self._handle_breakouts(index)

    def _handle_breakouts(self, index):
        """
        Handles the breakout logic and returns a signal string.
        """
        idx_loc = self.data.index.get_loc(index)
        if idx_loc == 0:
            return None
        previous_bar_close = self.data.iloc[idx_loc - 1]['Close']
        breakout_signal = None

        # Check for a Bullish Breakout
        if self.last_resistance_level and previous_bar_close > self.last_resistance_level.price:
            print(f"{index}: BULLISH BREAKOUT of {self.last_resistance_level.name} at price {self.last_resistance_level.price:.2f}")
            self.last_resistance_level.index_b = index
            self.last_resistance_level = None
            breakout_signal = 'Bullish'

        # Check for a Bearish Breakout
        if self.last_support_level and previous_bar_close < self.last_support_level.price:
            print(f"{index}: BEARISH BREAKOUT of {self.last_support_level.name} at price {self.last_support_level.price:.2f}")
            self.last_support_level.index_b = index
            self.last_support_level = None
            breakout_signal = 'Bearish'

        return breakout_signal
    
##########################################################################################################################
# https://www.youtube.com/watch?v=v3z3FuxLzjU
# Detecting Price Trends in python - Higher Highs, Higher Lows
##########################################################################################################################

def detect_trends(df):
    """
    Detects trends in the price data using the RealBreakOut indicator.
    """
    indicator = RealBreakOut()
    result_df = indicator.calculate(df)
    return result_df



# --- --- --- --- --- --- --- --- --- ---
# --- Simulation and Demonstration ---
# --- --- --- --- --- --- --- --- --- ---


if __name__ == "__main__":
    # Initialize MT5 client
    client = MT5Client()
    
    # Fetch data
    df = client.fetch_data("GBPUSD", timeframe="M5", start_pos=0, end_pos=1000)
    print(f"Fetched {len(df)} candles for GBPUSD M5")

    # 2. Instantiate the Indicator
    #indicator = RealBreakOut()

     # 3. Run the Calculation and get the DataFrame
    # print("--- Starting Indicator Calculation ---")
    # result_df = indicator.calculate(df)
    # print("\n--- Indicator Calculation Finished ---\n")
    
    # 4. Print the final DataFrame with indicator values
    # print("--- Final DataFrame with Indicator Columns ---")
    # print(result_df.to_string())
