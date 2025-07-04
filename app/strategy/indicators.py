import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import MetaTrader5 as mt5

# Standard indicators
def ema_series(data: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Args:
        data (pd.Series): A pandas Series of prices (e.g., 'Close' prices).
        period (int): The EMA period (span).

    Returns:
        pd.Series: A pandas Series with the calculated EMA values.
    """
    return data.ewm(span=period, adjust=False).mean()

def sma_series(data: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        data (pd.Series): A pandas Series of prices (e.g., 'Close' prices).
        period (int): The SMA period (window).

    Returns:
        pd.Series: A pandas Series with the calculated SMA values.
    """
    return data.rolling(window=period).mean()

def rsi_series(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.Series): A pandas Series of prices (e.g., 'Close' prices).
        period (int): The RSI period. Default is 14.

    Returns:
        pd.Series: A pandas Series with the calculated RSI values.
    """
    delta = data.diff()

    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi 


# Custom indicators
def adr_series(high: pd.Series, low: pd.Series, n: int, symbol_info: dict) -> pd.Series:
    """
    Returns `n`-period average daily range of array `arr`.
    """

    # Calculate daily ranges
    daily_range = (high - low) / symbol_info.trade_tick_size / 10

    # Calculate ADR
    adr = daily_range.rolling(window=n).mean()

    # Shift the ADR by one period to make today's ADR based on the previous value
    adr = adr.shift(1)

    return adr



def calculate_swingline_pivot_points(df):
    """
    Identifies fractal pivot points in the DataFrame based on swingline directions.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'swingline' column, as well as 'high' and 'low' columns.

    Returns:
    pd.DataFrame: The input DataFrame with an added 'isPivot' column.
                  Each pivot point is marked as:
                  -1 for a low pivot (local minima) in a downward swing,
                   1 for a high pivot (local maxima) in an upward swing,
                   and NaN for non-pivot rows.
    """

    # Initialize 'isPivot' column with NaN
    df['isPivot'] = np.nan

    # Create groups of consecutive 'swingline' values
    group_ids = (df['swingline'] != df['swingline'].shift()).cumsum()
    groups = df.groupby(group_ids)

    # Iterate over each group
    for group_id, group_data in groups:
        sig_value = group_data['swingline'].iloc[0]

        if sig_value == -1:
            # Find index of the minimum 'low' in this group
            min_low_idx = group_data['Low'].idxmin()
            df.at[min_low_idx, 'isPivot'] = -1
        elif sig_value == 1:
            # Find index of the maximum 'high' in this group
            max_high_idx = group_data['High'].idxmax()
            df.at[max_high_idx, 'isPivot'] = 1

    return df

############### Smart Money Concepts ###############

class SmartMoneyConcepts:
    """
    Smart Money Concepts (SMC) analysis class.
    
    This class provides methods to identify and analyze smart money movements
    in financial markets, including swing highs/lows, pivot points, order blocks,
    fair value gaps, and liquidity zones.
    """
    def __init__(self, symbol: str, min_swing_length: int = 3, min_pip_range: int = 2):
        """
        Initialize the SmartMoneyConcepts analyzer.
        """
        self._logger = None  # Will be set up when needed
        self._setup_logger()
        self.symbol = symbol
        self.pip_value = mt5.symbol_info(self.symbol).point * 10  # Convert point to pip value
        self.min_swing_length = min_swing_length
        self.min_pip_range = min_pip_range
    def _setup_logger(self):
        """Set up logger for the SMC class."""
        try:
            from app.util.logger import get_logger
            self._logger = get_logger(__name__)
        except ImportError:
            import logging
            self._logger = logging.getLogger(__name__)

    def calculate_swingline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates swing trend lines and adds them as columns to a DataFrame.

        This function processes a DataFrame with 'High' and 'Low' price columns,
        identifies market swing direction, and adds the following columns:
        - swingline: The swing direction (1 for upswing, -1 for downswing).
        - swing_value: The peak of the current swing (HighestHigh for upswings,
        LowestLow for downswings).
        - highest_low: The highest low point reached during the current upswing.
        - lowest_high: The lowest high point reached during the current downswing.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'High' and 'Low' columns.

        Returns:
            pd.DataFrame: The DataFrame with the four new columns added.
        """
        if 'High' not in df.columns or 'Low' not in df.columns:
            raise ValueError("Input DataFrame must contain 'High' and 'Low' columns.")
        
        if self._logger is None:
            self._setup_logger()
            
        self._logger.info("Calculating swingline for DataFrame")

        if len(df) < 2:
            self._logger.warning("DataFrame too short for swingline calculation")
            return df

        # Prepare lists to hold the calculated values for the new columns
        swingline = [np.nan] * len(df)
        swing_value = [np.nan] * len(df)
        highest_low_col = [np.nan] * len(df)
        lowest_high_col = [np.nan] * len(df)

        # --- Initialize State Variables from the first row ---
        # The logic starts with swing_direction = -1, so we begin in a downswing state.
        swing_direction = -1
        HighestHigh = df['High'].iloc[0]
        LowestLow = df['Low'].iloc[0]
        LowestHigh = df['High'].iloc[0]
        HighestLow = df['Low'].iloc[0]

        # --- Set initial values for the first row ---
        swingline[0] = swing_direction
        swing_value[0] = LowestLow  # In a downswing, swing_value is LowestLow
        highest_low_col[0] = HighestLow
        lowest_high_col[0] = LowestHigh
        
        # --- Process the rest of the DataFrame row by row ---
        for i in range(1, len(df)):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]

            if swing_direction == 1:
                # --- LOGIC FOR AN ACTIVE UPSWING ---
                if high > HighestHigh:
                    HighestHigh = high
                if low > HighestLow:
                    HighestLow = low
                
                # Check for a swing change to DOWN
                if high < HighestLow:
                    swing_direction = -1  # Change direction to downswing
                    LowestLow = low
                    LowestHigh = high
            else:  # swing_direction == -1
                # --- LOGIC FOR AN ACTIVE DOWNSWING ---
                if low < LowestLow:
                    LowestLow = low
                if high < LowestHigh:
                    LowestHigh = high
                
                # Check for a swing change to UP
                if low > LowestHigh:
                    swing_direction = 1  # Change direction to upswing
                    HighestHigh = high
                    HighestLow = low

            # Append the current state to our lists
            swingline[i] = swing_direction
            highest_low_col[i] = HighestLow
            lowest_high_col[i] = LowestHigh
            
            # Determine the swing_value based on the current swing direction
            if swing_direction == 1:
                swing_value[i] = HighestLow
            else:
                swing_value[i] = LowestHigh

        # Add the lists as new columns to the DataFrame
        df['swingline'] = swingline
        df['swing_value'] = swing_value
        df['highest_low'] = highest_low_col
        df['lowest_high'] = lowest_high_col

        self._logger.info(f"Swingline calculation completed. Final swingline: { df['swingline'].iloc[-1]}")
        
        return df
    

    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify fractal pivot points based on swingline directions.
        
        Args:
            df (pd.DataFrame): DataFrame with swingline column and OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with added pivot point indicators
        """
        if self._logger is None:
            self._setup_logger()
            
        self._logger.info("Calculating pivot points")
        
        df = df.copy()
        df['isPivot'] = np.nan

        # Create groups of consecutive swingline values and clean up the data
        group_ids = (df['swingline'] != df['swingline'].shift()).cumsum()
        groups = df.groupby(group_ids)

        for group_id, group_data in groups:
            # Skip groups where all swing_value values are the same (flat/ranging market)
            if group_data['swing_value'].nunique() == 1:
                continue

            # Skip groups with less than min_swing_length candles
            if len(group_data) < self.min_swing_length:
                continue

            # Skip if the range of swingline values is less than min_pip_range pips
            swingline_range = group_data['swing_value'].max() - group_data['swing_value'].min()
            if swingline_range < (self.min_pip_range * self.pip_value):
                continue
                
            sig_value = group_data['swingline'].iloc[0]

            if sig_value == -1:
                # Find index of the minimum low in this group
                min_low_idx = group_data['Low'].idxmin()
                df.at[min_low_idx, 'isPivot'] = -1
                
            elif sig_value == 1:
                # Find index of the maximum high in this group
                max_high_idx = group_data['High'].idxmax()
                df.at[max_high_idx, 'isPivot'] = 1

        # Forward fill the isPivot column
        df['swingline'] = df['isPivot'].ffill()
        df['isPivot'] = np.nan

        # Create groups of consecutive swingline values and calculate final pivot points
        group_ids = (df['swingline'] != df['swingline'].shift()).cumsum()
        groups = df.groupby(group_ids)

        for group_id, group_data in groups:
            sig_value = group_data['swingline'].iloc[0]

            if sig_value == -1:
                # Find index of the minimum low in this group
                min_low_idx = group_data['Low'].idxmin()
                df.at[min_low_idx, 'isPivot'] = -1
                
            elif sig_value == 1:
                # Find index of the maximum high in this group
                max_high_idx = group_data['High'].idxmax()
                df.at[max_high_idx, 'isPivot'] = 1

        self._logger.info(f"Pivot points calculation completed.")

        return df
    
    def identify_fair_value_gaps(self, df: pd.DataFrame, join_consecutive: bool = True) -> pd.DataFrame:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom

        Returns:
            pd.DataFrame: DataFrame with FVG indicator Columns
            fvg_type = 1 if bullish fair value gap, -1 if bearish fair value gap
            fvg_high = the top of the fair value gap
            fvg_low = the bottom of the fair value gap
            fvg_size = size of the gap (uses mt5 pip value)
            MitigatedIndex = the index of the candle that mitigated the fair value gap
        """
        if self._logger is None:
            self._setup_logger()
            
        self._logger.info("Identifying fair value gaps")
        
        df = df.copy()
        
        # Initialize FVG columns
        df['fvg_type'] = np.nan
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan
        df['fvg_size'] = np.nan
        df['MitigatedIndex'] = None
        
        if len(df) < 3:
            self._logger.warning("DataFrame too short for FVG calculation")
            return df
        
        fvg_list = []
        
        # Identify FVGs
        for i in range(1, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Check for bullish FVG (previous high < next low and current candle is bullish)
            if (prev['High'] < next_candle['Low'] and 
                current['Close'] > current['Open']):
                
                fvg_list.append({
                    'index': i,
                    'type': 1,  # bullish
                    'high': next_candle['Low'],
                    'low': prev['High'],
                    'size': (next_candle['Low'] - prev['High']) / self.pip_value
                })
                
            # Check for bearish FVG (previous low > next high and current candle is bearish)
            elif (prev['Low'] > next_candle['High'] and 
                  current['Close'] < current['Open']):
                
                fvg_list.append({
                    'index': i,
                    'type': -1,  # bearish
                    'high': prev['Low'],
                    'low': next_candle['High'],
                    'size': (prev['Low'] - next_candle['High']) / self.pip_value
                })
        
        # Join consecutive FVGs if requested
        if join_consecutive and fvg_list:
            merged_fvgs = []
            current_group = [fvg_list[0]]
            
            for fvg in fvg_list[1:]:
                if fvg['index'] == current_group[-1]['index'] + 1:
                    # Consecutive FVG, add to current group
                    current_group.append(fvg)
                else:
                    # Non-consecutive, process current group and start new one
                    if len(current_group) > 1:
                        # Merge consecutive FVGs
                        merged_fvg = {
                            'index': current_group[0]['index'],
                            'type': current_group[0]['type'],
                            'high': max(f['high'] for f in current_group),
                            'low': min(f['low'] for f in current_group),
                            'size': (max(f['high'] for f in current_group) - min(f['low'] for f in current_group)) / self.pip_value
                        }
                        merged_fvgs.append(merged_fvg)
                    else:
                        merged_fvgs.append(current_group[0])
                    current_group = [fvg]
            
            # Process the last group
            if len(current_group) > 1:
                merged_fvg = {
                    'index': current_group[0]['index'],
                    'type': current_group[0]['type'],
                    'high': max(f['high'] for f in current_group),
                    'low': min(f['low'] for f in current_group),
                    'size': (max(f['high'] for f in current_group) - min(f['low'] for f in current_group)) / self.pip_value
                }
                merged_fvgs.append(merged_fvg)
            else:
                merged_fvgs.append(current_group[0])
            
            fvg_list = merged_fvgs
        
        # Add FVGs to DataFrame and check for mitigation
        for fvg in fvg_list:
            idx = fvg['index']
            real_idx = df.index[idx]
            df.at[real_idx, 'fvg_type'] = fvg['type']
            df.at[real_idx, 'fvg_high'] = fvg['high']
            df.at[real_idx, 'fvg_low'] = fvg['low']
            df.at[real_idx, 'fvg_size'] = round(fvg['size'], 1)
            
            # Check for mitigation (price filling the gap)
            for j in range(idx + 1, len(df)):
                candle = df.iloc[j]
                if fvg['type'] == 1:  # Bullish FVG
                    if candle['Low'] <= fvg['low']:
                        df.at[real_idx, 'MitigatedIndex'] = str(df.index[j])
                        break
                else:  # Bearish FVG
                    if candle['High'] >= fvg['high']:
                        df.at[real_idx, 'MitigatedIndex'] = str(df.index[j])
                        break
        
        self._logger.info(f"Fair value gaps identification completed. Found {len(fvg_list)} FVGs")
        
        return df

############### End of Smart Money Concepts ###############



