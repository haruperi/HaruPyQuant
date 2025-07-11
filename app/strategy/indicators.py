import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import MetaTrader5 as mt5
from app.config.constants import ADR_PERIOD, STOP_ADR_RATIO

# Add the parent directory to the path to import app modules
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

# Also add the project root to the path
project_root = os.path.dirname(app_dir)
sys.path.insert(0, project_root)

try:
    from app.util.logger import get_logger
    logger = get_logger(__name__)
    print(f"Successfully imported logger from {app_dir}")
except ImportError as e:
    print(f"Warning: Could not import app logger: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback logging if app logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)




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
    logger.info(f"Calculating EMA for {period} period")
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
    logger.info(f"Calculating SMA for {period} period")
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
    logger.info(f"Calculating RSI for {period} period")
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
    logger.info(f"Calculating ADR for {n} period")
    # Calculate daily ranges
    daily_range = (high - low) / symbol_info.trade_tick_size / 10

    # Calculate ADR
    adr = daily_range.rolling(window=n).mean()

    # Shift the ADR by one period to make today's ADR based on the previous value
    adr = adr.shift(1)

    return adr

def get_adr(df, symbol_info, period=ADR_PERIOD):
    """
    Calculate the Average Daily Range (ADR) and the current daily range percentage.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns ['High', 'Low', 'Close'].
    period (int): The number of days over which to calculate the ADR.

    Returns:
    tuple: current ADR and current daily range percentage
    """
    logger.info(f"Calculating ADR for {period} period")
    # Calculate daily ranges
    df['daily_range'] = (df['High'] - df['Low']) / symbol_info.trade_tick_size / 10

    # Calculate ADR
    df['ADR'] = df['daily_range'].rolling(window=period).mean()

    # Shift the ADR by one period to make today's ADR based on the previous value
    df['ADR'] = df['ADR'].shift(1)

    # Stop Loss Level
    df['SL'] = round(df['ADR'] / STOP_ADR_RATIO)

    # Calculate the current daily range percentage
    current_daily_range = df['daily_range'].iloc[-1]
    current_adr = round(df['ADR'].iloc[-1])
    current_sl = df['SL'].iloc[-1]
    current_daily_range_percentage = round((current_daily_range / current_adr) * 100)

    #return df
    return current_adr, current_daily_range_percentage, current_sl




############### Smart Money Concepts ###############

class SmartMoneyConcepts:
    """
    Smart Money Concepts (SMC) analysis class.
    
    This class provides methods to identify and analyze smart money movements
    in financial markets, including swing highs/lows, swing points, order blocks,
    fair value gaps, and liquidity zones.
    """
    def __init__(self, symbol: str, min_swing_length: int = 3, min_pip_range: int = 3):
        """
        Initialize the SmartMoneyConcepts analyzer.
        """
        self.symbol = symbol
        self.pip_value = mt5.symbol_info(self.symbol).point * 10  # Convert point to pip value
        self.min_swing_length = min_swing_length
        self.min_pip_range = min_pip_range
        logger.info(f"SMC initialized for {symbol}")



    def calculate_swingline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates swing trend lines and adds them as columns to a DataFrame.

        This function processes a DataFrame with 'High' and 'Low' price columns,
        identifies market swing direction, and adds the following columns:
        - swingline: The swing direction (1 for upswing, -1 for downswing).
        - swingvalue: The peak of the current swing (HighestHigh for upswings,
        LowestLow for downswings).
        - highest_low: The highest low point reached during the current upswing.
        - lowest_high: The lowest high point reached during the current downswing.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'High' and 'Low' columns.

        Returns:
            pd.DataFrame: The DataFrame with the four new columns added.
        """
        if 'High' not in df.columns or 'Low' not in df.columns:
            logger.error("Input DataFrame must contain 'High' and 'Low' columns.")
            return df

        logger.info("Initiating swingline calculation")

        if len(df) < 2:
            logger.warning("DataFrame too short for swingline calculation")
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
        swing_value[0] = LowestLow  # In a downswing, swingvalue is LowestLow
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
            
            # Determine the swingvalue based on the current swing direction
            if swing_direction == 1:
                swing_value[i] = HighestLow
            else:
                swing_value[i] = LowestHigh

        # Add the lists as new columns to the DataFrame
        df['swingline'] = swingline
        df['swingvalue'] = swing_value
        df['highest_low'] = highest_low_col
        df['lowest_high'] = lowest_high_col

        logger.info(f"Finalizing swingline calculation. Final swingline: { df['swingline'].iloc[-1]}")
        
        return df
    

    def calculate_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify fractal swing points based on swingline directions.
        
        Args:
            df (pd.DataFrame): DataFrame with swingline column and OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with added swing point indicators
        """
        logger.info("Initiating Fractal Swing Points calculation")
        
        df = df.copy()
        df['swingpoint'] = np.nan

        # Create groups of consecutive swingline values and clean up the data
        group_ids = (df['swingline'] != df['swingline'].shift()).cumsum()
        groups = df.groupby(group_ids)

        for group_id, group_data in groups:
            # Skip groups where all swingvalue values are the same (flat/ranging market)
            # OR if the group is less than min_swing_length candles (default 3) and mix - min swing value is less than minimum pip range ( default 3 pips)

            range_swing_value = (group_data['swingvalue'].max() - group_data['swingvalue'].min()) / self.pip_value # get the range of the swing value in pips

            if group_data['swingvalue'].nunique() == 1 or (len(group_data) < self.min_swing_length and range_swing_value < self.min_pip_range):
                # Get the previous group's swingline value
                if group_id > 1:
                    prev_group_id = group_id - 1
                    prev_group = df[group_ids == prev_group_id]

                    # Handle case where swingline is flat but lower/higher than the last same direction swingline, then keep it dont remove it 
                    # Usually cased by news, rapid price movement or midnight gaps
                    same_swing_prev_group_id = group_id - 2
                    same_swing_prev_group = df[group_ids == same_swing_prev_group_id]
                    if group_data['swingline'].iloc[-1] == 1 and same_swing_prev_group['swingline'].iloc[-1] == 1 and group_data['swingvalue'].iloc[-1] > same_swing_prev_group['swingvalue'].iloc[-1]:
                        continue
                    if group_data['swingline'].iloc[-1] == -1 and same_swing_prev_group['swingline'].iloc[-1] == -1 and group_data['swingvalue'].iloc[-1] < same_swing_prev_group['swingvalue'].iloc[-1]:
                        continue

                    if len(prev_group) > 0:
                        prev_swingline_value = prev_group['swingline'].iloc[0]
                        # Update the current group's swingline values to match the previous group
                        df.loc[group_data.index, 'swingline'] = prev_swingline_value

        # Create groups of consecutive swingline values of clean data
        group_ids = (df['swingline'] != df['swingline'].shift()).cumsum()
        groups = df.groupby(group_ids)

        for group_id, group_data in groups:
            # Get the first swingline value in the group
            sig_value = group_data['swingline'].iloc[0]

            if sig_value == -1:
                # Find index of the minimum low in this group
                min_low_idx = group_data['Low'].idxmin()
                df.at[min_low_idx, 'swingpoint'] = -1
                
            elif sig_value == 1:
                # Find index of the maximum high in this group
                max_high_idx = group_data['High'].idxmax()
                df.at[max_high_idx, 'swingpoint'] = 1

        logger.info(f"Finalizing swing points calculation. Found {df['swingpoint'].notna().sum()} swing points")

        return df
    
    
    def calculate_support_resistance_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Logic:
        - Goal :- To Identify Support/Resistance levels using swing points (swing high point is the high when swingpoint = 1 or low when swingpoint = -1).
        - The first swing high point (swingpoint = 1) is the first resistance level.
        - The last swing low point (swingpoint = -1) is the first support level.
        - Each successive swing point is a potential resistance or support level but only extreme swing points levels are considered.
          which means internal swing points (i.e swing points between/ inside the current resistance/support levels) are ignored.
        - Once these levels are identified, they are marked as resistance or support levels. 
        - These levels extend forward in time until a breakout occurs
        - When current bar's close price is greater than the resistance level, the resistance level is broken.
        - Scan from the begining of the resistance to the current point of breakout and update support level to the lowest point in that range. 
        - Then set resistance to none until a new resistance level is identified by a new swing high point.
        - When current bar's close price is less than the support level, the support level is broken.
        - Scan from the begining of the support to the current point of breakout and update resistance level to the highest point in that range.
        - Then set support to none until a new support level is identified by a new swing low point.
        - ENHANCEMENT: Handle cases where the breakout candle itself might be the next swing point (e.g., news-driven moves).
        
        Args:
            df (pd.DataFrame): DataFrame with swingpoint column as well as the OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with Support and Resistance columns added
        """
        logger.info("Initiating support/resistance levels calculation")

        # --- Initialize variables ---
        df = df.copy()
        resistance = None
        support = None
        resistance_start_index = None
        support_start_index = None

        # --- Create new columns in the DataFrame for support and resistance and initialize with NaN
        df['Support'] = np.nan
        df['Resistance'] = np.nan
        
        # --- Iterate through each row of the DataFrame ---
        for i, row in df.iterrows():
            current_close = row['Close']
            current_high = row['High']
            current_low = row['Low']
            swingpoint = row['swingpoint']

            # --- RESISTANCE DETECTION ---
            # If we don't have a current resistance level and we find a swing high, we establish a new resistance level.
            if resistance is None and swingpoint == 1:
                resistance = current_high
                resistance_start_index = i

            # --- SUPPORT DETECTION ---
            # If we don't have a current support level and we find a swing low, we establish a new support level.
            if support is None and swingpoint == -1:
                support = current_low
                support_start_index = i

            # --- BREAKOUT ABOVE RESISTANCE ---
            # If a resistance level exists and the current closing price breaks above it.
            if resistance is not None and current_close > resistance:
                # The old resistance is broken. We now look for a new support level.
                # The new support is the lowest low since the previous resistance was formed.
                new_support = df.loc[resistance_start_index:i, 'Low'].min()
                
                support = new_support
                support_start_index = i
                
                # Reset resistance as it has been broken
                resistance = None
                resistance_start_index = None
                
                # ENHANCEMENT: Check if the current candle itself is a swing high (news-driven breakout)
                # If so, immediately establish it as the new resistance level
                if swingpoint == 1:
                    resistance = current_high
                    resistance_start_index = i

            # --- BREAKOUT BELOW SUPPORT ---
            # If a support level exists and the current closing price breaks below it.
            if support is not None and current_close < support:
                # The old support is broken. We now look for a new resistance level.
                # The new resistance is the highest high since the previous support was formed.
                new_resistance = df.loc[support_start_index:i, 'High'].max()

                resistance = new_resistance
                resistance_start_index = i
                
                # Reset support as it has been broken
                support = None
                support_start_index = None
                
                # ENHANCEMENT: Check if the current candle itself is a swing low (news-driven breakout)
                # If so, immediately establish it as the new support level
                if swingpoint == -1:
                    support = current_low
                    support_start_index = i

            # --- Extend current levels to the current bar ---
            # If a resistance level is currently active, we record it for the current index.
            if resistance is not None:
                df.loc[i, 'Resistance'] = resistance

            # If a support level is currently active, we record it for the current index.
            if support is not None:
                df.loc[i, 'Support'] = support
        

        logger.info(f"Finalizing support/resistance levels calculation. Current resistance levels: {df['Resistance'].iloc[-1] if not pd.isna(df['Resistance'].iloc[-1]) else 'None'}, Current support levels: {df['Support'].iloc[-1] if not pd.isna(df['Support'].iloc[-1]) else 'None'}")
        
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
        logger.info("Identifying fair value gaps")
        
        df = df.copy()
        
        # Initialize FVG columns
        df['fvg_type'] = np.nan
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan
        df['fvg_size'] = np.nan
        df['fvg_mitigated_index'] = None
        
        if len(df) < 3:
            logger.warning("DataFrame too short for FVG calculation")
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
                        df.at[real_idx, 'fvg_mitigated_index'] = str(df.index[j])
                        break
                else:  # Bearish FVG
                    if candle['High'] >= fvg['high']:
                        df.at[real_idx, 'fvg_mitigated_index'] = str(df.index[j])
                        break
        
        logger.info(f"Fair value gaps identification completed. Found {len(fvg_list)} FVGs")
        
        return df
    

    def calculate_order_blocks(self, df: pd.DataFrame, close_mitigation: bool = True) -> pd.DataFrame:
        # TODO: Add close_mitigation logic
        # TODO: Fix some issues with the logic
        """
        OB - Order Blocks
        For Bullish OB: block spans the entire contiguous Support region, bottom = Support, top = min(swingvalue) of swingline == -1 group that ends at the start of the Support region
        For Bearish OB: block spans the entire contiguous Resistance region, top = Resistance, bottom = max(swingvalue) of swingline == 1 group that ends at the start of the Resistance region
        """
        logger.info("Calculating order blocks (support/resistance region logic)")
        df = df.copy()
        df['Bullish_Order_Block_Top'] = np.nan
        df['Bullish_Order_Block_Bottom'] = np.nan
        df['Bullish_Order_Block_Mitigated'] = 0
        df['Bearish_Order_Block_Top'] = np.nan
        df['Bearish_Order_Block_Bottom'] = np.nan
        df['Bearish_Order_Block_Mitigated'] = 0

        swingline_groups = (df['swingline'] != df['swingline'].shift()).cumsum()
        # Bullish Order Blocks (Support regions)
        support_mask = df['Support'].notna()
        support_regions = []
        start = None
        for i, val in enumerate(support_mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                support_regions.append((start, i-1))
                start = None
        if start is not None:
            support_regions.append((start, len(df)-1))
        for region_start, region_end in support_regions:
            # Find the swingline == -1 group that ends at region_start
            group_id = swingline_groups.iloc[region_start]
            group_indices = df.index[(swingline_groups == group_id) & (df['swingline'] == -1)]
            min_swingvalue = df.loc[group_indices, 'swingvalue'].min() if len(group_indices) > 0 else np.nan
            for j in range(region_start, region_end+1):
                df.at[df.index[j], 'Bullish_Order_Block_Bottom'] = df.at[df.index[j], 'Support']
                df.at[df.index[j], 'Bullish_Order_Block_Top'] = min_swingvalue
        # Bearish Order Blocks (Resistance regions)
        resistance_mask = df['Resistance'].notna()
        resistance_regions = []
        start = None
        for i, val in enumerate(resistance_mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                resistance_regions.append((start, i-1))
                start = None
        if start is not None:
            resistance_regions.append((start, len(df)-1))
        for region_start, region_end in resistance_regions:
            # Find the swingline == 1 group that ends at region_start
            group_id = swingline_groups.iloc[region_start]
            group_indices = df.index[(swingline_groups == group_id) & (df['swingline'] == 1)]
            max_swingvalue = df.loc[group_indices, 'swingvalue'].max() if len(group_indices) > 0 else np.nan
            for j in range(region_start, region_end+1):
                df.at[df.index[j], 'Bearish_Order_Block_Top'] = df.at[df.index[j], 'Resistance']
                df.at[df.index[j], 'Bearish_Order_Block_Bottom'] = max_swingvalue
        # Mitigation logic: price closes inside the order block
        for i in range(len(df)):
            close = df.iloc[i]['Close']
            # Bullish mitigation
            top = df.iloc[i]['Bullish_Order_Block_Top']
            bottom = df.iloc[i]['Bullish_Order_Block_Bottom']
            if pd.notna(top) and pd.notna(bottom):
                if close <= top and close >= bottom:
                    df.at[df.index[i], 'Bullish_Order_Block_Mitigated'] = 1
            # Bearish mitigation
            top = df.iloc[i]['Bearish_Order_Block_Top']
            bottom = df.iloc[i]['Bearish_Order_Block_Bottom']
            if pd.notna(top) and pd.notna(bottom):
                if close >= bottom and close <= top:
                    df.at[df.index[i], 'Bearish_Order_Block_Mitigated'] = 1
        logger.info("Order blocks calculation (support/resistance region logic) completed.")
        return df
    

    def break_of_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BOS - Break of Structure
        This method detects breaks of structure when support or resistance levels are broken.
        
        Args:
            df (pd.DataFrame): DataFrame with Support and Resistance columns
            
        Returns:
            pd.DataFrame: DataFrame with added BOS column
            BOS = 1 if bullish break of structure (resistance broken), -1 if bearish break of structure (support broken)
        """
        logger.info("Calculating break of structure")
        
        df = df.copy()
        
        # Initialize BOS column
        df['BOS'] = np.nan
        
        if len(df) < 2:
            logger.warning("DataFrame too short for BOS calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['Resistance', 'Support']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Process each row to identify breaks of structure
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Check for bullish break of structure (Resistance disappears)
            if pd.isna(current['Resistance']) and pd.notna(previous['Resistance']) or (pd.notna(current['Resistance']) and pd.notna(previous['Resistance']) and current['Resistance'] > previous['Resistance']):
                df.at[df.index[i], 'BOS'] = 1  # Bullish break of structure
                
            # Check for bearish break of structure (Support disappears)
            elif pd.isna(current['Support']) and pd.notna(previous['Support']) or (pd.notna(current['Support']) and pd.notna(previous['Support']) and current['Support'] < previous['Support']):
                df.at[df.index[i], 'BOS'] = -1  # Bearish break of structure
        
        bos_count = len(df[df['BOS'].notna()])
        bullish_bos = len(df[df['BOS'] == 1])
        bearish_bos = len(df[df['BOS'] == -1])
        
        logger.info(f"Break of structure calculation completed. Found {bos_count} BOS events ({bullish_bos} bullish, {bearish_bos} bearish)")
        
        return df
    

    def change_of_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CHoCH - Change of Character
        Improved logic: Track the last broken resistance (when trend_bias == -1) and last broken support (when trend_bias == 1).
        When a new swingvalue is formed, compare it to the last broken level:
        - If trend_bias == -1 and swingvalue > last_broken_resistance, set CHoCH = 1 and update trend_bias.
        - If trend_bias == 1 and swingvalue < last_broken_support, set CHoCH = -1 and update trend_bias.
        Also, treat a change in resistance (current > previous, both not null, in downtrend) or support (current < previous, both not null, in uptrend) as a break event.
        Forward-fill trend_bias from each CHoCH event onward.
        Args:
            df (pd.DataFrame): DataFrame with BOS, swingvalue, Resistance, Support column
        Returns:
            pd.DataFrame: DataFrame with added CHoCH and trend_bias columns
        """
        logger.info("Initializing CHoCH and trend bias columns (improved logic)")
        df = df.copy()
        df['CHoCH'] = np.nan
        df['trend_bias'] = np.nan

        if 'BOS' not in df.columns:
            logger.error("Required column 'BOS' not found in DataFrame")
            raise ValueError("Required column 'BOS' not found in DataFrame")
        # Get BOS events only
        bos_events = df[df['BOS'].notna()]
        bos_indices = bos_events.index.tolist()
        bos_values = bos_events['BOS'].tolist()

        if len(bos_events) == 0:
            logger.warning("No BOS events found in DataFrame")
            return df

        # Set initial CHoCH and trend bias at the first BOS event
        df.at[bos_indices[0], 'CHoCH'] = bos_values[0]
        df.at[bos_indices[0], 'trend_bias'] = bos_values[0]
        # Forward fill trend_bias from the first BOS event onward
        df.loc[bos_indices[0]:, 'trend_bias'] = bos_values[0]

        last_trend_bias = bos_values[0]
        start_pos = df.index.get_loc(bos_indices[0])
        last_broken_resistance = None
        last_broken_support = None

        for i in range(start_pos + 1, len(df)):
            current = df.iloc[i]
            idx = df.index[i]
            prev = df.iloc[i-1]
            # Only check for CHoCH if trend_bias is set
            if pd.isna(df.at[idx, 'trend_bias']):
                df.at[idx, 'trend_bias'] = last_trend_bias
            trend_bias = df.at[idx, 'trend_bias']
            # Track last broken resistance/support
            # Resistance disappears (null) or jumps up (gap/news) in downtrend
            if (
                prev['Resistance'] is not None and not pd.isna(prev['Resistance']) and
                ((current['Resistance'] is None or pd.isna(current['Resistance'])) or
                 (current['Resistance'] is not None and not pd.isna(current['Resistance']) and current['Resistance'] > prev['Resistance'])) and
                trend_bias == -1
            ):
                last_broken_resistance = prev['Resistance']
            # Support disappears (null) or jumps down (gap/news) in uptrend
            if (
                prev['Support'] is not None and not pd.isna(prev['Support']) and
                ((current['Support'] is None or pd.isna(current['Support'])) or
                 (current['Support'] is not None and not pd.isna(current['Support']) and current['Support'] < prev['Support'])) and
                trend_bias == 1
            ):
                last_broken_support = prev['Support']
            # Bullish CHoCH: swingvalue breaks above last broken resistance while in downtrend
            if (
                trend_bias == -1 and
                last_broken_resistance is not None and
                current['swingvalue'] > last_broken_resistance
            ):
                df.at[idx, 'CHoCH'] = 1
                df.at[idx, 'trend_bias'] = 1
                df.loc[idx:, 'trend_bias'] = 1
                last_trend_bias = 1
                last_broken_resistance = None  # Reset after CHoCH
            # Bearish CHoCH: swingvalue breaks below last broken support while in uptrend
            elif (
                trend_bias == 1 and
                last_broken_support is not None and
                current['swingvalue'] < last_broken_support
            ):
                df.at[idx, 'CHoCH'] = -1
                df.at[idx, 'trend_bias'] = -1
                df.loc[idx:, 'trend_bias'] = -1
                last_trend_bias = -1
                last_broken_support = None  # Reset after CHoCH

        choch_count = len(df[df['CHoCH'].notna()])
        bullish_choch = len(df[df['CHoCH'] == 1])
        bearish_choch = len(df[df['CHoCH'] == -1])
        logger.info(f"Change of character calculation completed. Found {choch_count} CHoCH events ({bullish_choch} bullish, {bearish_choch} bearish)")
        return df
    

    

    def retracements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'CurrentRetracement%' and 'DeepestRetracement%' columns.
        Only calculates when trend_bias != swingline (i.e., moving against the trend).
        - For uptrend (trend_bias==1, swingline==-1): retracement from last support to current close.
        - For downtrend (trend_bias==-1, swingline==1): retracement from last resistance to current close.
        Args:
            df (pd.DataFrame): DataFrame with trend_bias, swingline, Support, Resistance, and Close columns
        Returns:
            pd.DataFrame: DataFrame with added retracement columns (values in actual percent, e.g., 67 for 67%)
        """
        logger.info("Calculating retracement statistics")
        df = df.copy()
        df['CurrentRetracement'] = np.nan
        df['DeepestRetracement'] = np.nan
        required_cols = ['trend_bias', 'swingline', 'Support', 'Resistance', 'Close', 'High', 'Low']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        # Track deepest retracement during each counter-trend move, reset when trend resumes or S/R changes
        deepest = 0.0
        last_trend_bias = None
        last_supp = None
        last_res = None
        for i in range(len(df)):
            tb = df['trend_bias'].iloc[i]
            sw = df['swingline'].iloc[i]
            close = df['Close'].iloc[i]
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            support = df['Support'].iloc[i]
            resistance = df['Resistance'].iloc[i]
            # Only calculate if both S/R exist, are not equal, and counter-trend
            if pd.isna(tb) or pd.isna(sw) or pd.isna(support) or pd.isna(resistance) or support == resistance:
                df.at[df.index[i], 'CurrentRetracement'] = np.nan
                df.at[df.index[i], 'DeepestRetracement'] = np.nan
                # Reset segment if S/R missing
                deepest = 0.0
                last_trend_bias = tb
                last_supp = support
                last_res = resistance
                continue
            # Reset deepest if trend resumes or S/R changes
            if tb == sw or tb != last_trend_bias or support != last_supp or resistance != last_res:
                deepest = 0.0
            retr = np.nan
            # Calculate retracement percent based on trend_bias
            if tb != sw:
                if tb == 1:
                    # Uptrend: Support=100%, Resistance=0%
                    base = resistance
                    top = support
                elif tb == -1:
                    # Downtrend: Resistance=100%, Support=0%
                    base = support
                    top = resistance
                else:
                    base = np.nan
                    top = np.nan
                if not np.isnan(base) and not np.isnan(top) and top != base:
                    retr = 100 * (close - base) / (top - base)
                    retr_high = 100 * (high - base) / (top - base)
                    retr_low = 100 * (low - base) / (top - base)
                    # Deepest is the largest magnitude (absolute value) seen so far in the segment
                    deepest = max(deepest, abs(retr), abs(retr_high), abs(retr_low))
                    df.at[df.index[i], 'CurrentRetracement'] = retr
                    df.at[df.index[i], 'DeepestRetracement'] = deepest
                else:
                    df.at[df.index[i], 'CurrentRetracement'] = np.nan
                    df.at[df.index[i], 'DeepestRetracement'] = np.nan
            else:
                df.at[df.index[i], 'CurrentRetracement'] = np.nan
                df.at[df.index[i], 'DeepestRetracement'] = np.nan
            last_trend_bias = tb
            last_supp = support
            last_res = resistance
        # After calculation, round to 2 decimal places
        df['CurrentRetracement'] = df['CurrentRetracement'].round(2)
        df['DeepestRetracement'] = df['DeepestRetracement'].round(2)
        return df
    

    def get_fib_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading fibonacci signals based on Close price crossing over swingvalue and retracement conditions (Fibonacci retracement > 50%).
        
        Signal Logic:
        - Signal = 1 (Buy): trend_bias = 1 and swingline = -1 and prev_close < swingvalue and current_close > swingvalue and CurrentRetracement > 50
        - Signal = -1 (Sell): trend_bias = -1 and swingline = 1 and prev_close > swingvalue and current_close < swingvalue and CurrentRetracement > 50
        
        Args:
            df (pd.DataFrame): DataFrame with trend_bias, swingline, swingvalue, Close, and CurrentRetracement columns
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        logger.info("Generating trading signals based on Close price crossing over swingvalue and retracement")
        
        df = df.copy()
        df['fib_signal'] = np.nan
        
        # Check if required columns exist
        required_cols = ['trend_bias', 'swingline', 'swingvalue', 'Close', 'CurrentRetracement']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        if len(df) < 2:
            logger.warning("DataFrame too short for signal generation")
            return df
        
        # Process each row to identify signals
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Check for bullish signal (Close crosses above swingvalue)
            if (current['trend_bias'] == 1 and 
                current['swingline'] == -1 and
                previous['Close'] < previous['swingvalue'] and
                current['Close'] > current['swingvalue'] and
                pd.notna(current['CurrentRetracement']) and
                current['CurrentRetracement'] > 50):
                
                df.at[df.index[i], 'fib_signal'] = 1
                logger.debug(f"Bullish signal generated at index {df.index[i]}: trend_bias={current['trend_bias']}, swingline={current['swingline']}, Close crossover from {previous['Close']:.5f} < {previous['swingvalue']:.5f} to {current['Close']:.5f} > {current['swingvalue']:.5f}, retracement={current['CurrentRetracement']}")
            
            # Check for bearish signal (Close crosses below swingvalue)
            elif (current['trend_bias'] == -1 and 
                  current['swingline'] == 1 and
                  previous['Close'] > previous['swingvalue'] and
                  current['Close'] < current['swingvalue'] and
                  pd.notna(current['CurrentRetracement']) and
                  current['CurrentRetracement'] > 50):
                
                df.at[df.index[i], 'fib_signal'] = -1
                logger.debug(f"Bearish signal generated at index {df.index[i]}: trend_bias={current['trend_bias']}, swingline={current['swingline']}, Close crossover from {previous['Close']:.5f} > {previous['swingvalue']:.5f} to {current['Close']:.5f} < {current['swingvalue']:.5f}, retracement={current['CurrentRetracement']}")
        
        bullish_signals = len(df[df['fib_signal'] == 1])
        bearish_signals = len(df[df['fib_signal'] == -1])
        total_signals = bullish_signals + bearish_signals
        
        logger.info(f"Fibonacci signal generation completed. Found {total_signals} signals ({bullish_signals} bullish, {bearish_signals} bearish)")
        
        return df

    def get_bos_retest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates BOS retest signals based on trend_bias, swingline, swingvalue, BOS, and Close.
        Adds a column 'retest_signal' to df.
        
        Signal Logic:
        - Signal = 1 (Buy): trend_bias = 1 and swingline = -1 and prev_close < swingvalue and current_close > swingvalue and retested the last BOS (1), i.e. the lowest point in this current swingline group is below the last BOS (1)
        - Signal = -1 (Sell): trend_bias = -1 and swingline = 1 and prev_close > swingvalue and current_close < swingvalue and retested the last BOS (-1), i.e. the highest point in this current swingline group is above the last BOS (-1)
        """

        logger.info("Initiating BOS retest signal generation")
        
        df = df.copy()
        df['retest_signal'] = np.nan
        required_cols = ['trend_bias', 'swingline', 'swingvalue', 'BOS', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        last_bos_1_idx = None
        last_bos_1_val = None
        last_bos_neg1_idx = None
        last_bos_neg1_val = None
        # Track swingline groups
        swingline_groups = (df['swingline'] != df['swingline'].shift()).cumsum()
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            group_id = swingline_groups.iloc[i]
            group_indices = df.index[swingline_groups == group_id]
            group_lows = df.loc[group_indices, 'Low'] if 'Low' in df.columns else None
            group_highs = df.loc[group_indices, 'High'] if 'High' in df.columns else None
            # Track last BOS(1) and BOS(-1)
            if not np.isnan(current['BOS']):
                if current['BOS'] == 1:
                    last_bos_1_idx = i
                    last_bos_1_val = current['swingvalue']
                elif current['BOS'] == -1:
                    last_bos_neg1_idx = i
                    last_bos_neg1_val = current['swingvalue']
            # Buy signal
            if (
                current['trend_bias'] == 1 and
                current['swingline'] == -1 and
                previous['Close'] < previous['swingvalue'] and
                current['Close'] > current['swingvalue'] and
                last_bos_1_val is not None and
                group_lows is not None and
                group_lows.min() < last_bos_1_val
            ):
                df.at[df.index[i], 'retest_signal'] = 1
            # Sell signal
            elif (
                current['trend_bias'] == -1 and
                current['swingline'] == 1 and
                previous['Close'] > previous['swingvalue'] and
                current['Close'] < current['swingvalue'] and
                last_bos_neg1_val is not None and
                group_highs is not None and
                group_highs.max() > last_bos_neg1_val
            ):
                df.at[df.index[i], 'retest_signal'] = -1
        logger.info("BOS retest signal generation completed")
        return df
    

    def run_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the Smart Money Concepts indicators on the DataFrame.
        """
        df = self.identify_fair_value_gaps(df, join_consecutive=True)
        df = self.calculate_swingline(df)
        df = self.calculate_swing_points(df)
        df = self.calculate_support_resistance_levels(df)
        df = self.calculate_order_blocks(df, close_mitigation=True)
        df = self.break_of_structure(df)
        df = self.change_of_character(df)
        df = self.retracements(df)
        df = self.get_fib_signal(df)
        df = self.get_bos_retest_signals(df)
        # Save results
        df.to_csv(f"smc_analysis.csv")
        logger.info(f"Data saved to smc_analysis.csv")
        logger.info("Smart Money Concepts indicators calculation completed")
        return df


    



############### End of Smart Money Concepts ###############



