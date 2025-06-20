import pandas as pd
import numpy as np

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



############### Swingline ###############
def _update_swing_highs(current_high, current_low, highest_high, highest_low):
    """Update highest high and highest low values."""
    if current_high > highest_high:
        highest_high = current_high
    if current_low > highest_low:
        highest_low = current_low
    return highest_high, highest_low

def _update_swing_lows(current_high, current_low, lowest_high, lowest_low):
    """Update lowest high and lowest low values."""
    if current_low < lowest_low:
        lowest_low = current_low
    if current_high < lowest_high:
        lowest_high = current_high
    return lowest_high, lowest_low

def _check_swing_change(current_high, current_low, current_open, current_close, 
                       prev_high, prev_low, highest_low, lowest_high, swingline):
    """Check and determine if swing direction should change."""
    if swingline == 1:
        if current_high < highest_low and current_close < current_open and current_close < prev_low:
            return -1, current_low, current_high
    elif swingline == -1:
        if current_low > lowest_high and current_close > current_open and current_close > prev_high:
            return 1, current_high, current_low
    return swingline, None, None

def calculate_swingline(df):
    """
    Calculates the swingline direction based on price action.
    Returns DataFrame with added 'swingline', 'highest_low', 'lowest_high', and 'swing_value' columns.
    """
    df = df.copy()
    df['swingline'] = -1
    df['highest_low'] = np.nan
    df['lowest_high'] = np.nan

    highest_high = df['High'].iloc[0]
    lowest_low = df['Low'].iloc[0]
    lowest_high = df['High'].iloc[0]
    highest_low = df['Low'].iloc[0]
    swingline = -1

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        if swingline == 1:
            highest_high, highest_low = _update_swing_highs(
                current['High'], current['Low'], highest_high, highest_low
            )
        else:
            lowest_high, lowest_low = _update_swing_lows(
                current['High'], current['Low'], lowest_high, lowest_low
            )

        new_swingline, new_high, new_low = _check_swing_change(
            current['High'], current['Low'], current['Open'], current['Close'],
            prev['High'], prev['Low'], highest_low, lowest_high, swingline
        )

        if new_swingline != swingline:
            swingline = new_swingline
            if swingline == 1:
                highest_high, highest_low = new_high, new_low
            else:
                lowest_high, lowest_low = new_high, new_low

        df.loc[df.index[i], 'highest_low'] = highest_low
        df.loc[df.index[i], 'lowest_high'] = lowest_high
        df.loc[df.index[i], 'swingline'] = swingline
        df.loc[df.index[i], 'swing_value'] = highest_low if swingline == 1 else lowest_high

    df = df.drop(['highest_low', 'lowest_high', 'swing_value'], axis=1)

    return df

############### End of Swingline ###############

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



