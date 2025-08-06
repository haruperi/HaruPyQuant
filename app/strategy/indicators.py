import pandas as pd
import numpy as np
from scipy.stats import gmean
from typing import Dict
from app.config.setup import *

logger = get_logger(__name__)

######################### Standard indicators #########################

def ma(df: pd.DataFrame, period: int = FAST_MA_PERIOD, ma_type: str = "EMA", column: str = 'Close') -> pd.DataFrame:
    """
    Calculates Moving Average (MA) and adds it to the DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing price data.
        period (int): The MA period (window/span).
        ma_type (str): The type of moving average. Options: "EMA" (default), "SMA", "WMA".
        column (str): The column name to use for MA calculation. Default is 'Close'.

    Returns:
        pd.DataFrame: The DataFrame with an added MA column containing the calculated values.
    """
    logger.info(f"Calculating {ma_type} for {period} period using {column} column")
    
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return df
    
    data = df[column]
    
    if ma_type.upper() == "EMA":
        ma_values = data.ewm(span=period, adjust=False).mean()
        column_name = f'ema_{period}'
    elif ma_type.upper() == "SMA":
        ma_values = data.rolling(window=period).mean()
        column_name = f'sma_{period}'
    elif ma_type.upper() == "WMA":
        # Calculate Weighted Moving Average
        weights = np.arange(1, period + 1)
        ma_values = data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        column_name = f'wma_{period}'
    else:
        logger.error(f"Unsupported MA type: {ma_type}. Supported types: EMA, SMA, WMA")
        return df
    
    # Add MA column to the DataFrame
    df_result = df.copy()
    # Convert to pandas Series and shift the MA values to the previous value (align with previous bar)
    ma_values = pd.Series(ma_values).shift(1)
    df_result[column_name] = ma_values
    logger.info(f"Added {column_name} column to the DataFrame. Last value: {ma_values.iloc[-1]}")
    
    return df_result

def rsi(df: pd.DataFrame, period: int = RSI_PERIOD, column: str = 'Close') -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) and adds it to the DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing price data.
        period (int): The RSI period. Default is 14.
        column (str): The column name to use for RSI calculation. Default is 'Close'.

    Returns:
        pd.DataFrame: The DataFrame with an added 'rsi' column containing the calculated RSI values.
    """
    logger.info(f"Calculating RSI for {period} period using {column} column")
    
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return df
    
    data = df[column]
    delta = data.diff()

    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Add RSI column to the DataFrame
    df_result = df.copy()
    rsi_values = pd.Series(rsi).shift(1)
    df_result['rsi'] = rsi_values
    logger.info(f"Added 'rsi' column to the DataFrame. Last value: {rsi_values.iloc[-1]}")
    
    return df_result

def williams_percent(df: pd.DataFrame, period: int = WILLIAMS_R_PERIOD) -> pd.DataFrame:
    """
    Calculates the Williams %R indicator and adds it to the DataFrame.

    Williams %R is a momentum indicator that measures overbought/oversold levels.
    It ranges from 0 to -100, where:
    - 0 to -20: Overbought (potential sell signal)
    - -80 to -100: Oversold (potential buy signal)

    Args:
        df (pd.DataFrame): A pandas DataFrame containing OHLC data.
        period (int): The lookback period for calculating the indicator. Default is 14.

    Returns:
        pd.DataFrame: The DataFrame with an added 'williams_r' column containing the calculated Williams %R values.
    """
    logger.info(f"Calculating Williams %R for {period} period")
    
    # Check if required columns exist
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return df
    
    # Calculate Williams %R
    # Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    
    # Calculate Williams %R
    williams_r = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
    
    # Add Williams %R column to the DataFrame
    df_result = df.copy()
    williams_r_values = pd.Series(williams_r).shift(1)
    df_result['williams_r'] = williams_r_values
    logger.info(f"Added 'williams_r' column to the DataFrame. Last value: {williams_r_values.iloc[-1]}")
    
    return df_result

def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR) indicator and adds it to the DataFrame.

    ATR is a volatility indicator that measures market volatility by decomposing
    the entire range of an asset price for that period. It helps in setting
    stop-loss levels and determining position sizing.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing OHLC data.
        period (int): The period for calculating the ATR. Default is 14.

    Returns:
        pd.DataFrame: The DataFrame with an added 'atr' column containing the calculated ATR values.
    """
    logger.info(f"Calculating ATR for {period} period")
    
    # Check if required columns exist
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return df
    
    # Calculate True Range (TR)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Compute True Range (TR)
    previous_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute ATR using Simple Moving Average (SMA) - this matches MT5
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    # Add ATR column to the DataFrame
    df_result = df.copy()
    atr_values = pd.Series(atr).shift(1)
    df_result['atr'] = atr_values
    logger.info(f"Added 'atr' column to the DataFrame. Last value: {atr_values.iloc[-1]}")
    
    return df_result

######################### Custom indicators #########################

def adr(df, symbol_info, period=ADR_PERIOD):
    """
    Calculate the Average Daily Range (ADR) and the current daily range percentage.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns ['High', 'Low', 'Close'].
    period (int): The number of days over which to calculate the ADR.

    Returns:
    tuple: current ADR and current daily range percentage
    """
    logger.info(f"Calculating ADR for {period} period")
    
    # Check if symbol_info is valid
    if symbol_info is None:
        logger.error("symbol_info is None, cannot calculate ADR")
        return None
    
    # Check if required attributes exist
    if not hasattr(symbol_info, 'trade_tick_size') or symbol_info.trade_tick_size is None:
        logger.error(f"symbol_info missing trade_tick_size attribute: {symbol_info}")
        return None
    
    # Check if DataFrame has required columns
    required_columns = ['High', 'Low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"DataFrame missing required columns for ADR calculation: {missing_columns}")
        return None
    
    try:
        # Calculate daily ranges
        df['daily_range'] = (df['High'] - df['Low']) / symbol_info.trade_tick_size / 10

        # Calculate ADR
        df['ADR'] = df['daily_range'].rolling(window=period).mean()

        # Shift the ADR by one period to make today's ADR based on the previous value
        df['ADR'] = df['ADR'].shift(1)

        # Stop Loss Level
        df['SL'] = round(df['ADR'] / STOP_ADR_RATIO)

        logger.info(f"Added 'daily_range', 'ADR', 'SL' columns to the DataFrame. Last value: {df['daily_range'].iloc[-1]}, {df['ADR'].iloc[-1]}, {df['SL'].iloc[-1]}")

        return df
    except Exception as e:
        logger.error(f"Error calculating ADR: {e}")
        return None

def currency_index(base_currency: str, ohlc_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculates a synthetic currency index OHLC using the geometric mean of its constituent pairs.

    Args:
        base_currency (str): The currency for which to calculate the index (e.g., 'USD', 'EUR').
        ohlc_data (dict[str, pd.DataFrame]): A dictionary where keys are currency pair names
                                             (e.g., 'EURUSD') and values are pandas DataFrames.
                                             Each DataFrame must have 'Open', 'High', 'Low', 'Close'
                                             columns and a DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame with the calculated OHLC for the currency index.
                      Returns an empty DataFrame if the base currency is not supported or
                      if required pair data is missing.
    """
    # --- CONFIGURATION ---
    # Defines the currency pairs for each index.
    # 'True' means the pair's price should be inverted (1/price).
    logger.info(f"Calculating currency index for {base_currency}")
    INDEX_CONFIG = {
        'USD': {'EURUSD': True, 'GBPUSD': True, 'AUDUSD': True, 'NZDUSD': True, 'USDCAD': False, 'USDCHF': False, 'USDJPY': False},
        'EUR': {'EURUSD': False, 'EURGBP': False, 'EURAUD': False, 'EURNZD': False, 'EURCAD': False, 'EURCHF': False, 'EURJPY': False},
        'GBP': {'GBPUSD': False, 'GBPAUD': False, 'GBPNZD': False, 'GBPCAD': False, 'GBPCHF': False, 'GBPJPY': False, 'EURGBP': True},
        'JPY': {'USDJPY': True, 'EURJPY': True, 'GBPJPY': True, 'AUDJPY': True, 'NZDJPY': True, 'CADJPY': True, 'CHFJPY': True},
        'AUD': {'AUDUSD': False, 'AUDNZD': False, 'AUDCAD': False, 'AUDCHF': False, 'AUDJPY': False, 'EURAUD': True, 'GBPAUD': True},
        'NZD': {'NZDUSD': False, 'NZDCAD': False, 'NZDCHF': False, 'NZDJPY': False, 'EURNZD': True, 'GBPNZD': True, 'AUDNZD': True},
        'CAD': {'CADCHF': False, 'CADJPY': False, 'USDCAD': True, 'EURCAD': True, 'GBPCAD': True, 'AUDCAD': True, 'NZDCAD': True},
        'CHF': {'CHFJPY': False, 'USDCHF': True, 'EURCHF': True, 'GBPCHF': True, 'AUDCHF': True, 'NZDCHF': True, 'CADCHF': True}
    }

    if base_currency not in INDEX_CONFIG:
        logger.error(f"Error: Currency '{base_currency}' is not supported.")
        return pd.DataFrame()

    pairs_config = INDEX_CONFIG[base_currency]
    
    # --- Data Normalization ---
    # Store the normalized OHLC Series for all required pairs
    normalized_ohlc_series = {
        'Open': [], 'High': [], 'Low': [], 'Close': []
    }

    for pair, invert in pairs_config.items():
        if pair not in ohlc_data:
            logger.error(f"Error: Missing OHLC data for required pair '{pair}'.")
            return pd.DataFrame()

        df_pair = ohlc_data[pair].copy()

        # Ensure required columns exist
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(df_pair.columns):
            logger.error(f"Error: DataFrame for '{pair}' is missing one or more required columns: Open, High, Low, Close.")
            return pd.DataFrame()

        if invert:
            # When inverting, the new high is 1/low and the new low is 1/high.
            # We create temporary columns to avoid modifying the original DataFrame in place.
            df_norm_high = 1 / df_pair['Low']
            df_norm_low = 1 / df_pair['High']
            
            normalized_ohlc_series['Open'].append(1 / df_pair['Open'])
            normalized_ohlc_series['High'].append(df_norm_high)
            normalized_ohlc_series['Low'].append(df_norm_low)
            normalized_ohlc_series['Close'].append(1 / df_pair['Close'])
        else:
            normalized_ohlc_series['Open'].append(df_pair['Open'])
            normalized_ohlc_series['High'].append(df_pair['High'])
            normalized_ohlc_series['Low'].append(df_pair['Low'])
            normalized_ohlc_series['Close'].append(df_pair['Close'])

    # --- Calculation ---
    # Create a new DataFrame to hold the index results
    index_df = pd.DataFrame(index=ohlc_data[list(pairs_config.keys())[0]].index)

    # Calculate the geometric mean for each OHLC component
    for ohlc_type in ['Open', 'High', 'Low', 'Close']:
        # Concatenate all series of the same type (e.g., all 'Open' series)
        ohlc_matrix = pd.concat(normalized_ohlc_series[ohlc_type], axis=1)
        # Calculate the geometric mean row-wise
        # Formula: (x1 * x2 * ... * xn)^(1/n)
        # We use numpy's prod for product and power for the nth root.
        index_df[ohlc_type] = np.power(ohlc_matrix.prod(axis=1), 1.0/ohlc_matrix.shape[1])

    logger.info(f"Completed currency index calculation for {base_currency}")

    return index_df

def currency_index_purple_trading(target_currency: str, ohlc_data_dict: dict) -> pd.DataFrame:
    """
    Calculates a currency index using a geometric mean approach.

    This function simulates a currency index by calculating the geometric mean of its
    exchange rate against a basket of other currencies. The formula is a standard
    method for index calculation, as the specific "Purple Trading formula" is proprietary.

    The function correctly handles both direct and indirect currency pairs.
    For example, when calculating the USD index, EURUSD is used directly,
    but USDJPY is inverted (1/USDJPY) to represent the value of USD in terms of JPY.

    Args:
        target_currency (str): The 3-letter code for the currency to be indexed (e.g., 'USD').
        ohlc_data_dict (dict): A dictionary where keys are currency pair strings
                               (e.g., 'EURUSD') and values are pandas DataFrames
                               with 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        pd.DataFrame: A pandas DataFrame with 'Open', 'High', 'Low', 'Close' columns
                      for the calculated index, or an empty DataFrame if no relevant
                      pairs are found.
    """
    target_currency = target_currency.upper()
    relevant_pairs = {}
    
    # --- Step 1: Identify relevant pairs and determine if they need to be inverted ---
    for pair, data in ohlc_data_dict.items():
        pair = pair.upper()
        if target_currency in pair:
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                print(f"Warning: Skipping {pair} as its data is not a pandas DataFrame.")
                continue

            # Check for required columns
            required_cols = {'Open', 'High', 'Low', 'Close'}
            if not required_cols.issubset(data.columns):
                print(f"Warning: Skipping {pair} as it's missing one or more required columns: {required_cols}.")
                continue

            # Determine if the pair needs to be inverted.
            # If the target currency is the base currency (first 3 letters), we need to invert.
            # e.g., for JPY index, USDJPY -> JPY/USD is 1/USDJPY
            if pair.startswith(target_currency):
                # Invert the OHLC data
                # Note: For High/Low, the inverse relationship is swapped.
                # The inverse of the high price is the new low, and vice versa.
                inverted_data = pd.DataFrame({
                    'Open': 1 / data['Open'],
                    'High': 1 / data['Low'],
                    'Low': 1 / data['High'],
                    'Close': 1 / data['Close']
                })
                relevant_pairs[pair] = inverted_data
            # If the target currency is the quote currency (last 3 letters), use as is.
            # e.g., for USD index, EURUSD is used directly.
            elif pair.endswith(target_currency):
                relevant_pairs[pair] = data.copy()

    if not relevant_pairs:
        print(f"Error: No relevant currency pairs found for '{target_currency}' in the provided data.")
        return pd.DataFrame()

    # --- Step 2: Combine the data and calculate the geometric mean ---
    # Concatenate all relevant 'Close' prices for calculation
    combined_data = pd.concat([df['Close'].rename(pair) for pair, df in relevant_pairs.items()], axis=1)

    # Calculate the geometric mean. The formula is (p1 * p2 * ... * pn)^(1/n)
    # This is equivalent to exp( (log(p1) + log(p2) + ... + log(pn)) / n )
    # Using logs helps with numerical stability.
    num_pairs = len(relevant_pairs)
    index_close = combined_data.apply(lambda x: x.prod()**(1/num_pairs), axis=1)

    # --- Step 3: Calculate OHLC for the index ---
    # This is a simplification. A true index OHLC would require tick-level data.
    # Here, we approximate by taking the geometric mean of the respective OHLC values.
    ohlc_frames = [df[['Open', 'High', 'Low', 'Close']].rename(columns=lambda c: f"{pair}_{c}") for pair, df in relevant_pairs.items()]
    full_ohlc_data = pd.concat(ohlc_frames, axis=1)

    def geometric_mean_row(row, ohlc_type):
        prices = [row[f'{pair}_{ohlc_type}'] for pair in relevant_pairs.keys() if f'{pair}_{ohlc_type}' in row]
        if not prices:
            return None
        # Filter out any non-positive values before calculation
        prices = [p for p in prices if p > 0]
        if not prices:
            return None
        
        product = 1
        for p in prices:
            product *= p
        return product ** (1 / len(prices))

    index_df = pd.DataFrame(index=index_close.index)
    index_df['Open'] = full_ohlc_data.apply(lambda row: geometric_mean_row(row, 'Open'), axis=1)
    index_df['High'] = full_ohlc_data.apply(lambda row: geometric_mean_row(row, 'High'), axis=1)
    index_df['Low'] = full_ohlc_data.apply(lambda row: geometric_mean_row(row, 'Low'), axis=1)
    index_df['Close'] = index_close
    
    # --- Step 4: Normalize the index to a base value (e.g., 100 or 1000) for readability ---
    # This makes it easier to track percentage changes from the start.
    initial_value = index_df['Close'].iloc[0]
    if pd.notna(initial_value) and initial_value > 0:
        index_df = (index_df / initial_value) * 1000  # Start index at 1000

    return index_df

def geometric_mean_index(target_currency, ohlc_data_dict):
    """
    Calculates a currency index using the normalized geometric mean method from a dictionary of OHLC DataFrames.

    This method computes an index by taking the geometric mean of a currency's performance
    against a basket of its peers. It accepts a dictionary where keys are currency symbols
    and values are their corresponding OHLC DataFrames. It returns a single DataFrame 
    with the calculated OHLC values for the index.

    Args:
        target_currency (str): The 3-letter code for the currency to be indexed (e.g., 'USD').
        ohlc_data_dict (dict): A dictionary where keys are currency pair strings (e.g., 'EURUSD')
                               and values are pandas DataFrames with 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        pd.DataFrame: A pandas DataFrame with 'Open', 'High', 'Low', 'Close' columns for the calculated index.
    """
    # Define the currency pairs for the target currency.
    currency_baskets = {
        'USD': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
        'EUR': ['EURUSD', 'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF', 'EURJPY'],
    }

    if target_currency not in currency_baskets:
        raise ValueError(f"Basket for '{target_currency}' is not defined.")
        
    if not ohlc_data_dict:
        return pd.DataFrame()
        
    sample_df = next(iter(ohlc_data_dict.values()))
    index_ohlc_df = pd.DataFrame(index=sample_df.index)
    pairs = currency_baskets[target_currency]

    for ohlc_type in ['Open', 'High', 'Low', 'Close']:
        # Create a single DataFrame for the current OHLC type
        price_data_list = [df[ohlc_type].rename(symbol) for symbol, df in ohlc_data_dict.items() if ohlc_type in df.columns]
        if not price_data_list:
            print(f"Warning: No data found for OHLC type '{ohlc_type}'. Skipping.")
            continue
        price_data = pd.concat(price_data_list, axis=1)

        normalized_prices = pd.DataFrame(index=price_data.index)

        # Normalize each pair in the basket
        for pair in pairs:
            if pair not in price_data.columns:
                print(f"Warning: Data for '{pair}' not found for {ohlc_type}. Skipping.")
                continue

            price_series = price_data[pair]

            if target_currency == pair[3:]:
                price_series = 1 / price_series

            start_price = price_series.iloc[0]
            normalized_prices[pair] = (price_series / start_price) * 100
        
        # Calculate geometric mean for the current OHLC type
        # Drop any rows with NaN values to ensure gmean works correctly
        valid_normalized_prices = normalized_prices.dropna()
        index_values = gmean(valid_normalized_prices, axis=1)
        
        index_ohlc_df[ohlc_type] = pd.Series(index_values, index=valid_normalized_prices.index)

    return index_ohlc_df

def strength_meter_latest_price(ohlc_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Calculates a currency strength meter based on the most recent OHLC data.

    The function first calculates a synthetic index for each of the 8 major
    currencies using the geometric mean of its constituent pairs. It then
    measures strength by calculating the percentage change from the Open to
    the Close of the calculated index for the latest available period.

    Args:
        ohlc_data (Dict[str, pd.DataFrame]):
            A dictionary where keys are currency pair names (e.g., 'EURUSD')
            and values are pandas DataFrames. Each DataFrame must have 'Open',
            'High', 'Low', 'Close' columns and a DatetimeIndex. The calculation
            will be based on the last row of each DataFrame.

    Returns:
        pd.Series: A pandas Series with currency codes as the index and their
                   calculated strength (% change) as values, sorted from
                   strongest to weakest. Returns an empty Series if data is
                   insufficient.
    """
    logger.info(f"Calculating currency strength meter")
    # --- CONFIGURATION ---
    # Defines the currency pairs and inversion logic for each index.
    CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    INDEX_CONFIG = {
        'USD': {'EURUSD': True, 'GBPUSD': True, 'AUDUSD': True, 'NZDUSD': True, 'USDCAD': False, 'USDCHF': False, 'USDJPY': False},
        'EUR': {'EURUSD': False, 'EURGBP': False, 'EURAUD': False, 'EURNZD': False, 'EURCAD': False, 'EURCHF': False, 'EURJPY': False},
        'GBP': {'GBPUSD': False, 'GBPAUD': False, 'GBPNZD': False, 'GBPCAD': False, 'GBPCHF': False, 'GBPJPY': False, 'EURGBP': True},
        'JPY': {'USDJPY': True, 'EURJPY': True, 'GBPJPY': True, 'AUDJPY': True, 'NZDJPY': True, 'CADJPY': True, 'CHFJPY': True},
        'AUD': {'AUDUSD': False, 'AUDNZD': False, 'AUDCAD': False, 'AUDCHF': False, 'AUDJPY': False, 'EURAUD': True, 'GBPAUD': True},
        'NZD': {'NZDUSD': False, 'NZDCAD': False, 'NZDCHF': False, 'NZDJPY': False, 'EURNZD': True, 'GBPNZD': True, 'AUDNZD': True},
        'CAD': {'CADCHF': False, 'CADJPY': False, 'USDCAD': True, 'EURCAD': True, 'GBPCAD': True, 'AUDCAD': True, 'NZDCAD': True},
        'CHF': {'CHFJPY': False, 'USDCHF': True, 'EURCHF': True, 'GBPCHF': True, 'AUDCHF': True, 'NZDCHF': True, 'CADCHF': True}
    }

    strengths = {}

    for currency in CURRENCIES:
        pairs_config = INDEX_CONFIG[currency]
        
        # --- Data Normalization for the latest period ---
        normalized_open = []
        normalized_close = []

        for pair, invert in pairs_config.items():
            if pair not in ohlc_data or ohlc_data[pair].empty:
                logger.warning(f"Warning: Missing or empty data for required pair '{pair}' for {currency} index. Skipping.")
                # Return empty series if any data is missing to ensure accuracy
                return pd.Series(dtype=np.float64)

            # Get the latest row of data
            latest_data = ohlc_data[pair].iloc[-2]
            logger.info(f"Latest data for {pair}: {latest_data}")
            o, c = latest_data['Open'], latest_data['Close']

            if invert:
                normalized_open.append(1 / o)
                normalized_close.append(1 / c)
            else:
                normalized_open.append(o)
                normalized_close.append(c)
        
        # --- Index and Strength Calculation ---
        # Geometric mean for index open and close
        index_open = np.power(np.prod(normalized_open), 1.0/len(normalized_open))
        index_close = np.power(np.prod(normalized_close), 1.0/len(normalized_close))

        # Calculate strength as percentage change
        strength = ((index_close - index_open) / index_open) * 100
        strengths[currency] = strength

    # Convert to a pandas Series and sort
    strength_series = pd.Series(strengths)
    logger.info(f"Completed currency strength meter calculation. Last value: {strength_series.iloc[-1]}")
    return strength_series.sort_values(ascending=False)

def strength_meter_average_price(ohlc_data: Dict[str, pd.DataFrame], lookback_period: int = 12) -> pd.Series:
    """
    Calculates currency strength based on the average performance over a lookback period.

    This improved function calculates the synthetic index for each currency, then
    computes the average percentage change (Open to Close) over the specified
    number of recent periods (candles) to provide a more stable strength value.

    Args:
        ohlc_data (Dict[str, pd.DataFrame]):
            A dictionary of DataFrames for each currency pair. Each DataFrame
            must contain at least `lookback_period` rows of data.
        lookback_period (int, optional):
            The number of recent periods to average for the strength calculation.
            Defaults to 14.

    Returns:
        pd.Series: A pandas Series of currency strength values, sorted from
                   strongest to weakest.
    """
    logger.info(f"Calculating currency strength meter")
    CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    strengths = {}

    for currency in CURRENCIES:
        # --- Step 1: Slice the data for the lookback period ---
        # Create a new dictionary containing only the last `lookback_period` rows for each pair
        sliced_ohlc_data = {}
        for pair, df in ohlc_data.items():
            if len(df) < lookback_period:
                logger.warning(f"Warning: Data for {pair} has only {len(df)} rows, less than lookback period {lookback_period}. Skipping {currency}.")
                # Set strength to NaN and continue to next currency
                strengths[currency] = np.nan
                break
            sliced_ohlc_data[pair] = df.tail(lookback_period)
        
        # If a break occurred, continue to the next currency in the main loop
        if strengths.get(currency) is np.nan:
            continue

        # --- Step 2: Calculate the index for the sliced period ---
        index_df = currency_index(currency, sliced_ohlc_data)
        if index_df.empty:
            logger.error(f"Error: Index DataFrame is empty for {currency}. Skipping.")
            strengths[currency] = np.nan
            continue

        # --- Step 3: Calculate strength as the average performance ---
        # Calculate the percentage change for each period in the index
        period_performance = (index_df['Close'] - index_df['Open']) / index_df['Open'] * 100
        
        # The final strength is the average (mean) of these period performances
        average_strength = period_performance.mean()
        strengths[currency] = average_strength

    # Convert to a pandas Series, drop any currencies that failed, and sort
    strength_series = pd.Series(strengths).dropna()
    logger.info(f"Completed currency strength meter calculation. Last value: {strength_series.iloc[-1]}")
    return strength_series.sort_values(ascending=False)

def currency_strength_rsi(symbols=FOREX_SYMBOLS, timeframe="M5", strength_lookback=12, strength_rsi=12):
    """
    Calculate currency strength based on RSI values for a set of currency pairs.

    The function fetches historical data for each currency pair, computes the RSI for
    a specified lookback period, and aggregates these values to calculate the relative
    strength of major currencies (USD, EUR, GBP, CHF, JPY, AUD, CAD, NZD).

    Parameters:
        symbols (list of str): List of currency pairs (e.g., ['EURUSD', 'GBPUSD']).
        timeframe (str): Timeframe for fetching data (e.g., 'H1', 'D1').
        strength_lookback (int): Number of past periods to include in calculating RSI.
        strength_rsi (int): RSI period for calculation.
        strength_loc (int): Position index to fetch the latest strength readings.

    Returns:
        pd.Series: A series of currency strength values (sorted in descending order).
    """
    logger.info(f"Calculating currency strength")
    # Initialize MT5 client
    try:
        mt5_client = MT5Client()
    except Exception as e:
        logger.error(f"Could not initialize MT5Client: {e}")
        return pd.DataFrame()
    
    data = pd.DataFrame()
    for symbol in symbols:
        df = mt5_client.fetch_data(symbol, timeframe, start_pos=0, end_pos=strength_lookback)
        #df = fetch_data(symbol, timeframe, start_date="2025-02-01", end_date="2025-02-19")
        #df = fetch_data(symbol, timeframe, start_pos=140, end_pos=265)
        if df is not None and not df.empty:
            df = rsi(df, strength_rsi, "Close")
            data[symbol] = df['rsi']
        else:
            logger.warning(f"No data fetched for {symbol}")

    strength = pd.DataFrame()
    strength["USD"] = 1 / 7 * (
                (100 - data.EURUSD) + (100 - data.GBPUSD) + data.USDCAD + data.USDJPY + (100 - data.NZDUSD) + (
                    100 - data.AUDUSD) + data.USDCHF)
    strength["EUR"] = 1 / 7 * (data.EURUSD + data.EURGBP + data.EURAUD + data.EURNZD + data.EURCHF + data.EURCAD)
    strength["GBP"] = 1 / 7 * (
                data.GBPUSD + data.GBPJPY + data.GBPAUD + data.GBPNZD + data.GBPCAD + data.GBPCHF + (100 - data.EURGBP))
    strength["CHF"] = 1 / 7 * ((100 - data.EURCHF) + (100 - data.GBPCHF) + (100 - data.NZDCHF) + (100 - data.AUDCHF) + (
                100 - data.CADCHF) + data.CHFJPY + (100 - data.USDCHF))
    strength["JPY"] = 1 / 7 * ((100 - data.EURJPY) + (100 - data.GBPJPY) + (100 - data.USDJPY) + (100 - data.CHFJPY) + (
                100 - data.CADJPY) + (100 - data.NZDJPY) + (100 - data.AUDJPY))
    strength["AUD"] = 1 / 7 * ((100 - data.EURAUD) + (100 - data.GBPAUD) + (
                100 - data.AUDJPY) + data.AUDNZD + data.AUDCAD + data.AUDCHF + data.AUDUSD)
    strength["CAD"] = 1 / 7 * (
                (100 - data.EURCAD) + (100 - data.GBPCAD) + (100 - data.USDCAD) + data.CADJPY + (100 - data.AUDCAD) + (
                    100 - data.NZDCAD) + data.CADCHF)
    strength["NZD"] = 1 / 7 * (
                (100 - data.EURNZD) + (100 - data.GBPNZD) + data.NZDJPY + data.NZDUSD + data.NZDCAD + data.NZDCHF + (
                    100 - data.AUDNZD))

    strength_df = strength.shift(1)  # Shift all columns by one row

    # strength_df = strength_df.diff().iloc[-1].round(2).sort_values(ascending=False)  # Differance between current and previous
    #strength_df = strength.iloc[-1].apply(lambda x: x - 50).round(2).sort_values(ascending=False) # Differance from neutral 50

    # strength_df = strength_df.diff().round(2) # Differance between current and previous dataframe
    strength_df = strength_df.map(lambda x: round(x - 50, 2))  # Differance from neutral 50 dataframe

    logger.info(f"Completed currency strength calculation. Last value: {strength_df.iloc[-1]}")

    return strength_df

def ltf_close_above_below_hft(ltf_df, htf_df):
    """
    This function aligns lower time frame (LTF) data with higher time frame (HTF) data
    to analyze the close price movement above or below previous HTF extremes.

    Parameters:
        ltf_df (pd.DataFrame): DataFrame containing lower time frame data.
        htf_df (pd.DataFrame): DataFrame containing higher time frame data.

    Returns:
        pd.DataFrame: Updated DataFrame with additional columns including swingline
                      and significant close alerts.
    """
    logger.info(f"Calculating LTF close above or below HTF extremes")

    # Shift htf_df close to ensure we only compare to fully closed m5 bars
    htf_df['prev_High_HTF'] = htf_df['High'].shift(1)
    htf_df['prev_Low_HTF'] = htf_df['Low'].shift(1)

    # Merge the two DataFrames on their datetime index
    aligned_df = pd.merge(
        ltf_df, htf_df,
        how='outer',  # Use 'outer' to keep all timestamps from both DataFrames
        left_index=True,
        right_index=True,
        suffixes=('', '_htf')  # Add suffixes to differentiate columns from m1_df and m5_df
    )

    # Forward-fill missing values using `ffill` directly
    aligned_df.ffill(inplace=True)  # Forward-fill missing values in place

    aligned_df = aligned_df[["Open", "High", "prev_High_HTF", "Low", "prev_Low_HTF", "Close"]]

    df = aligned_df.copy()

    # Calculate the significant close and then the swingline direction
    df['swingline'] = np.select(
        [
            (df['Close'] > df['prev_High_HTF']) & (df['Close'] > df['Open']),  # Condition for 1
            (df['Close'] < df['prev_Low_HTF']) & (df['Close'] < df['Open'])  # Condition for -1
        ],
        [1, -1],  # Values to assign (1 for the first condition, -1 for the second condition)
        default=np.nan  # Set default to NaN to allow forward filling later
    )

    df['swingline'] = pd.Series(df['swingline']).ffill()  # Fill NaN values with the previous value (propagate values where conditions are not met)

    return df

######################## Bundled Indicators in Classes ###############

class CandlestickPatterns:
    """
    Candlestick pattern recognition class.
    
    This class provides methods to identify various candlestick patterns
    in financial market data, including doji, engulfing patterns, and more.
    """

    def __init__(self):
        """
        Initialize the CandlestickPatterns analyzer.
        """
        logger.info("CandlestickPatterns initialized")

    def doji(self, df: pd.DataFrame, tolerance: float = 0.1) -> pd.DataFrame:
        """
        Identifies doji candlestick patterns.
        
        A doji is a candlestick where the open and close prices are very close to each other,
        indicating indecision in the market.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            tolerance (float): Percentage tolerance for doji identification (default 0.1%)
            
        Returns:
            pd.DataFrame: DataFrame with added 'doji' column (1 for doji, 0 otherwise)
        """
        logger.info("Identifying doji patterns")
        
        df = df.copy()
        df['doji'] = 0
        
        if len(df) < 1:
            logger.warning("DataFrame too short for doji calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return df
        
        # Calculate body size and total range
        body_size = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        
        # Calculate tolerance threshold (percentage of total range)
        tolerance_threshold = total_range * (tolerance / 100)
        
        # Identify doji patterns
        # Doji: body size is less than tolerance threshold
        doji_mask = body_size <= tolerance_threshold
        
        df.loc[doji_mask, 'doji'] = 1
        
        doji_count = doji_mask.sum()
        logger.info(f"Doji pattern identification completed. Found {doji_count} doji patterns")
        
        return df

    def engulfing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies bullish and bearish engulfing patterns.
        
        Bullish engulfing: current candle completely engulfs the previous bearish candle
        Bearish engulfing: current candle completely engulfs the previous bullish candle
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with added 'engulfing' column (1 for bullish, -1 for bearish, 0 otherwise)
        """
        logger.info("Identifying engulfing patterns")
        
        df = df.copy()
        df['engulfing'] = 0
        
        if len(df) < 2:
            logger.warning("DataFrame too short for engulfing calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return df
        
        # Calculate body sizes and determine candle types
        current_body_size = df['Close'] - df['Open']
        previous_body_size = df['Close'].shift(1) - df['Open'].shift(1)
        
        # Determine if candles are bullish or bearish
        current_bullish = current_body_size > 0
        previous_bullish = previous_body_size > 0
        
        # Calculate engulfing conditions
        # Bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (
            current_bullish &  # Current candle is bullish
            ~previous_bullish &  # Previous candle is bearish
            (df['Open'] <= df['Close'].shift(1)) &  # Current open <= previous close
            (df['Close'] >= df['Open'].shift(1))  # Current close >= previous open
        )
        
        # Bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (
            ~current_bullish &  # Current candle is bearish
            previous_bullish &  # Previous candle is bullish
            (df['Open'] >= df['Close'].shift(1)) &  # Current open >= previous close
            (df['Close'] <= df['Open'].shift(1))  # Current close <= previous open
        )
        
        # Set engulfing values
        df.loc[bullish_engulfing, 'engulfing'] = 1
        df.loc[bearish_engulfing, 'engulfing'] = -1
        
        bullish_count = bullish_engulfing.sum()
        bearish_count = bearish_engulfing.sum()
        total_count = bullish_count + bearish_count
        
        logger.info(f"Engulfing pattern identification completed. Found {total_count} engulfing patterns ({bullish_count} bullish, {bearish_count} bearish)")
        
        return df

    def pinbar(self, df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 0.6) -> pd.DataFrame:
        """
        Identifies pinbar (hammer/hanging man) candlestick patterns.
        
        A pinbar has a small body and a long shadow (wick) in one direction,
        indicating potential reversal signals.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            body_ratio (float): Maximum ratio of body to total range for pinbar (default 0.3)
            shadow_ratio (float): Minimum ratio of shadow to total range for pinbar (default 0.6)
            
        Returns:
            pd.DataFrame: DataFrame with added 'pinbar' column (1 for bullish pinbar, -1 for bearish pinbar, 0 otherwise)
        """
        logger.info("Identifying pinbar patterns")
        
        df = df.copy()
        df['pinbar'] = 0
        
        if len(df) < 1:
            logger.warning("DataFrame too short for pinbar calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return df
        
        # Calculate body size and total range
        body_size = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        
        # Calculate upper and lower shadows
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Determine if body is small enough (body ratio condition)
        body_condition = body_size <= (total_range * body_ratio)
        
        # Determine shadow conditions
        upper_shadow_condition = upper_shadow >= (total_range * shadow_ratio)
        lower_shadow_condition = lower_shadow >= (total_range * shadow_ratio)
        
        # Identify pinbar patterns
        # Bullish pinbar: long lower shadow, small body
        bullish_pinbar = body_condition & lower_shadow_condition & (lower_shadow > upper_shadow)
        
        # Bearish pinbar: long upper shadow, small body
        bearish_pinbar = body_condition & upper_shadow_condition & (upper_shadow > lower_shadow)
        
        # Set pinbar values
        df.loc[bullish_pinbar, 'pinbar'] = 1
        df.loc[bearish_pinbar, 'pinbar'] = -1
        
        bullish_count = bullish_pinbar.sum()
        bearish_count = bearish_pinbar.sum()
        total_count = bullish_count + bearish_count
        
        logger.info(f"Pinbar pattern identification completed. Found {total_count} pinbar patterns ({bullish_count} bullish, {bearish_count} bearish)")
        
        return df

    def marubozu(self, df: pd.DataFrame, shadow_tolerance: float = 0.05) -> pd.DataFrame:
        """
        Identifies marubozu candlestick patterns.
        
        A marubozu is a candlestick with no or very small shadows (wicks),
        indicating strong directional momentum.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            shadow_tolerance (float): Maximum ratio of shadow to total range (default 0.05 = 5%)
            
        Returns:
            pd.DataFrame: DataFrame with added 'marubozu' column (1 for bullish marubozu, -1 for bearish marubozu, 0 otherwise)
        """
        logger.info("Identifying marubozu patterns")
        
        df = df.copy()
        df['marubozu'] = 0
        
        if len(df) < 1:
            logger.warning("DataFrame too short for marubozu calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return df
        
        # Calculate body size and total range
        body_size = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        
        # Calculate upper and lower shadows
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Determine if shadows are small enough (marubozu condition)
        shadow_condition = (upper_shadow <= total_range * shadow_tolerance) & (lower_shadow <= total_range * shadow_tolerance)
        
        # Determine if body is substantial (not a doji)
        body_condition = body_size > (total_range * 0.1)  # Body should be at least 10% of total range
        
        # Determine candle direction
        bullish = df['Close'] > df['Open']
        bearish = df['Close'] < df['Open']
        
        # Identify marubozu patterns
        # Bullish marubozu: bullish candle with minimal shadows
        bullish_marubozu = shadow_condition & body_condition & bullish
        
        # Bearish marubozu: bearish candle with minimal shadows
        bearish_marubozu = shadow_condition & body_condition & bearish
        
        # Set marubozu values
        df.loc[bullish_marubozu, 'marubozu'] = 1
        df.loc[bearish_marubozu, 'marubozu'] = -1
        
        bullish_count = bullish_marubozu.sum()
        bearish_count = bearish_marubozu.sum()
        total_count = bullish_count + bearish_count
        
        logger.info(f"Marubozu pattern identification completed. Found {total_count} marubozu patterns ({bullish_count} bullish, {bearish_count} bearish)")
        
        return df

class SmartMoneyConcepts:
    """
    Smart Money Concepts (SMC) analysis class.
    
    This class provides methods to identify and analyze smart money movements
    in financial markets, including swing highs/lows, swing points, order blocks,
    fair value gaps, and liquidity zones.
    """

    def __init__(self, mt5_client, symbol: str, min_swing_length: int = 3, min_pip_range: int = 3):
        """
        Initialize the SmartMoneyConcepts analyzer.
        """
        self.mt5_client = mt5_client
        self.symbol = symbol
        self.min_swing_length = min_swing_length
        self.min_pip_range = min_pip_range
        self.symbol_info = None
        
        # Try to get pip value from MT5, with fallback
        try:
            self.symbol_info = self.mt5_client.get_symbol_info(self.symbol)
            if self.symbol_info and hasattr(self.symbol_info, 'point') and self.symbol_info.point:
                self.pip_value = self.symbol_info.point * 10  # Convert point to pip value
            else:
                # Fallback for currency indices or when symbol not found
                self.pip_value = 0.0001  # Default pip value for most forex pairs
                logger.warning(f"Symbol {self.symbol} not found in MT5 or has no 'point' attribute, using default pip value")
        except Exception as e:
            # Fallback for any MT5 errors
            self.pip_value = 0.0001  # Default pip value for most forex pairs
            logger.warning(f"Error getting symbol info for {self.symbol}: {e}, using default pip value")
        
        logger.info(f"SMC initialized for {self.symbol} with pip value: {self.pip_value}")

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
        - HOD_swing: The highest ever swingvalue reached (all-time high).
        - LOD_swing: The lowest ever swingvalue reached (all-time low).
        - Reversal_Trigger: Reversal trigger indicator (1 for bearish reversal, -1 for bullish reversal, 0 for no trigger).

        Args:
            df (pd.DataFrame): Input DataFrame containing 'High' and 'Low' columns.

        Returns:
            pd.DataFrame: The DataFrame with the seven new columns added.
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
        hod_swing_col = [np.nan] * len(df)
        lod_swing_col = [np.nan] * len(df)
        reversal_trigger_col = [0] * len(df)

        # --- Initialize State Variables from the first row ---
        # The logic starts with swing_direction = -1, so we begin in a downswing state.
        swing_direction = -1
        HighestHigh = df['High'].iloc[0]
        LowestLow = df['Low'].iloc[0]
        LowestHigh = df['High'].iloc[0]
        HighestLow = df['Low'].iloc[0]
        
        # Initialize HOD and LOD tracking
        HOD_swing = LowestLow  # Start with the first swingvalue
        LOD_swing = LowestLow  # Start with the first swingvalue
        
        # Initialize reversal trigger tracking
        reversal_trigger = 0

        # --- Set initial values for the first row ---
        swingline[0] = swing_direction
        swing_value[0] = LowestLow  # In a downswing, swingvalue is LowestLow
        highest_low_col[0] = HighestLow
        lowest_high_col[0] = LowestHigh
        hod_swing_col[0] = HOD_swing
        lod_swing_col[0] = LOD_swing
        reversal_trigger_col[0] = reversal_trigger
 
        
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

            # Determine the current swingvalue based on the swing direction
            current_swingvalue = HighestLow if swing_direction == 1 else LowestHigh
            
            # Update HOD and LOD tracking
            if current_swingvalue > HOD_swing:
                HOD_swing = current_swingvalue
            if current_swingvalue < LOD_swing:
                LOD_swing = current_swingvalue

            # Check for reversal triggers
            if swing_direction == -1:  # Downswing
                if low < LOD_swing and reversal_trigger == 0:
                    reversal_trigger = 1
            elif swing_direction == 1:  # Upswing
                if high > HOD_swing and reversal_trigger == 0:
                    reversal_trigger = -1
            
            # Reset reversal trigger when swing direction changes
            if i > 0 and swingline[i-1] != swing_direction:
                reversal_trigger = 0

            # Append the current state to our lists
            swingline[i] = swing_direction
            swing_value[i] = current_swingvalue
            highest_low_col[i] = HighestLow
            lowest_high_col[i] = LowestHigh
            hod_swing_col[i] = HOD_swing
            lod_swing_col[i] = LOD_swing
            reversal_trigger_col[i] = reversal_trigger

        # Add the lists as new columns to the DataFrame
        df['swingline'] = swingline
        df['swingvalue'] = swing_value
        df['highest_low'] = highest_low_col
        df['lowest_high'] = lowest_high_col
        df['HOD_swing'] = hod_swing_col
        df['LOD_swing'] = lod_swing_col
        df['Reversal_Trigger'] = reversal_trigger_col

        logger.info(f"Finalizing swingline calculation. Final swingline: { df['swingline'].iloc[-1]}")
        
        return df

    def _add_h1_swingline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts dataframe to H1 timeframe, calculates swingline, and merges back to original timeframe.
        
        Args:
            df (pd.DataFrame): Original dataframe with datetime index and OHLC data
            
        Returns:
            pd.DataFrame: Original dataframe with added swinglineH1 and swingvalueH1 columns
        """
        logger.info("Calculating H1 swingline and merging back to original timeframe")
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex for H1 resampling")
            return df
        
        # Create a copy to avoid modifying original
        df_original = df.copy()
        
        # Resample to H1 timeframe
        # Use OHLC aggregation rules for proper candle formation
        ohlc_agg = {
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum' if 'Volume' in df.columns else 'sum'
        }
        
        # Resample to H1
        df_h1 = df.resample('h', label='right', closed='right').agg(ohlc_agg).dropna()
        
        if len(df_h1) < 2:
            logger.warning("H1 resampled data too short for swingline calculation")
            return df_original
        
        # Calculate swingline on H1 data
        df_h1 = pd.DataFrame(df_h1)
        df_h1 = self.calculate_swingline(df_h1)
        
        # Select only the swingline and swingvalue columns for merging
        h1_swingline_data = df_h1[['swingline', 'swingvalue']].copy()
        
        # Create new DataFrame with renamed columns
        h1_swingline_data = pd.DataFrame({
            'swinglineH1': h1_swingline_data['swingline'],
            'swingvalueH1': h1_swingline_data['swingvalue']
        }, index=h1_swingline_data.index)
        
        # Forward fill the H1 swingline data to match the original timeframe
        # This ensures each original timeframe candle gets the H1 swingline values
        h1_swingline_filled = h1_swingline_data.reindex(df_original.index, method='ffill')
        
        # Merge back to original dataframe
        df_original['swinglineH1'] = h1_swingline_filled['swinglineH1']
        df_original['swingvalueH1'] = h1_swingline_filled['swingvalueH1']
        
        logger.info(f"H1 swingline calculation completed. Added swinglineH1 and swingvalueH1 columns")
        
        return df_original

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
                group_id_int = int(group_id) if isinstance(group_id, (int, float, str)) else 0
                if group_id_int > 1:
                    prev_group_id = group_id_int - 1
                    prev_group = df[group_ids == prev_group_id]

                    # Handle case where swingline is flat but lower/higher than the last same direction swingline, then keep it dont remove it 
                    # Usually cased by news, rapid price movement or midnight gaps
                    if group_id_int > 2:  # Only check if there's a group 2 positions back
                        same_swing_prev_group_id = group_id_int - 2
                        same_swing_prev_group = df[group_ids == same_swing_prev_group_id]
                        if len(same_swing_prev_group) > 0:  # Check if the group exists and has data
                            group_df = pd.DataFrame(group_data)
                            prev_group_df = pd.DataFrame(same_swing_prev_group)
                            if group_df['swingline'].iloc[-1] == 1 and prev_group_df['swingline'].iloc[-1] == 1 and group_df['swingvalue'].iloc[-1] > prev_group_df['swingvalue'].iloc[-1]:
                                continue
                            if group_df['swingline'].iloc[-1] == -1 and prev_group_df['swingline'].iloc[-1] == -1 and group_df['swingvalue'].iloc[-1] < prev_group_df['swingvalue'].iloc[-1]:
                                continue

                    if len(prev_group) > 0:
                        prev_group_df = pd.DataFrame(prev_group)
                        prev_swingline_value = prev_group_df['swingline'].iloc[0]
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

        start_pos_int = int(start_pos) if isinstance(start_pos, (int, float, str)) else 0
        for i in range(start_pos_int + 1, len(df)):
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
        Generates BOS retest signals based on trend_bias, swingline, swingvalue, BOS, and High/Low.
        It adds a column 'retest_signal' to df.
        
        Signals continue on subsequent candles until swingline switches direction.
        Each BOS level can only be retested ONCE - after swingline changes, that level is consumed.

        Signal Logic:
        - Signal = 1 (Bullish): trend_bias = 1 and swingline = -1 and Low < swingvalue of the last BOS (==1), 
          retesting last bullish broken level (continues until swingline switches, then level consumed)
        - Signal = -1 (Bearish): trend_bias = -1 and swingline = 1 and High > swingvalue of the last BOS (== -1), 
          retesting last bearish broken level (continues until swingline switches, then level consumed)

        Args:
            df (pd.DataFrame): DataFrame with trend_bias, swingline, swingvalue, BOS, High, and Low columns

        Returns:
            pd.DataFrame: DataFrame with added retest_signal column
        """
        logger.info("Calculating BOS retest signals (each level consumed after first retest)")
        
        df = df.copy()
        
        # Initialize retest_signal column
        df['retest_signal'] = np.nan
        
        if len(df) < 2:
            logger.warning("DataFrame too short for BOS retest signal calculation")
            return df
        
        # Check if required columns exist
        required_columns = ['trend_bias', 'swingline', 'swingvalue', 'BOS', 'High', 'Low']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Track the last BOS events and their swingvalues
        last_bullish_bos_swingvalue = None
        last_bearish_bos_swingvalue = None
        
        # Track previous swingline to detect direction changes
        prev_swingline = None
        
        # Track if we're currently in a retest phase
        in_bullish_retest = False
        in_bearish_retest = False
        
        # Track consumed BOS levels (levels that have been retested and should not be retested again)
        consumed_bullish_bos_levels = set()
        consumed_bearish_bos_levels = set()
        
        # Process each row to identify BOS retest signals
        for i in range(len(df)):
            current = df.iloc[i]
            current_swingline = current['swingline'] if pd.notna(current['swingline']) else None
            
            # Flag to skip retest checking when swingline just changed
            swingline_just_changed = False
            
            # Update last BOS swingvalues when BOS events occur
            if pd.notna(current['BOS']):
                if current['BOS'] == 1:  # Bullish BOS
                    last_bullish_bos_swingvalue = current['swingvalue']
                    in_bullish_retest = False  # Reset retest state for new BOS
                    logger.debug(f"Updated last bullish BOS swingvalue to {last_bullish_bos_swingvalue} at index {df.index[i]}")
                elif current['BOS'] == -1:  # Bearish BOS
                    last_bearish_bos_swingvalue = current['swingvalue']
                    in_bearish_retest = False  # Reset retest state for new BOS
                    logger.debug(f"Updated last bearish BOS swingvalue to {last_bearish_bos_swingvalue} at index {df.index[i]}")
            
            # Check for swingline direction change to reset retest phases
            if prev_swingline is not None and current_swingline is not None:
                if prev_swingline != current_swingline:
                    swingline_just_changed = True
                    # Swingline changed direction, exit current retest phases and mark BOS levels as consumed
                    if in_bullish_retest and current_swingline == 1:  # Changed from -1 to 1
                        in_bullish_retest = False
                        # Mark the bullish BOS level as consumed (cannot be retested again)
                        if last_bullish_bos_swingvalue is not None:
                            consumed_bullish_bos_levels.add(last_bullish_bos_swingvalue)
                            logger.debug(f"Swingline changed from -1 to 1 at index {df.index[i]} - exiting bullish retest and consuming BOS level {last_bullish_bos_swingvalue}")
                    elif in_bearish_retest and current_swingline == -1:  # Changed from 1 to -1
                        in_bearish_retest = False
                        # Mark the bearish BOS level as consumed (cannot be retested again)
                        if last_bearish_bos_swingvalue is not None:
                            consumed_bearish_bos_levels.add(last_bearish_bos_swingvalue)
                            logger.debug(f"Swingline changed from 1 to -1 at index {df.index[i]} - exiting bearish retest and consuming BOS level {last_bearish_bos_swingvalue}")
                    
                    # Reset both retest phases when swingline changes
                    in_bullish_retest = False
                    in_bearish_retest = False
                    logger.debug(f"Reset all retest phases at index {df.index[i]} due to swingline change from {prev_swingline} to {current_swingline}")
            
            # Check for retest conditions only if we have all required data AND swingline didn't just change
            if (pd.notna(current['trend_bias']) and 
                pd.notna(current['swingline']) and 
                pd.notna(current['High']) and 
                pd.notna(current['Low']) and
                not swingline_just_changed):
                
                # Bullish retest signal
                # trend_bias = 1 and swingline = -1 and Low < swingvalue of the last BOS (==1) and level not consumed
                if (current['trend_bias'] == 1 and 
                    current['swingline'] == -1 and 
                    last_bullish_bos_swingvalue is not None and
                    current['Low'] < last_bullish_bos_swingvalue and
                    last_bullish_bos_swingvalue not in consumed_bullish_bos_levels):
                    
                    df.at[df.index[i], 'retest_signal'] = 1
                    if not in_bullish_retest:
                        in_bullish_retest = True
                        logger.debug(f"Entering bullish retest phase at index {df.index[i]}: "
                                   f"trend_bias={current['trend_bias']}, swingline={current['swingline']}, "
                                   f"Low={current['Low']:.5f} < last_bullish_BOS_swingvalue={last_bullish_bos_swingvalue:.5f}")
                    else:
                        logger.debug(f"Continuing bullish retest signal at index {df.index[i]}")
                
                # Bearish retest signal  
                # trend_bias = -1 and swingline = 1 and High > swingvalue of the last BOS (== -1) and level not consumed
                elif (current['trend_bias'] == -1 and 
                      current['swingline'] == 1 and 
                      last_bearish_bos_swingvalue is not None and
                      current['High'] > last_bearish_bos_swingvalue and
                      last_bearish_bos_swingvalue not in consumed_bearish_bos_levels):
                    
                    df.at[df.index[i], 'retest_signal'] = -1
                    if not in_bearish_retest:
                        in_bearish_retest = True
                        logger.debug(f"Entering bearish retest phase at index {df.index[i]}: "
                                   f"trend_bias={current['trend_bias']}, swingline={current['swingline']}, "
                                   f"High={current['High']:.5f} > last_bearish_BOS_swingvalue={last_bearish_bos_swingvalue:.5f}")
                    else:
                        logger.debug(f"Continuing bearish retest signal at index {df.index[i]}")
                
                # If conditions are no longer met, exit retest phases
                else:
                    if in_bullish_retest:
                        in_bullish_retest = False
                        logger.debug(f"Exiting bullish retest phase at index {df.index[i]} - conditions no longer met")
                    if in_bearish_retest:
                        in_bearish_retest = False
                        logger.debug(f"Exiting bearish retest phase at index {df.index[i]} - conditions no longer met")
            
            # Update previous swingline for next iteration
            if current_swingline is not None:
                prev_swingline = current_swingline
        
        # Count and log results
        retest_signals = df[df['retest_signal'].notna()]
        bullish_signals = len(df[df['retest_signal'] == 1])
        bearish_signals = len(df[df['retest_signal'] == -1])
        
        logger.info(f"BOS retest signal calculation completed. Found {len(retest_signals)} retest signals "
                   f"({bullish_signals} bullish, {bearish_signals} bearish) - each BOS level retested only once")
        
        return df


    def run_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the Smart Money Concepts indicators on the DataFrame.
        """
        #df = self.identify_fair_value_gaps(df, join_consecutive=True)
        
        
        # Calculate H1 swingline
        df = self._add_h1_swingline(df)
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

###################### Plotting #########################################################

def plot_candlestick(df, title="Candlestick Chart", volume=False, filename="candlestick_chart.html"):
    """
    Plots a candlestick chart and auto-opens it in the browser as an HTML file.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close'] and a datetime index.
        title (str): Title of the chart.
        volume (bool): Whether to plot volume as a subplot (requires 'Volume' column).
        filename (str): Output HTML file name.
    """
    import plotly.graph_objs as go
    import plotly.offline as pyo
    import webbrowser
    import os

    # Ensure required columns exist
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df_plot = df.copy()
    if not isinstance(df_plot.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        df_plot.index = pd.to_datetime(df_plot.index)

    data = [
        go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name="Candles"
        )
    ]

    if volume and 'Volume' in df_plot.columns:
        data.append(
            go.Bar(
                x=df_plot.index,
                y=df_plot['Volume'],
                name="Volume",
                marker=dict(color='rgba(128,128,128,0.3)'),
                yaxis='y2'
            )
        )
        layout = go.Layout(
            title=title,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(orientation="h"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
    else:
        layout = go.Layout(
            title=title,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            margin=dict(l=40, r=40, t=40, b=40)
        )

    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename=filename, auto_open=False)
    # Open in default browser
    abs_path = os.path.abspath(filename)
    webbrowser.open(f"file://{abs_path}")



####################### UTILS #########################################################

def resample_data(df, timeframe='D', ohlc_dict=None, volume_col='Volume'):
    """
    Resample OHLCV dataframe to a new timeframe.

    Args:
        df (pd.DataFrame): DataFrame with at least ['Open', 'High', 'Low', 'Close'] columns.
        timeframe (str): Pandas resample rule (e.g., 'D' for daily, 'H' for hourly).
        ohlc_dict (dict, optional): Custom aggregation for OHLC columns.
        volume_col (str): Name of the volume column, if present.

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    if ohlc_dict is None:
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        if volume_col in df.columns:
            ohlc_dict[volume_col] = 'sum'

    resampled = df.resample(timeframe).agg(ohlc_dict)
    # Drop rows where 'Open' is NaN (no data in that period)
    resampled = resampled.dropna(subset=['Open'])
    return resampled






###################### MAIN (For Testing Indicators)#########################################################

if __name__ == "__main__":

    # Fetching data
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 6, 30)
    start_date_core = start_date - timedelta(days=15) # Remove 15 days for start date for df_core

    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)

    symbol = "EURUSD"
    symbol_info = mt5_client.get_symbol_info(symbol)

    data = mt5_client.fetch_data(symbol, "M5", start_date=start_date, end_date=end_date)

    # sub-df for plotting
    #df_plot = data[:100]
    #plot_candlestick(df_plot, title="EURUSD M5")

    # Resample to Daily
    # df_plot = resample_data(data, timeframe='D')
    # print(df_plot)
    # plot_candlestick(df_plot, title="EURUSD D")

    import bt
    #price_data = bt.get("aapl", start=start_date, end=end_date)
    #print(price_data)

    data = ma(data, period=10, ma_type="EMA", column='Close')
    data = ma(data, period=20, ma_type="EMA", column='Close')
    data["signal"] = 0

    data["signal"][data.ema_10 > data.ema_20] = 1
    data["signal"][data.ema_10 < data.ema_20] = -1
    print(data)
  




































