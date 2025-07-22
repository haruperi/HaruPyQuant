# Technical Indicators Documentation

This document provides comprehensive documentation for all technical indicators available in the HaruPyQuant strategy module. Each indicator includes detailed descriptions, parameters, return values, and practical use case examples.

## Table of Contents

1. [Standard Indicators](#standard-indicators)
   - [Moving Average (MA)](#moving-average-ma)
   - [Relative Strength Index (RSI)](#relative-strength-index-rsi)
   - [Williams %R](#williams-r)
   - [Average True Range (ATR)](#average-true-range-atr)

2. [Custom Indicators](#custom-indicators)
   - [Average Daily Range (ADR)](#average-daily-range-adr)
   - [Currency Index](#currency-index)
   - [Currency Strength Meter](#currency-strength-meter)
   - [Currency Strength RSI](#currency-strength-rsi)
   - [LTF Close Above/Below HTF](#ltf-close-abovebelow-htf)

3. [Candlestick Patterns](#candlestick-patterns)
   - [Doji](#doji)
   - [Engulfing Patterns](#engulfing-patterns)
   - [Pinbar](#pinbar)
   - [Marubozu](#marubozu)

4. [Smart Money Concepts (SMC)](#smart-money-concepts-smc)
   - [Swingline Calculation](#swingline-calculation)
   - [H1 Swingline](#h1-swingline)
   - [Swing Points](#swing-points)
   - [Support/Resistance Levels](#supportresistance-levels)
   - [Fair Value Gaps (FVG)](#fair-value-gaps-fvg)
   - [Order Blocks](#order-blocks)
   - [Break of Structure (BOS)](#break-of-structure-bos)
   - [Change of Character (CHoCH)](#change-of-character-choch)
   - [Retracements](#retracements)
   - [Fibonacci Signals](#fibonacci-signals)
   - [BOS Retest Signals](#bos-retest-signals)

---

## Standard Indicators

### Moving Average (MA)

**Function:** `calculate_ma(df, period, ma_type, column)`

Calculates various types of moving averages and adds them to the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing price data
- `period` (int): MA period (window/span), default from config
- `ma_type` (str): Type of moving average ("EMA", "SMA", "WMA"), default "EMA"
- `column` (str): Column name to use for calculation, default 'Close'

**Returns:**
- DataFrame with added MA column (e.g., 'ema_20', 'sma_50')

**Use Case Example:**
```python
from app.strategy.indicators import calculate_ma
from app.data.mt5_client import MT5Client

# Initialize MT5 client
mt5_client = MT5Client()

# Fetch EURUSD data
df = mt5_client.fetch_data("EURUSD", "H1", start_pos=0, end_pos=100)

# Calculate 20-period EMA
df = calculate_ma(df, period=20, ma_type="EMA", column='Close')
print(f"Latest EMA: {df['ema_20'].iloc[-1]}")

# Calculate 50-period SMA
df = calculate_ma(df, period=50, ma_type="SMA", column='Close')
print(f"Latest SMA: {df['sma_50'].iloc[-1]}")

# Trading strategy: Buy when price crosses above EMA
if df['Close'].iloc[-1] > df['ema_20'].iloc[-1] and df['Close'].iloc[-2] <= df['ema_20'].iloc[-2]:
    print("Bullish signal: Price crossed above EMA")
```

### Relative Strength Index (RSI)

**Function:** `calculate_rsi(df, period, column)`

Calculates the Relative Strength Index, a momentum oscillator that measures the speed and magnitude of price changes.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing price data
- `period` (int): RSI period, default from config
- `column` (str): Column name to use for calculation, default 'Close'

**Returns:**
- DataFrame with added 'rsi' column

**Use Case Example:**
```python
from app.strategy.indicators import calculate_rsi

# Calculate 14-period RSI
df = calculate_rsi(df, period=14, column='Close')

# Trading strategy: RSI oversold/overbought signals
latest_rsi = df['rsi'].iloc[-1]

if latest_rsi < 30:
    print(f"Oversold condition: RSI = {latest_rsi:.2f}")
elif latest_rsi > 70:
    print(f"Overbought condition: RSI = {latest_rsi:.2f}")

# RSI divergence detection
if (df['Close'].iloc[-1] > df['Close'].iloc[-5] and 
    df['rsi'].iloc[-1] < df['rsi'].iloc[-5]):
    print("Bearish divergence detected")
```

### Williams %R

**Function:** `calculate_williams_percent(df, period)`

Calculates Williams %R, a momentum indicator that measures overbought/oversold levels.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing OHLC data
- `period` (int): Lookback period, default from config

**Returns:**
- DataFrame with added 'williams_r' column

**Use Case Example:**
```python
from app.strategy.indicators import calculate_williams_percent

# Calculate Williams %R
df = calculate_williams_percent(df, period=14)

# Trading strategy: Williams %R signals
latest_wr = df['williams_r'].iloc[-1]

if latest_wr < -80:
    print(f"Oversold: Williams %R = {latest_wr:.2f}")
elif latest_wr > -20:
    print(f"Overbought: Williams %R = {latest_wr:.2f}")

# Combined with RSI for confirmation
if (df['rsi'].iloc[-1] < 30 and df['williams_r'].iloc[-1] < -80):
    print("Strong oversold signal from both indicators")
```

### Average True Range (ATR)

**Function:** `calculate_atr(df, period)`

Calculates the Average True Range, a volatility indicator that measures market volatility.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing OHLC data
- `period` (int): ATR period, default from config

**Returns:**
- DataFrame with added 'atr' column

**Use Case Example:**
```python
from app.strategy.indicators import calculate_atr

# Calculate ATR
df = calculate_atr(df, period=14)

# Dynamic stop loss calculation
latest_atr = df['atr'].iloc[-1]
entry_price = 1.0850  # Example entry price

# Set stop loss at 2 ATR below entry for long position
stop_loss = entry_price - (2 * latest_atr)
print(f"Dynamic stop loss: {stop_loss:.5f}")

# Volatility-based position sizing
account_balance = 10000
risk_per_trade = 0.02  # 2% risk per trade
risk_amount = account_balance * risk_per_trade
position_size = risk_amount / (2 * latest_atr)
print(f"Position size: {position_size:.2f} lots")
```

---

## Custom Indicators

### Average Daily Range (ADR)

**Function:** `get_adr(df, symbol_info, period)`

Calculates the Average Daily Range and current daily range percentage for position sizing and stop loss calculation.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data
- `symbol_info`: MT5 symbol information
- `period` (int): Period for ADR calculation, default from config

**Returns:**
- DataFrame with added 'daily_range', 'ADR', 'SL' columns

**Use Case Example:**
```python
from app.strategy.indicators import get_adr
from app.data.mt5_client import MT5Client

mt5_client = MT5Client()
symbol_info = mt5_client.get_symbol_info("EURUSD")

# Calculate ADR
df = get_adr(df, symbol_info, period=20)

# ADR-based stop loss
latest_adr = df['ADR'].iloc[-1]
entry_price = 1.0850

# Set stop loss at 50% of ADR
stop_loss_pips = latest_adr * 0.5
stop_loss = entry_price - (stop_loss_pips * 0.0001)  # Convert pips to price
print(f"ADR-based stop loss: {stop_loss:.5f}")

# Current daily range analysis
current_range = df['daily_range'].iloc[-1]
if current_range > latest_adr * 1.5:
    print("High volatility day - consider wider stops")
```

### Currency Index

**Function:** `calculate_currency_index(base_currency, ohlc_data)`

Calculates a synthetic currency index using the geometric mean of constituent pairs.

**Parameters:**
- `base_currency` (str): Currency for index calculation (e.g., 'USD', 'EUR')
- `ohlc_data` (dict): Dictionary of currency pair DataFrames

**Returns:**
- DataFrame with synthetic OHLC for the currency index

**Use Case Example:**
```python
from app.strategy.indicators import calculate_currency_index
from app.data.mt5_client import MT5Client

mt5_client = MT5Client()

# Fetch data for USD index pairs
pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
ohlc_data = {}

for pair in pairs:
    df = mt5_client.fetch_data(pair, "H1", start_pos=0, end_pos=100)
    ohlc_data[pair] = df

# Calculate USD index
usd_index = calculate_currency_index('USD', ohlc_data)

# Analyze USD strength
usd_strength = (usd_index['Close'].iloc[-1] - usd_index['Open'].iloc[-1]) / usd_index['Open'].iloc[-1] * 100
print(f"USD strength: {usd_strength:.2f}%")

# Trading strategy: Strong USD = sell USD pairs
if usd_strength > 0.5:
    print("USD is strong - consider selling USD pairs")
```

### Currency Strength Meter

**Function:** `calculate_strength_meter_latest_price(ohlc_data)`

Calculates currency strength based on the most recent OHLC data.

**Parameters:**
- `ohlc_data` (dict): Dictionary of currency pair DataFrames

**Returns:**
- Series with currency strength values sorted from strongest to weakest

**Use Case Example:**
```python
from app.strategy.indicators import calculate_strength_meter_latest_price

# Calculate currency strength
strength_series = calculate_strength_meter_latest_price(ohlc_data)

print("Currency Strength Ranking:")
for currency, strength in strength_series.items():
    print(f"{currency}: {strength:.2f}%")

# Find strongest and weakest currencies
strongest = strength_series.index[0]
weakest = strength_series.index[-1]

# Trading strategy: Buy strongest vs weakest
if strongest != weakest:
    pair = f"{strongest}{weakest}"
    print(f"Consider buying {pair} (strongest vs weakest)")
```

### Currency Strength RSI

**Function:** `calculate_currency_strength_rsi(symbols, timeframe, strength_lookback, strength_rsi)`

Calculates currency strength based on RSI values for currency pairs.

**Parameters:**
- `symbols` (list): List of currency pairs
- `timeframe` (str): Timeframe for data
- `strength_lookback` (int): Number of periods for RSI calculation
- `strength_rsi` (int): RSI period

**Returns:**
- DataFrame with currency strength values

**Use Case Example:**
```python
from app.strategy.indicators import calculate_currency_strength_rsi

# Calculate currency strength using RSI
strength_df = calculate_currency_strength_rsi(
    symbols=['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
    timeframe="M5",
    strength_lookback=12,
    strength_rsi=12
)

# Get latest strength values
latest_strength = strength_df.iloc[-1]
print("Latest Currency Strength (RSI-based):")
for currency, strength in latest_strength.items():
    print(f"{currency}: {strength:.2f}")

# Find extreme strength values
strong_currencies = latest_strength[latest_strength > 10].index.tolist()
weak_currencies = latest_strength[latest_strength < -10].index.tolist()

print(f"Strong currencies: {strong_currencies}")
print(f"Weak currencies: {weak_currencies}")
```

### LTF Close Above/Below HTF

**Function:** `calculate_ltf_close_above_below_hft(ltf_df, htf_df)`

Analyzes lower timeframe close price movement relative to higher timeframe extremes.

**Parameters:**
- `ltf_df` (pd.DataFrame): Lower timeframe DataFrame
- `htf_df` (pd.DataFrame): Higher timeframe DataFrame

**Returns:**
- DataFrame with swingline and significant close alerts

**Use Case Example:**
```python
from app.strategy.indicators import calculate_ltf_close_above_below_htf

# Fetch M5 and H1 data
m5_df = mt5_client.fetch_data("EURUSD", "M5", start_pos=0, end_pos=200)
h1_df = mt5_client.fetch_data("EURUSD", "H1", start_pos=0, end_pos=50)

# Analyze LTF vs HTF
result_df = calculate_ltf_close_above_below_htf(m5_df, h1_df)

# Find significant breakouts
breakouts = result_df[result_df['swingline'].notna()]
print(f"Found {len(breakouts)} significant breakouts")

# Trading strategy: Trade in direction of HTF swingline
latest_swingline = result_df['swingline'].iloc[-1]
if latest_swingline == 1:
    print("HTF swingline is bullish - look for long opportunities")
elif latest_swingline == -1:
    print("HTF swingline is bearish - look for short opportunities")
```

---

## Candlestick Patterns

### Doji

**Function:** `CandlestickPatterns.doji(df, tolerance)`

Identifies doji candlestick patterns indicating market indecision.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data
- `tolerance` (float): Percentage tolerance for doji identification, default 0.1

**Returns:**
- DataFrame with added 'doji' column (1 for doji, 0 otherwise)

**Use Case Example:**
```python
from app.strategy.indicators import CandlestickPatterns

patterns = CandlestickPatterns()

# Identify doji patterns
df = patterns.doji(df, tolerance=0.1)

# Find recent doji patterns
recent_doji = df[df['doji'] == 1].tail(5)
print(f"Found {len(recent_doji)} doji patterns in recent data")

# Trading strategy: Doji at support/resistance
for idx in recent_doji.index:
    if (df.loc[idx, 'Close'] > df.loc[idx, 'Open'] and  # Bullish doji
        df.loc[idx, 'Low'] <= support_level):  # At support
        print(f"Bullish doji at support: {idx}")
```

### Engulfing Patterns

**Function:** `CandlestickPatterns.engulfing(df)`

Identifies bullish and bearish engulfing patterns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data

**Returns:**
- DataFrame with added 'engulfing' column (1 for bullish, -1 for bearish, 0 otherwise)

**Use Case Example:**
```python
# Identify engulfing patterns
df = patterns.engulfing(df)

# Find recent engulfing patterns
recent_engulfing = df[df['engulfing'] != 0].tail(5)
print(f"Found {len(recent_engulfing)} engulfing patterns")

# Trading strategy: Engulfing at key levels
for idx in recent_engulfing.index:
    if (df.loc[idx, 'engulfing'] == 1 and  # Bullish engulfing
        df.loc[idx, 'Low'] <= support_level):  # At support
        print(f"Bullish engulfing at support: {idx}")
    elif (df.loc[idx, 'engulfing'] == -1 and  # Bearish engulfing
          df.loc[idx, 'High'] >= resistance_level):  # At resistance
        print(f"Bearish engulfing at resistance: {idx}")
```

### Pinbar

**Function:** `CandlestickPatterns.pinbar(df, body_ratio, shadow_ratio)`

Identifies pinbar (hammer/hanging man) patterns with long shadows.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data
- `body_ratio` (float): Maximum body to total range ratio, default 0.3
- `shadow_ratio` (float): Minimum shadow to total range ratio, default 0.6

**Returns:**
- DataFrame with added 'pinbar' column (1 for bullish, -1 for bearish, 0 otherwise)

**Use Case Example:**
```python
# Identify pinbar patterns
df = patterns.pinbar(df, body_ratio=0.3, shadow_ratio=0.6)

# Find recent pinbars
recent_pinbars = df[df['pinbar'] != 0].tail(5)
print(f"Found {len(recent_pinbars)} pinbar patterns")

# Trading strategy: Pinbar at key levels
for idx in recent_pinbars.index:
    if (df.loc[idx, 'pinbar'] == 1 and  # Bullish pinbar
        df.loc[idx, 'Low'] <= support_level):  # At support
        print(f"Bullish pinbar at support: {idx}")
        # Entry: Above pinbar high
        entry_price = df.loc[idx, 'High']
        stop_loss = df.loc[idx, 'Low']
        print(f"Entry: {entry_price}, Stop: {stop_loss}")
```

### Marubozu

**Function:** `CandlestickPatterns.marubozu(df, shadow_tolerance)`

Identifies marubozu patterns with minimal shadows indicating strong momentum.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data
- `shadow_tolerance` (float): Maximum shadow to total range ratio, default 0.05

**Returns:**
- DataFrame with added 'marubozu' column (1 for bullish, -1 for bearish, 0 otherwise)

**Use Case Example:**
```python
# Identify marubozu patterns
df = patterns.marubozu(df, shadow_tolerance=0.05)

# Find recent marubozu patterns
recent_marubozu = df[df['marubozu'] != 0].tail(5)
print(f"Found {len(recent_marubozu)} marubozu patterns")

# Trading strategy: Marubozu continuation
for idx in recent_marubozu.index:
    if df.loc[idx, 'marubozu'] == 1:  # Bullish marubozu
        print(f"Strong bullish momentum: {idx}")
        # Look for continuation trades
        if df.loc[idx, 'Close'] > df.loc[idx, 'Open']:
            print("Consider long continuation trade")
```

---

## Smart Money Concepts (SMC)

### Swingline Calculation

**Function:** `SmartMoneyConcepts.calculate_swingline(df)`

Calculates swing trend lines and identifies market swing direction.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'High' and 'Low' columns

**Returns:**
- DataFrame with added swingline columns

**Use Case Example:**
```python
from app.strategy.indicators import SmartMoneyConcepts
from app.data.mt5_client import MT5Client

mt5_client = MT5Client()
smc = SmartMoneyConcepts(mt5_client, "EURUSD")

# Calculate swingline
df = smc.calculate_swingline(df)

# Analyze current swing direction
current_swing = df['swingline'].iloc[-1]
if current_swing == 1:
    print("Currently in upswing")
    swing_value = df['swingvalue'].iloc[-1]
    print(f"Swing value (support): {swing_value:.5f}")
elif current_swing == -1:
    print("Currently in downswing")
    swing_value = df['swingvalue'].iloc[-1]
    print(f"Swing value (resistance): {swing_value:.5f}")

# Trading strategy: Trade in swing direction
if current_swing == 1:
    # Look for long opportunities above swing value
    if df['Close'].iloc[-1] > swing_value:
        print("Price above swing value - bullish setup")
```

### H1 Swingline

**Function:** `SmartMoneyConcepts._add_h1_swingline(df)`

Converts data to H1 timeframe, calculates swingline, and merges back to original timeframe.

**Parameters:**
- `df` (pd.DataFrame): Original DataFrame with datetime index

**Returns:**
- DataFrame with added H1 swingline columns

**Use Case Example:**
```python
# Add H1 swingline to M5 data
df = smc._add_h1_swingline(df)

# Compare M5 and H1 swing directions
m5_swing = df['swingline'].iloc[-1]
h1_swing = df['swinglineH1'].iloc[-1]

if m5_swing == h1_swing:
    print("M5 and H1 swing directions aligned - stronger signal")
else:
    print("M5 and H1 swing directions conflicting - weaker signal")

# Trading strategy: Aligned timeframes
if m5_swing == 1 and h1_swing == 1:
    print("Strong bullish alignment - look for long trades")
elif m5_swing == -1 and h1_swing == -1:
    print("Strong bearish alignment - look for short trades")
```

### Swing Points

**Function:** `SmartMoneyConcepts.calculate_swing_points(df)`

Identifies fractal swing points based on swingline directions.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with swingline column

**Returns:**
- DataFrame with added swing point indicators

**Use Case Example:**
```python
# Calculate swing points
df = smc.calculate_swing_points(df)

# Find recent swing points
recent_swings = df[df['swingpoint'].notna()].tail(5)
print(f"Found {len(recent_swings)} recent swing points")

# Analyze swing point quality
for idx in recent_swings.index:
    swing_type = df.loc[idx, 'swingpoint']
    if swing_type == 1:  # Swing high
        print(f"Swing high at {idx}: {df.loc[idx, 'High']:.5f}")
    elif swing_type == -1:  # Swing low
        print(f"Swing low at {idx}: {df.loc[idx, 'Low']:.5f}")

# Trading strategy: Swing point retests
current_price = df['Close'].iloc[-1]
for idx in recent_swings.index:
    if df.loc[idx, 'swingpoint'] == 1:  # Swing high
        swing_high = df.loc[idx, 'High']
        if abs(current_price - swing_high) < 0.0005:  # Within 5 pips
            print(f"Price near swing high resistance: {swing_high:.5f}")
```

### Support/Resistance Levels

**Function:** `SmartMoneyConcepts.calculate_support_resistance_levels(df)`

Identifies dynamic support and resistance levels based on swing points.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with swingpoint column

**Returns:**
- DataFrame with added Support and Resistance columns

**Use Case Example:**
```python
# Calculate support/resistance levels
df = smc.calculate_support_resistance_levels(df)

# Get current levels
current_support = df['Support'].iloc[-1]
current_resistance = df['Resistance'].iloc[-1]
current_price = df['Close'].iloc[-1]

print(f"Current price: {current_price:.5f}")
if pd.notna(current_support):
    print(f"Support: {current_support:.5f}")
if pd.notna(current_resistance):
    print(f"Resistance: {current_resistance:.5f}")

# Trading strategy: Level-based entries
if pd.notna(current_support) and current_price <= current_support * 1.0001:
    print("Price at support - look for long entry")
elif pd.notna(current_resistance) and current_price >= current_resistance * 0.9999:
    print("Price at resistance - look for short entry")
```

### Fair Value Gaps (FVG)

**Function:** `SmartMoneyConcepts.identify_fair_value_gaps(df, join_consecutive)`

Identifies fair value gaps where price gaps occur between candles.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLC data
- `join_consecutive` (bool): Whether to merge consecutive FVGs, default True

**Returns:**
- DataFrame with added FVG indicator columns

**Use Case Example:**
```python
# Identify fair value gaps
df = smc.identify_fair_value_gaps(df, join_consecutive=True)

# Find recent FVGs
recent_fvgs = df[df['fvg_type'].notna()].tail(5)
print(f"Found {len(recent_fvgs)} recent fair value gaps")

# Analyze FVG characteristics
for idx in recent_fvgs.index:
    fvg_type = df.loc[idx, 'fvg_type']
    fvg_high = df.loc[idx, 'fvg_high']
    fvg_low = df.loc[idx, 'fvg_low']
    fvg_size = df.loc[idx, 'fvg_size']
    mitigated = df.loc[idx, 'fvg_mitigated_index']
    
    if fvg_type == 1:
        print(f"Bullish FVG: {fvg_low:.5f} - {fvg_high:.5f} ({fvg_size:.1f} pips)")
    else:
        print(f"Bearish FVG: {fvg_low:.5f} - {fvg_high:.5f} ({fvg_size:.1f} pips)")
    
    if pd.notna(mitigated):
        print(f"FVG mitigated at: {mitigated}")
    else:
        print("FVG still open")

# Trading strategy: FVG retests
current_price = df['Close'].iloc[-1]
for idx in recent_fvgs.index:
    if pd.isna(df.loc[idx, 'fvg_mitigated_index']):  # Unmitigated FVG
        fvg_high = df.loc[idx, 'fvg_high']
        fvg_low = df.loc[idx, 'fvg_low']
        fvg_type = df.loc[idx, 'fvg_type']
        
        if fvg_type == 1 and fvg_low <= current_price <= fvg_high:
            print(f"Price in bullish FVG - potential long entry")
        elif fvg_type == -1 and fvg_low <= current_price <= fvg_high:
            print(f"Price in bearish FVG - potential short entry")
```

### Order Blocks

**Function:** `SmartMoneyConcepts.calculate_order_blocks(df, close_mitigation)`

Identifies order blocks based on support/resistance regions.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with support/resistance data
- `close_mitigation` (bool): Whether to use close-based mitigation, default True

**Returns:**
- DataFrame with added order block columns

**Use Case Example:**
```python
# Calculate order blocks
df = smc.calculate_order_blocks(df, close_mitigation=True)

# Find active order blocks
bullish_obs = df[df['Bullish_Order_Block_Top'].notna()].tail(3)
bearish_obs = df[df['Bearish_Order_Block_Top'].notna()].tail(3)

print(f"Found {len(bullish_obs)} bullish order blocks")
print(f"Found {len(bearish_obs)} bearish order blocks")

# Analyze order block characteristics
current_price = df['Close'].iloc[-1]

for idx in bullish_obs.index:
    top = df.loc[idx, 'Bullish_Order_Block_Top']
    bottom = df.loc[idx, 'Bullish_Order_Block_Bottom']
    mitigated = df.loc[idx, 'Bullish_Order_Block_Mitigated']
    
    if not mitigated and bottom <= current_price <= top:
        print(f"Price in bullish order block: {bottom:.5f} - {top:.5f}")
        print("Potential long entry zone")

for idx in bearish_obs.index:
    top = df.loc[idx, 'Bearish_Order_Block_Top']
    bottom = df.loc[idx, 'Bearish_Order_Block_Bottom']
    mitigated = df.loc[idx, 'Bearish_Order_Block_Mitigated']
    
    if not mitigated and bottom <= current_price <= top:
        print(f"Price in bearish order block: {bottom:.5f} - {top:.5f}")
        print("Potential short entry zone")
```

### Break of Structure (BOS)

**Function:** `SmartMoneyConcepts.break_of_structure(df)`

Detects breaks of structure when support or resistance levels are broken.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with Support and Resistance columns

**Returns:**
- DataFrame with added BOS column

**Use Case Example:**
```python
# Calculate break of structure
df = smc.break_of_structure(df)

# Find recent BOS events
recent_bos = df[df['BOS'].notna()].tail(5)
print(f"Found {len(recent_bos)} recent break of structure events")

# Analyze BOS characteristics
for idx in recent_bos.index:
    bos_type = df.loc[idx, 'BOS']
    if bos_type == 1:
        print(f"Bullish BOS at {idx} - resistance broken")
        # Look for retest opportunities
        broken_resistance = df.loc[idx-1, 'Resistance']
        print(f"Broken resistance level: {broken_resistance:.5f}")
    elif bos_type == -1:
        print(f"Bearish BOS at {idx} - support broken")
        # Look for retest opportunities
        broken_support = df.loc[idx-1, 'Support']
        print(f"Broken support level: {broken_support:.5f}")

# Trading strategy: BOS confirmation
latest_bos = df['BOS'].iloc[-1]
if pd.notna(latest_bos):
    if latest_bos == 1:
        print("Recent bullish BOS - look for continuation trades")
    elif latest_bos == -1:
        print("Recent bearish BOS - look for continuation trades")
```

### Change of Character (CHoCH)

**Function:** `SmartMoneyConcepts.change_of_character(df)`

Identifies changes of character when price breaks above previous resistance or below previous support.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with BOS, swingvalue, Resistance, Support columns

**Returns:**
- DataFrame with added CHoCH and trend_bias columns

**Use Case Example:**
```python
# Calculate change of character
df = smc.change_of_character(df)

# Find recent CHoCH events
recent_choch = df[df['CHoCH'].notna()].tail(5)
print(f"Found {len(recent_choch)} recent change of character events")

# Analyze trend bias
current_trend_bias = df['trend_bias'].iloc[-1]
if pd.notna(current_trend_bias):
    if current_trend_bias == 1:
        print("Current trend bias: Bullish")
    elif current_trend_bias == -1:
        print("Current trend bias: Bearish")

# Trading strategy: Trend bias alignment
for idx in recent_choch.index:
    choch_type = df.loc[idx, 'CHoCH']
    if choch_type == 1:
        print(f"Bullish CHoCH at {idx} - trend changed to bullish")
        # Look for long opportunities
    elif choch_type == -1:
        print(f"Bearish CHoCH at {idx} - trend changed to bearish")
        # Look for short opportunities

# Confirm trend with multiple timeframes
if current_trend_bias == 1 and df['swinglineH1'].iloc[-1] == 1:
    print("Strong bullish alignment across timeframes")
elif current_trend_bias == -1 and df['swinglineH1'].iloc[-1] == -1:
    print("Strong bearish alignment across timeframes")
```

### Retracements

**Function:** `SmartMoneyConcepts.retracements(df)`

Calculates retracement percentages during counter-trend moves.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with trend_bias, swingline, Support, Resistance, Close columns

**Returns:**
- DataFrame with added retracement columns

**Use Case Example:**
```python
# Calculate retracements
df = smc.retracements(df)

# Analyze current retracement
current_retracement = df['CurrentRetracement'].iloc[-1]
deepest_retracement = df['DeepestRetracement'].iloc[-1]

if pd.notna(current_retracement):
    print(f"Current retracement: {current_retracement:.1f}%")
    print(f"Deepest retracement: {deepest_retracement:.1f}%")

# Trading strategy: Retracement-based entries
if pd.notna(current_retracement):
    if current_retracement > 61.8:  # Beyond Fibonacci 61.8%
        print("Deep retracement - potential reversal zone")
    elif current_retracement > 50:  # Beyond 50%
        print("Moderate retracement - continuation likely")
    elif current_retracement < 23.6:  # Below Fibonacci 23.6%
        print("Shallow retracement - strong trend")

# Find high retracement periods
high_retracements = df[df['CurrentRetracement'] > 50].tail(10)
print(f"Found {len(high_retracements)} periods with >50% retracement")
```

### Fibonacci Signals

**Function:** `SmartMoneyConcepts.get_fib_signal(df)`

Generates trading signals based on price crossing over swingvalue with retracement confirmation.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with trend_bias, swingline, swingvalue, Close, CurrentRetracement columns

**Returns:**
- DataFrame with added fib_signal column

**Use Case Example:**
```python
# Generate Fibonacci signals
df = smc.get_fib_signal(df)

# Find recent signals
recent_signals = df[df['fib_signal'].notna()].tail(5)
print(f"Found {len(recent_signals)} recent Fibonacci signals")

# Analyze signal characteristics
for idx in recent_signals.index:
    signal_type = df.loc[idx, 'fib_signal']
    retracement = df.loc[idx, 'CurrentRetracement']
    swingvalue = df.loc[idx, 'swingvalue']
    
    if signal_type == 1:
        print(f"Bullish signal at {idx}:")
        print(f"  Retracement: {retracement:.1f}%")
        print(f"  Swing value: {swingvalue:.5f}")
        print(f"  Entry price: {df.loc[idx, 'Close']:.5f}")
    elif signal_type == -1:
        print(f"Bearish signal at {idx}:")
        print(f"  Retracement: {retracement:.1f}%")
        print(f"  Swing value: {swingvalue:.5f}")
        print(f"  Entry price: {df.loc[idx, 'Close']:.5f}")

# Trading strategy: Signal confirmation
latest_signal = df['fib_signal'].iloc[-1]
if pd.notna(latest_signal):
    if latest_signal == 1:
        print("Recent bullish Fibonacci signal - consider long position")
        # Set stop loss below swing value
        stop_loss = df['swingvalue'].iloc[-1]
        print(f"Suggested stop loss: {stop_loss:.5f}")
    elif latest_signal == -1:
        print("Recent bearish Fibonacci signal - consider short position")
        # Set stop loss above swing value
        stop_loss = df['swingvalue'].iloc[-1]
        print(f"Suggested stop loss: {stop_loss:.5f}")
```

### BOS Retest Signals

**Function:** `SmartMoneyConcepts.get_bos_retest_signals(df)`

Generates retest signals for broken structure levels.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with trend_bias, swingline, swingvalue, BOS, High, Low columns

**Returns:**
- DataFrame with added retest_signal column

**Use Case Example:**
```python
# Generate BOS retest signals
df = smc.get_bos_retest_signals(df)

# Find recent retest signals
recent_retests = df[df['retest_signal'].notna()].tail(5)
print(f"Found {len(recent_retests)} recent BOS retest signals")

# Analyze retest characteristics
for idx in recent_retests.index:
    retest_type = df.loc[idx, 'retest_signal']
    swingvalue = df.loc[idx, 'swingvalue']
    
    if retest_type == 1:
        print(f"Bullish retest at {idx}:")
        print(f"  Retesting level: {swingvalue:.5f}")
        print(f"  Current low: {df.loc[idx, 'Low']:.5f}")
        print(f"  Entry price: {df.loc[idx, 'Close']:.5f}")
    elif retest_type == -1:
        print(f"Bearish retest at {idx}:")
        print(f"  Retesting level: {swingvalue:.5f}")
        print(f"  Current high: {df.loc[idx, 'High']:.5f}")
        print(f"  Entry price: {df.loc[idx, 'Close']:.5f}")

# Trading strategy: Retest entries
latest_retest = df['retest_signal'].iloc[-1]
if pd.notna(latest_retest):
    if latest_retest == 1:
        print("Bullish retest signal - consider long entry")
        # Set stop loss below retest level
        stop_loss = df['swingvalue'].iloc[-1]
        print(f"Suggested stop loss: {stop_loss:.5f}")
    elif latest_retest == -1:
        print("Bearish retest signal - consider short entry")
        # Set stop loss above retest level
        stop_loss = df['swingvalue'].iloc[-1]
        print(f"Suggested stop loss: {stop_loss:.5f}")

# Monitor retest success rate
retest_signals = df[df['retest_signal'].notna()]
if len(retest_signals) > 0:
    print(f"Total retest signals: {len(retest_signals)}")
    # Analyze success rate based on subsequent price action
```

---

## Complete SMC Analysis Example

Here's a complete example of running all SMC indicators:

```python
from app.strategy.indicators import SmartMoneyConcepts
from app.data.mt5_client import MT5Client

# Initialize
mt5_client = MT5Client()
smc = SmartMoneyConcepts(mt5_client, "EURUSD")

# Fetch data
df = mt5_client.fetch_data("EURUSD", "M5", start_pos=0, end_pos=500)

# Run complete SMC analysis
df = smc.run_smc(df)

# Analyze results
print("=== SMC Analysis Results ===")

# Current market structure
current_swing = df['swingline'].iloc[-1]
current_trend = df['trend_bias'].iloc[-1]
current_price = df['Close'].iloc[-1]

print(f"Current swing direction: {'Bullish' if current_swing == 1 else 'Bearish'}")
print(f"Current trend bias: {'Bullish' if current_trend == 1 else 'Bearish'}")
print(f"Current price: {current_price:.5f}")

# Support and resistance
current_support = df['Support'].iloc[-1]
current_resistance = df['Resistance'].iloc[-1]

if pd.notna(current_support):
    print(f"Current support: {current_support:.5f}")
if pd.notna(current_resistance):
    print(f"Current resistance: {current_resistance:.5f}")

# Recent signals
recent_fib_signals = df[df['fib_signal'].notna()].tail(3)
recent_retest_signals = df[df['retest_signal'].notna()].tail(3)

print(f"\nRecent Fibonacci signals: {len(recent_fib_signals)}")
print(f"Recent retest signals: {len(recent_retest_signals)}")

# Trading opportunities
if len(recent_fib_signals) > 0:
    latest_fib = recent_fib_signals.iloc[-1]
    if latest_fib['fib_signal'] == 1:
        print("Recent bullish Fibonacci signal - consider long entry")
    elif latest_fib['fib_signal'] == -1:
        print("Recent bearish Fibonacci signal - consider short entry")

if len(recent_retest_signals) > 0:
    latest_retest = recent_retest_signals.iloc[-1]
    if latest_retest['retest_signal'] == 1:
        print("Recent bullish retest signal - consider long entry")
    elif latest_retest['retest_signal'] == -1:
        print("Recent bearish retest signal - consider short entry")

# Market conditions
current_retracement = df['CurrentRetracement'].iloc[-1]
if pd.notna(current_retracement):
    print(f"Current retracement: {current_retracement:.1f}%")

# Save analysis for further review
df.to_csv("smc_analysis.csv")
print("\nAnalysis saved to smc_analysis.csv")
```

---

## Best Practices

1. **Combine Multiple Indicators**: Use multiple indicators for confirmation rather than relying on single signals.

2. **Timeframe Alignment**: Ensure signals align across multiple timeframes for stronger confirmation.

3. **Risk Management**: Always use proper stop losses and position sizing based on ATR or ADR.

4. **Market Context**: Consider overall market conditions and news events when interpreting signals.

5. **Backtesting**: Always backtest strategies before live trading.

6. **Continuous Monitoring**: Regularly review and adjust indicator parameters based on market conditions.

7. **Documentation**: Keep detailed records of all trades and signal performance for analysis.

---

## Performance Considerations

- **Data Requirements**: SMC indicators require sufficient historical data for accurate calculations.
- **Computational Load**: Running all SMC indicators can be computationally intensive.
- **Real-time Updates**: Consider caching results for real-time applications.
- **Memory Usage**: Large datasets may require memory optimization.

---

This documentation provides a comprehensive guide to all available indicators in the HaruPyQuant system. Each indicator includes practical examples and trading strategies to help you implement effective trading systems. 