# Modular Chart System Documentation

## Overview

The chart system has been refactored to be highly modular and flexible. You can now easily toggle indicators on/off by either:
1. **Configuration Method**: Using a dictionary to enable/disable indicators
2. **Explicit Calls Method**: Commenting/uncommenting function calls directly in the code

## Two Methods Available

### Method 1: Configuration Dictionary

Use the `plot()` function with an indicators configuration dictionary:

```python
from app.data.charts import plot

# Define which indicators to enable
indicators = {
    'bollinger': {'window': 20, 'std': 2, 'enabled': True},
    'moving_averages': {'periods': [12, 26], 'enabled': True},
    'rsi': {'window': 14, 'enabled': False},  # RSI disabled
    'volume': {'enabled': True},
    'earnings_dividends': {'enabled': False},
    'price_line': {'enabled': True}
}

plot("KO", indicators)
```

### Method 2: Explicit Function Calls

Use the `plot_with_explicit_calls()` function and modify it directly:

```python
from app.data.charts import plot_with_explicit_calls

# Edit the function in app/data/charts.py to comment/uncomment lines:
def plot_with_explicit_calls(symbol, min_candles=100):
    # ... setup code ...
    
    # ===== TOGGLE THESE INDICATORS BY COMMENTING/UNCOMMENTING =====
    
    # Bollinger Bands
    add_bollinger_bands(ax1, df_last_100, window=20, std=2)
    
    # Moving Averages (can specify multiple periods)
    # add_moving_averages(ax1, df_last_100, periods=[12, 26, 50])  # Commented out
    
    # Price Line
    add_price_line(ax1, df_last_100)
    
    # ... rest of the function ...
```

## Available Indicators

### 1. Bollinger Bands
- **Function**: `add_bollinger_bands(ax, df, window=20, std=2, color='blue', alpha=0.15)`
- **Parameters**:
  - `window`: Period for moving average (default: 20)
  - `std`: Standard deviation multiplier (default: 2)
  - `color`: Band color (default: 'blue')
  - `alpha`: Transparency (default: 0.15)

### 2. Moving Averages
- **Function**: `add_moving_averages(ax, df, periods, colors=None)`
- **Parameters**:
  - `periods`: List of periods (e.g., [12, 26, 50])
  - `colors`: List of colors (optional, auto-assigned if None)

### 3. RSI (Relative Strength Index)
- **Function**: `add_rsi(ax, df, window=14, color='#FF86FF')`
- **Parameters**:
  - `window`: RSI period (default: 14)
  - `color`: RSI line color (default: '#FF86FF')

### 4. Volume Bars
- **Function**: `add_volume_bars(ax, df, volume_scale, color_map=None)`
- **Parameters**:
  - `volume_scale`: Scaling factor for volume display
  - `color_map`: Custom color function (optional)

### 5. Earnings and Dividends
- **Function**: `add_earnings_dividends(ax, ticker, df, y_position, volume_scale)`
- **Parameters**:
  - `ticker`: yfinance ticker object
  - `y_position`: Vertical position for markers
  - `volume_scale`: Scaling factor

### 6. Price Line
- **Function**: `add_price_line(ax, df, offset_weeks=10)`
- **Parameters**:
  - `offset_weeks`: Horizontal offset for price label (default: 10)

## Common Chart Configurations

### Minimal Chart (Candlesticks Only)
```python
indicators = {
    'bollinger': {'enabled': False},
    'moving_averages': {'enabled': False},
    'rsi': {'enabled': False},
    'volume': {'enabled': False},
    'earnings_dividends': {'enabled': False},
    'price_line': {'enabled': False}
}
```

### Trend Analysis Chart
```python
indicators = {
    'bollinger': {'enabled': False},
    'moving_averages': {'periods': [12, 26, 50], 'enabled': True},
    'rsi': {'enabled': False},
    'volume': {'enabled': True},
    'earnings_dividends': {'enabled': False},
    'price_line': {'enabled': True}
}
```

### Volatility Analysis Chart
```python
indicators = {
    'bollinger': {'window': 20, 'std': 2, 'enabled': True},
    'moving_averages': {'enabled': False},
    'rsi': {'enabled': False},
    'volume': {'enabled': True},
    'earnings_dividends': {'enabled': False},
    'price_line': {'enabled': False}
}
```

### Momentum Analysis Chart
```python
indicators = {
    'bollinger': {'enabled': False},
    'moving_averages': {'enabled': False},
    'rsi': {'window': 14, 'enabled': True},
    'volume': {'enabled': True},
    'earnings_dividends': {'enabled': False},
    'price_line': {'enabled': False}
}
```

## Adding New Indicators

### Step 1: Create the Indicator Function
```python
def add_new_indicator(ax, df, param1=10, param2=20, color='red'):
    """
    Add a new custom indicator to the chart.
    
    Args:
        ax: matplotlib axis
        df: DataFrame with OHLCV data
        param1: First parameter (default: 10)
        param2: Second parameter (default: 20)
        color: Line color (default: 'red')
    """
    # Calculate your indicator
    indicator_data = calculate_indicator(df['Close'], param1, param2)
    
    # Handle 2D/1D data robustly
    if hasattr(indicator_data, 'ndim') and indicator_data.ndim == 2:
        indicator_data = indicator_data.iloc[:, 0]
    indicator_data = pd.Series(indicator_data, index=df.index).squeeze()
    
    # Remove NaN values for plotting
    valid_mask = ~indicator_data.isna()
    if valid_mask.any():
        valid_index = df.index[valid_mask]
        valid_data = indicator_data.loc[valid_mask]
        ax.plot(valid_index, valid_data, color=color, linewidth=1, label=f'Indicator{param1}')
    
    ax.legend(loc='upper left')
```

### Step 2: Add to Configuration Method
Add to the indicators dictionary:
```python
indicators = {
    # ... existing indicators ...
    'new_indicator': {'param1': 10, 'param2': 20, 'enabled': True}
}
```

### Step 3: Add to Explicit Calls Method
Add the function call to `plot_with_explicit_calls()`:
```python
# New Indicator
add_new_indicator(ax1, df_last_100, param1=10, param2=20)
```

## Example Scripts

### Basic Usage
```python
from app.data.charts import plot, plot_with_explicit_calls

# Method 1: Configuration
plot("KO", indicators)

# Method 2: Explicit calls
plot_with_explicit_calls("KO")
```

### Advanced Usage
```python
from scripts.chart_configurations import (
    create_minimal_chart,
    create_trend_analysis_chart,
    create_volatility_analysis_chart,
    create_momentum_analysis_chart,
    create_comprehensive_chart,
    create_custom_chart
)

# Create different chart types
create_minimal_chart("KO")
create_trend_analysis_chart("KO")
create_volatility_analysis_chart("KO")
create_momentum_analysis_chart("KO")
create_comprehensive_chart("KO")
create_custom_chart("KO", ['bollinger', 'rsi'])
```

## Benefits of This System

1. **Easy Toggle**: Comment/uncomment one line to enable/disable an indicator
2. **Parameter Control**: Easily modify indicator parameters
3. **Extensible**: Add new indicators without changing existing code
4. **Flexible**: Use either configuration or explicit calls based on preference
5. **Robust**: Handles 1D/2D data automatically
6. **Maintainable**: Clear separation of concerns

## Best Practices

1. **Always handle NaN values** in new indicator functions
2. **Use robust data handling** with `.squeeze()` and `.loc[]`
3. **Provide sensible defaults** for all parameters
4. **Add proper documentation** for new indicator functions
5. **Test with different data shapes** to ensure robustness
6. **Use consistent naming conventions** for indicator functions

## Troubleshooting

### Common Issues

1. **"Data must be 1-dimensional" error**: Use `.squeeze()` and check for 2D data
2. **"not 1-dimensional" error**: Use `.loc[valid_mask]` instead of `[valid_mask]`
3. **Shape mismatch**: Ensure all data has the same index
4. **NaN values**: Always filter out NaN values before plotting

### Debug Tips

1. Print data shapes before plotting: `print(f"Shape: {data.shape}")`
2. Check for NaN values: `print(f"NaN count: {data.isna().sum()}")`
3. Verify index alignment: `print(f"Index match: {data.index.equals(df.index)}")` 