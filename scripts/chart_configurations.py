#!/usr/bin/env python3
"""
Chart Configurations - Practical examples of different chart setups
This script shows how to create different chart configurations by modifying the plot_with_explicit_calls function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.charts import plot

def create_minimal_chart(symbol="KO"):
    """
    Minimal chart with only candlesticks.
    To achieve this, in plot_with_explicit_calls, comment out all indicator calls except:
    - add_candlesticks(ax1, df_last_100)
    - Final chart setup
    """
    print(f"Creating minimal chart for {symbol}...")
    
    # Use configuration method with all indicators disabled
    indicators = {
        'bollinger': {'enabled': False},
        'moving_averages': {'enabled': False},
        'rsi': {'enabled': False},
        'volume': {'enabled': False},
        'earnings_dividends': {'enabled': False},
        'price_line': {'enabled': False}
    }
    
    plot(symbol, indicators)

def create_trend_analysis_chart(symbol="KO"):
    """
    Chart optimized for trend analysis with moving averages.
    To achieve this, in plot_with_explicit_calls, enable:
    - add_candlesticks(ax1, df_last_100)
    - add_moving_averages(ax1, df_last_100, periods=[12, 26, 50])
    - add_price_line(ax1, df_last_100)
    """
    print(f"Creating trend analysis chart for {symbol}...")
    
    indicators = {
        'bollinger': {'enabled': False},
        'moving_averages': {'periods': [12, 26, 50], 'enabled': True},
        'rsi': {'enabled': False},
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': False},
        'price_line': {'enabled': True}
    }
    
    plot(symbol, indicators)

def create_volatility_analysis_chart(symbol="KO"):
    """
    Chart optimized for volatility analysis with Bollinger Bands.
    To achieve this, in plot_with_explicit_calls, enable:
    - add_candlesticks(ax1, df_last_100)
    - add_bollinger_bands(ax1, df_last_100, window=20, std=2)
    - add_volume_bars(ax1_twin, df_last_100, volume_scale)
    """
    print(f"Creating volatility analysis chart for {symbol}...")
    
    indicators = {
        'bollinger': {'window': 20, 'std': 2, 'enabled': True},
        'moving_averages': {'enabled': False},
        'rsi': {'enabled': False},
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': False},
        'price_line': {'enabled': False}
    }
    
    plot(symbol, indicators)

def create_momentum_analysis_chart(symbol="KO"):
    """
    Chart optimized for momentum analysis with RSI.
    To achieve this, in plot_with_explicit_calls, enable:
    - add_candlesticks(ax1, df_last_100)
    - add_rsi(ax2, df_last_100, window=14)
    - add_volume_bars(ax1_twin, df_last_100, volume_scale)
    """
    print(f"Creating momentum analysis chart for {symbol}...")
    
    indicators = {
        'bollinger': {'enabled': False},
        'moving_averages': {'enabled': False},
        'rsi': {'window': 14, 'enabled': True},
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': False},
        'price_line': {'enabled': False}
    }
    
    plot(symbol, indicators)

def create_comprehensive_chart(symbol="KO"):
    """
    Comprehensive chart with all indicators.
    To achieve this, in plot_with_explicit_calls, enable all indicators.
    """
    print(f"Creating comprehensive chart for {symbol}...")
    
    indicators = {
        'bollinger': {'window': 20, 'std': 2, 'enabled': True},
        'moving_averages': {'periods': [12, 26, 50], 'enabled': True},
        'rsi': {'window': 14, 'enabled': True},
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': True},
        'price_line': {'enabled': True}
    }
    
    plot(symbol, indicators)

def create_custom_chart(symbol="KO", enabled_indicators=None):
    """
    Custom chart with specific indicators enabled.
    
    Args:
        symbol (str): Stock symbol
        enabled_indicators (list): List of indicator names to enable
    """
    if enabled_indicators is None:
        enabled_indicators = ['bollinger', 'moving_averages']
    
    print(f"Creating custom chart for {symbol} with indicators: {enabled_indicators}")
    
    # Default configuration
    indicators = {
        'bollinger': {'window': 20, 'std': 2, 'enabled': False},
        'moving_averages': {'periods': [12, 26], 'enabled': False},
        'rsi': {'window': 14, 'enabled': False},
        'volume': {'enabled': False},
        'earnings_dividends': {'enabled': False},
        'price_line': {'enabled': False}
    }
    
    # Enable only the specified indicators
    for indicator in enabled_indicators:
        if indicator in indicators:
            indicators[indicator]['enabled'] = True
    
    plot(symbol, indicators)

if __name__ == "__main__":
    print("Chart Configurations - Different Analysis Types")
    print("=" * 60)
    
    symbol = "KO"
    
    # Example 1: Minimal chart
    print("\n1. Minimal Chart (Candlesticks only)")
    create_minimal_chart(symbol)
    
    # Example 2: Trend analysis
    print("\n2. Trend Analysis Chart")
    create_trend_analysis_chart(symbol)
    
    # Example 3: Volatility analysis
    print("\n3. Volatility Analysis Chart")
    create_volatility_analysis_chart(symbol)
    
    # Example 4: Momentum analysis
    print("\n4. Momentum Analysis Chart")
    create_momentum_analysis_chart(symbol)
    
    # Example 5: Comprehensive analysis
    print("\n5. Comprehensive Chart")
    create_comprehensive_chart(symbol)
    
    # Example 6: Custom configuration
    print("\n6. Custom Chart (Bollinger + RSI only)")
    create_custom_chart(symbol, ['bollinger', 'rsi'])
    
    print("\n" + "=" * 60)
    print("How to use the explicit calls method:")
    print("1. Open app/data/charts.py")
    print("2. Find the plot_with_explicit_calls function")
    print("3. Comment/uncomment the indicator function calls as needed")
    print("4. Save and run the script")
    
    print("\nExample modifications:")
    print("# Bollinger Bands")
    print("add_bollinger_bands(ax1, df_last_100, window=20, std=2)")
    print("# Moving Averages")
    print("# add_moving_averages(ax1, df_last_100, periods=[12, 26, 50])  # Commented out")
    print("# RSI")
    print("# add_rsi(ax2, df_last_100, window=14)  # Commented out") 