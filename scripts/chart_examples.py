#!/usr/bin/env python3
"""
Chart Examples - Demonstrating the modular chart system
This script shows how to easily toggle indicators by commenting/uncommenting function calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.charts import plot_with_explicit_calls, plot

def example_minimal_chart():
    """Example with only candlesticks - no indicators"""
    print("Creating minimal chart with only candlesticks...")
    
    # Copy the plot_with_explicit_calls function and comment out all indicators
    # This would be done by editing the function directly
    
def example_bollinger_only():
    """Example with only Bollinger Bands"""
    print("Creating chart with only Bollinger Bands...")
    
    # In plot_with_explicit_calls, comment out everything except:
    # - add_candlesticks(ax1, df_last_100)
    # - add_bollinger_bands(ax1, df_last_100, window=20, std=2)
    # - Final chart setup

def example_ma_only():
    """Example with only Moving Averages"""
    print("Creating chart with only Moving Averages...")
    
    # In plot_with_explicit_calls, comment out everything except:
    # - add_candlesticks(ax1, df_last_100)
    # - add_moving_averages(ax1, df_last_100, periods=[12, 26, 50])
    # - Final chart setup

def example_rsi_only():
    """Example with only RSI"""
    print("Creating chart with only RSI...")
    
    # In plot_with_explicit_calls, comment out everything except:
    # - add_candlesticks(ax1, df_last_100)
    # - add_rsi(ax2, df_last_100, window=14)
    # - Final chart setup

def example_configuration_method():
    """Example using the configuration dictionary method"""
    print("Creating chart using configuration method...")
    
    # Define which indicators to enable
    indicators = {
        'bollinger': {'window': 20, 'std': 2, 'enabled': True},
        'moving_averages': {'periods': [12, 26], 'enabled': True},
        'rsi': {'window': 14, 'enabled': False},  # RSI disabled
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': False},  # Earnings/Dividends disabled
        'price_line': {'enabled': True}
    }
    
    plot("KO", indicators)

if __name__ == "__main__":
    print("Chart Examples - Modular Indicator System")
    print("=" * 50)
    
    # Example 1: Using configuration method
    print("\n1. Configuration Method Example:")
    example_configuration_method()
    
    # Example 2: Using explicit calls (you would modify the function directly)
    print("\n2. Explicit Calls Method:")
    print("To use this method, edit the plot_with_explicit_calls function")
    print("and comment/uncomment the indicator function calls as needed.")
    
    print("\nExample modifications you can make:")
    print("- Comment out 'add_bollinger_bands(...)' to disable Bollinger Bands")
    print("- Comment out 'add_moving_averages(...)' to disable Moving Averages")
    print("- Comment out 'add_rsi(...)' to disable RSI")
    print("- Comment out 'add_volume_bars(...)' to disable Volume")
    print("- Comment out 'add_earnings_dividends(...)' to disable Earnings/Dividends")
    print("- Comment out 'add_price_line(...)' to disable Price Line")
    
    print("\nThe beauty of this system is that you can easily:")
    print("- Toggle any indicator by commenting/uncommenting one line")
    print("- Add new indicators by adding new function calls")
    print("- Customize parameters directly in the function calls")
    print("- See exactly which indicators are active in your code") 