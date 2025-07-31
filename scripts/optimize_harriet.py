#!/usr/bin/env python3
"""
Harriet Strategy Optimization Script
Allows optimization with different profit maximization metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from app.data.mt5_client import MT5Client
from app.backtesting import Backtest
from app.strategy.harriet import HarrietHedgingStrategy, get_data, generate_signals
from app.config.setup import *

def optimize_harriet_strategy(maximize_metric='Return [%]', show_top_n=10):
    """
    Optimize Harriet strategy with specified profit maximization metric
    
    Args:
        maximize_metric (str): Metric to maximize ('Return [%]', 'Equity Final [$]', 'Profit Factor', etc.)
        show_top_n (int): Number of top results to display
    """
    
    # Configuration
    symbol = "EURUSD"
    lt_interval = "M5"
    ht_interval = "H1"
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 7, 28)
    
    # Strategy parameters
    ht_min_dist = 5  # In pips
    lt_min_dist = 2  # In pips
    
    print(f"=== HARRIET STRATEGY OPTIMIZATION ===")
    print(f"Maximizing: {maximize_metric}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframes: {lt_interval} / {ht_interval}")
    print()
    
    # Initialize MT5 client
    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)
    symbol_info = mt5_client.get_symbol_info(symbol)
    point = symbol_info.point
    
    # Fetch and prepare data
    print("Fetching data...")
    data = get_data(mt5_client, symbol, lt_interval, ht_interval, start_date, end_date)
    
    if data is None or data.empty:
        print("No data available for optimization")
        return
    
    # Generate signals
    print("Generating signals...")
    data_with_signals = generate_signals(data, ht_min_dist, lt_min_dist, point)
    
    # Create backtest
    bt = Backtest(
        data_with_signals,
        HarrietHedgingStrategy,
        cash=10000,
        commission=.0002,
        exclusive_orders=True
    )
    
    # Define parameter ranges
    take_profit_range = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    stop_loss_range = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    print(f"Optimizing {len(take_profit_range) * len(stop_loss_range)} parameter combinations...")
    
    # Run optimization
    optimized_bt, heatmap = bt.optimize(
        take_profit_pips=take_profit_range,
        stop_loss_pips=stop_loss_range,
        return_heatmap=True,
        maximize=maximize_metric
    )
    
    # Display results
    print(f"\n=== OPTIMIZATION RESULTS (Maximizing: {maximize_metric}) ===")
    print("Best Parameters:")
    print(f"  Take Profit Pips: {optimized_bt['_strategy'].take_profit_pips}")
    print(f"  Stop Loss Pips: {optimized_bt['_strategy'].stop_loss_pips}")
    
    print(f"\nBest Performance Metrics:")
    print(f"  Final Equity: ${optimized_bt['Equity Final [$]']:.2f}")
    print(f"  Return: {optimized_bt['Return [%]']:.2f}%")
    print(f"  Total Trades: {optimized_bt['# Trades']}")
    print(f"  Win Rate: {optimized_bt['Win Rate [%]']:.2f}%")
    print(f"  Profit Factor: {optimized_bt['Profit Factor']:.2f}")
    print(f"  Max Drawdown: {optimized_bt['Max. Drawdown [%]']:.2f}%")
    print(f"  Sharpe Ratio: {optimized_bt['Sharpe Ratio']:.2f}")
    
    # Show top N results
    print(f"\n=== TOP {show_top_n} PARAMETER COMBINATIONS ===")
    sorted_results = heatmap.sort_values(ascending=False)
    print(f"Rank | Take Profit | Stop Loss | {maximize_metric}")
    print("-" * 50)
    
    for i, (params, score) in enumerate(sorted_results.head(show_top_n).items(), 1):
        tp, sl = params
        print(f"{i:4d} | {tp:11d} | {sl:9d} | {score:.4f}")
    
    print(f"\nTotal combinations tested: {len(heatmap)}")
    
    return optimized_bt, heatmap

def main():
    """Main function to run different optimization scenarios"""
    
    print("Harriet Strategy Profit Optimization")
    print("=" * 50)
    
    # Available optimization metrics
    metrics = {
        '1': ('Return [%]', 'Return Percentage'),
        '2': ('Equity Final [$]', 'Absolute Profit'),
        '3': ('Profit Factor', 'Profit Factor'),
        '4': ('Sharpe Ratio', 'Risk-Adjusted Return'),
        '5': ('SQN', 'System Quality Number')
    }
    
    print("\nAvailable optimization metrics:")
    for key, (metric, description) in metrics.items():
        print(f"  {key}. {description} ({metric})")
    
    choice = input("\nSelect optimization metric (1-5, default=1): ").strip() or '1'
    
    if choice not in metrics:
        print("Invalid choice, using Return [%]")
        choice = '1'
    
    metric, description = metrics[choice]
    print(f"\nOptimizing for: {description}")
    
    # Run optimization
    optimized_bt, heatmap = optimize_harriet_strategy(maximize_metric=metric)
    
    # Ask if user wants to see detailed results for best parameters
    show_details = input("\nShow detailed results for best parameters? (y/n): ").strip().lower()
    if show_details == 'y':
        print("\n" + "="*60)
        print("DETAILED RESULTS FOR BEST PARAMETERS")
        print("="*60)
        print(optimized_bt)

if __name__ == "__main__":
    main() 