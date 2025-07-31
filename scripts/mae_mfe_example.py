#!/usr/bin/env python3
"""
Example script demonstrating MAE and MFE calculations in pips.
"""

import pandas as pd
import numpy as np

def calculate_mae_mfe_pips(entry_price, exit_price, entry_bar, exit_bar, is_long, ohlc_data):
    """
    Calculate MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) in pips.
    
    Args:
        entry_price: Entry price of the trade
        exit_price: Exit price of the trade
        entry_bar: Entry bar index
        exit_bar: Exit bar index
        is_long: True if long trade, False if short trade
        ohlc_data: DataFrame with OHLC data
        
    Returns:
        tuple: (MAE, MFE) in pips
    """
    # Get price data during the trade period
    trade_highs = ohlc_data['High'].values[entry_bar:exit_bar + 1]
    trade_lows = ohlc_data['Low'].values[entry_bar:exit_bar + 1]
    
    if is_long:
        # Long trade: MFE is highest high, MAE is lowest low
        mfe = (trade_highs.max() - entry_price) * 10000  # Convert to pips
        mae = (trade_lows.min() - entry_price) * 10000  # Convert to pips
    else:
        # Short trade: MFE is lowest low, MAE is highest high
        mfe = (entry_price - trade_lows.min()) * 10000  # Convert to pips
        mae = (entry_price - trade_highs.max()) * 10000  # Convert to pips
    
    return mae, mfe

def main():
    """
    Example of MAE and MFE calculations in pips.
    """
    # Create sample OHLC data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=50, freq='H')
    
    # Create sample price data with some volatility
    base_price = 1.2000
    prices = []
    for i in range(50):
        # Add some random movement
        change = np.random.normal(0, 0.001)
        base_price += change
        high = base_price + abs(np.random.normal(0, 0.0005))
        low = base_price - abs(np.random.normal(0, 0.0005))
        close = base_price + np.random.normal(0, 0.0002)
        open_price = base_price + np.random.normal(0, 0.0002)
        
        prices.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 5000)
        })
    
    ohlc_data = pd.DataFrame(prices, index=dates)
    
    print("Sample OHLC Data (first 10 rows):")
    print(ohlc_data.head(10))
    print("\n" + "="*80 + "\n")
    
    # Example trades
    trades = [
        {
            'entry_price': 1.2000,
            'exit_price': 1.2050,
            'entry_bar': 10,
            'exit_bar': 15,
            'is_long': True,
            'size': 1000
        },
        {
            'entry_price': 1.2100,
            'exit_price': 1.2080,
            'entry_bar': 25,
            'exit_bar': 30,
            'is_long': False,
            'size': -2000
        }
    ]
    
    print("Calculating MAE and MFE in pips for sample trades:")
    print("-" * 60)
    
    for i, trade in enumerate(trades, 1):
        mae, mfe = calculate_mae_mfe_pips(
            trade['entry_price'],
            trade['exit_price'],
            trade['entry_bar'],
            trade['exit_bar'],
            trade['is_long'],
            ohlc_data
        )
        
        trade_type = "LONG" if trade['is_long'] else "SHORT"
        pnl_pips = (trade['exit_price'] - trade['entry_price']) * 10000
        if not trade['is_long']:
            pnl_pips = -pnl_pips
        
        print(f"Trade {i} ({trade_type}):")
        print(f"  Entry Price: {trade['entry_price']:.4f}")
        print(f"  Exit Price:  {trade['exit_price']:.4f}")
        print(f"  PnL Pips:    {pnl_pips:.1f}")
        print(f"  MAE Pips:    {mae:.1f} (worst point)")
        print(f"  MFE Pips:    {mfe:.1f} (best point)")
        print(f"  MAE/MFE:     {abs(mae/mfe):.2f} (risk/reward ratio)")
        print()
    
    # Create a sample trades DataFrame with MAE and MFE in pips
    sample_trades = pd.DataFrame({
        'Size': [1000, -2000],
        'EntryBar': [10, 25],
        'ExitBar': [15, 30],
        'EntryPrice': [1.2000, 1.2100],
        'ExitPrice': [1.2050, 1.2080],
        'Type': ['Buy', 'Sell'],  # Trade type based on direction
        'PnL': [50.0, -40.0],
        'ReturnPct': [0.0042, -0.0033],
        'PnLPips': [50.0, -20.0],
        'Commission': [2.4, 4.8],  # Commission in dollars
        'MAE': [-5.0, -8.0],  # Maximum Adverse Excursion in pips
        'MFE': [12.0, 6.0],   # Maximum Favorable Excursion in pips
        'EntryTime': ['2025-01-01 10:00:00', '2025-01-01 14:00:00'],
        'ExitTime': ['2025-01-01 10:30:00', '2025-01-01 14:30:00'],
        'Duration': ['0 days 00:30:00', '0 days 00:30:00'],
        'Tag': [None, None]
    })
    
    print("Sample Trades DataFrame with MAE and MFE in pips:")
    print(sample_trades)
    print("\n" + "="*80 + "\n")
    
    # Show some statistics
    print("Trade Analysis (in pips):")
    print(f"Average MAE: {sample_trades['MAE'].mean():.1f} pips")
    print(f"Average MFE: {sample_trades['MFE'].mean():.1f} pips")
    print(f"MAE/MFE Ratio: {abs(sample_trades['MAE'].mean() / sample_trades['MFE'].mean()):.2f}")
    print(f"Total Commission: ${sample_trades['Commission'].sum():.2f}")
    
    # Analysis by trade type
    buy_trades = sample_trades[sample_trades['Type'] == 'Buy']
    sell_trades = sample_trades[sample_trades['Type'] == 'Sell']
    
    print(f"\nBuy Trades ({len(buy_trades)}):")
    if len(buy_trades) > 0:
        print(f"  Average PnL: {buy_trades['PnLPips'].mean():.1f} pips")
        print(f"  Average MAE: {buy_trades['MAE'].mean():.1f} pips")
        print(f"  Average MFE: {buy_trades['MFE'].mean():.1f} pips")
    
    print(f"\nSell Trades ({len(sell_trades)}):")
    if len(sell_trades) > 0:
        print(f"  Average PnL: {sell_trades['PnLPips'].mean():.1f} pips")
        print(f"  Average MAE: {sell_trades['MAE'].mean():.1f} pips")
        print(f"  Average MFE: {sell_trades['MFE'].mean():.1f} pips")
    
    # Risk analysis
    print("\nRisk Analysis:")
    for i, trade in sample_trades.iterrows():
        risk_reward = abs(trade['MAE'] / trade['MFE']) if trade['MFE'] != 0 else float('inf')
        print(f"Trade {i+1} ({trade['Type']}): Risk/Reward = {risk_reward:.2f}")
    
    # Pip-based analysis
    print("\nPip-based Analysis:")
    for i, trade in sample_trades.iterrows():
        print(f"Trade {i+1} ({trade['Type']}):")
        print(f"  PnL: {trade['PnLPips']:.1f} pips")
        print(f"  Commission: ${trade['Commission']:.2f}")
        print(f"  Net PnL: {trade['PnLPips']:.1f} pips - ${trade['Commission']:.2f}")
        print(f"  MAE: {trade['MAE']:.1f} pips (worst drawdown)")
        print(f"  MFE: {trade['MFE']:.1f} pips (best run)")
        print(f"  Risk/Reward: {abs(trade['MAE'] / trade['MFE']):.2f}")
        print()

if __name__ == "__main__":
    main()