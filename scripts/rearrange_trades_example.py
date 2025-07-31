#!/usr/bin/env python3
"""
Example script demonstrating how to rearrange trades DataFrame columns.
"""

import pandas as pd
from app.util.helper import rearrange_trades_columns

def main():
    """
    Example of how to use the rearrange_trades_columns function.
    """
    # Create a sample trades DataFrame (similar to what you'd get from backtesting)
    sample_trades = pd.DataFrame({
        'Size': [1000, -2000, 1500],
        'EntryBar': [10, 25, 40],
        'ExitBar': [15, 30, 45],
        'EntryPrice': [1.2000, 1.2100, 1.2050],
        'ExitPrice': [1.2050, 1.2080, 1.2070],
        'Type': ['Buy', 'Sell', 'Buy'],  # Trade type based on direction
        'SL': [1.1950, 1.2150, 1.2000],
        'TP': [1.2100, 1.2050, 1.2150],
        'PnL': [50.0, -40.0, 30.0],
        'ReturnPct': [0.0042, -0.0033, 0.0017],
        'PnLPips': [5.0, -4.0, 2.0],
        'Commission': [2.4, 4.8, 3.6],  # Commission in dollars
        'MAE': [-5.0, -8.0, -3.0],  # Maximum Adverse Excursion in pips
        'MFE': [12.0, 6.0, 9.0],    # Maximum Favorable Excursion in pips
        'EntryTime': ['2025-01-01 10:00:00', '2025-01-01 14:00:00', '2025-01-01 18:00:00'],
        'ExitTime': ['2025-01-01 10:30:00', '2025-01-01 14:30:00', '2025-01-01 18:30:00'],
        'Duration': ['0 days 00:30:00', '0 days 00:30:00', '0 days 00:30:00'],
        'Tag': [None, None, None]
    })
    
    print("Original trades DataFrame:")
    print(sample_trades)
    print("\n" + "="*80 + "\n")
    
    # Rearrange the columns in the desired order
    rearranged_trades = rearrange_trades_columns(sample_trades)
    
    print("Rearranged trades DataFrame:")
    print(rearranged_trades)
    print("\n" + "="*80 + "\n")
    
    print("Column order in rearranged DataFrame:")
    for i, col in enumerate(rearranged_trades.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Show trade analysis by type
    print("\n" + "="*80 + "\n")
    print("Trade Analysis by Type:")
    
    # Buy trades analysis
    buy_trades = rearranged_trades[rearranged_trades['Type'] == 'Buy']
    sell_trades = rearranged_trades[rearranged_trades['Type'] == 'Sell']
    
    print(f"Buy Trades ({len(buy_trades)}):")
    if len(buy_trades) > 0:
        print(f"  Average PnL: {buy_trades['PnLPips'].mean():.1f} pips")
        print(f"  Average Commission: ${buy_trades['Commission'].mean():.2f}")
        print(f"  Win Rate: {(buy_trades['PnLPips'] > 0).mean() * 100:.1f}%")
    
    print(f"\nSell Trades ({len(sell_trades)}):")
    if len(sell_trades) > 0:
        print(f"  Average PnL: {sell_trades['PnLPips'].mean():.1f} pips")
        print(f"  Average Commission: ${sell_trades['Commission'].mean():.2f}")
        print(f"  Win Rate: {(sell_trades['PnLPips'] > 0).mean() * 100:.1f}%")
    
    # Individual trade analysis
    print("\n" + "="*80 + "\n")
    print("Individual Trade Analysis:")
    for i, trade in rearranged_trades.iterrows():
        print(f"Trade {i+1} ({trade['Type']}):")
        print(f"  PnL: {trade['PnLPips']:.1f} pips")
        print(f"  Commission: ${trade['Commission']:.2f}")
        print(f"  Net PnL: {trade['PnLPips']:.1f} pips - ${trade['Commission']:.2f}")
        print(f"  MAE: {trade['MAE']:.1f} pips (worst drawdown)")
        print(f"  MFE: {trade['MFE']:.1f} pips (best run)")
        risk_reward = abs(trade['MAE'] / trade['MFE']) if trade['MFE'] != 0 else float('inf')
        print(f"  Risk/Reward: {risk_reward:.2f}")
        print()
    
    # You can also save the rearranged DataFrame to CSV
    # rearranged_trades.to_csv('rearranged_trades.csv', index=False)
    # print("\nRearranged trades saved to 'rearranged_trades.csv'")

if __name__ == "__main__":
    main()