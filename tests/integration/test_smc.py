import os
import sys
import pandas as pd
from datetime import datetime, timedelta, UTC
import numpy as np

# Add project root to the Python path   
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.data.mt5_client import MT5Client
from app.config.constants import DEFAULT_CONFIG_PATH, ALL_SYMBOLS
from app.strategy.indicators import SmartMoneyConcepts

if __name__ == '__main__':
    # Initialize MT5 client
    client = MT5Client()
    
    # Fetch data
    df = client.fetch_data("GBPUSD", timeframe="M5", start_pos=0, end_pos=1000)
    print(df)

 
    
    smc = SmartMoneyConcepts("GBPUSD", min_swing_length=3, min_pip_range=2)
    df_smc = smc.calculate_swingline(df)
    df_smc = smc.calculate_pivot_points(df_smc)

    # Check Fair Value Gap (FVG)
    df_smc = smc.identify_fair_value_gaps(df_smc)

    
    # # Perform complete SMC analysis
    # print("Performing SMC analysis...")
    # df_smc = smc.analyze_smc(df)
    
    # # Display results
    # print("\nSMC Analysis Results:")
    # print(f"Total candles: {len(df_smc)}")
    # print(f"Swingline values: {df_smc['swingline'].value_counts().to_dict()}")
    
    # Get pivot points
    #pivot_points = df_smc[df_smc['isPivot'].notna()]
    #print(f"Pivot points: {pivot_points}")
    
    # # Count order blocks
    # order_block_counts = df_smc['order_block_type'].value_counts().dropna()
    # print(f"Order blocks: {order_block_counts.to_dict()}")
    
    # # Count fair value gaps
    # fvg_counts = df_smc['fvg_type'].value_counts().dropna()
    # print(f"Fair value gaps: {fvg_counts.to_dict()}")
    
    # # Count liquidity zones
    # liquidity_counts = df_smc['liquidity_zone_type'].value_counts().dropna()
    # print(f"Liquidity zones: {liquidity_counts.to_dict()}")
    
    # Save results
    df_smc.to_csv("smc_test.csv")
    print(f"\nResults saved to smc_test.csv")
    
    # Display sample of results
    print("\nSample of SMC analysis results:")
    print(df_smc)
    # sample_columns = ['Open', 'High', 'Low', 'Close', 'swingline', 'isPivot', 
    #                  'pivot_type', 'order_block_type', 'fvg_type', 'liquidity_zone_type']
    # available_columns = [col for col in sample_columns if col in df_smc.columns]
    # print(df_smc[available_columns].tail(10))