
def rearrange_trades_columns(trades_df):
    """
    Rearrange the trades DataFrame columns in the desired order.
    
    Args:
        trades_df: The trades DataFrame from backtesting results
        
    Returns:
        DataFrame with columns rearranged in the specified order:
        EntryTime, EntryPrice, ExitTime, ExitPrice, Size, Type, SL, TP, 
        PnLPips, PnL, ReturnPct, Commission, MAE, MFE, Duration, EntryBar, ExitBar, Tag
        
        Note: MAE and MFE are calculated in pips (not percentages)
    """
    # Define the desired column order
    desired_order = [
        'EntryTime',
        'EntryPrice', 
        'ExitTime',
        'ExitPrice',
        'Size',
        'Type',
        'SL',
        'TP',
        'PnLPips',
        'PnL',
        'ReturnPct',
        'Commission',
        'MAE',
        'MFE',
        'Duration',
        'EntryBar',
        'ExitBar',
        'Tag'
    ]
    
    # Get only the columns that exist in the DataFrame
    existing_columns = [col for col in desired_order if col in trades_df.columns]
    
    # Add any remaining columns that weren't in the desired order
    remaining_columns = [col for col in trades_df.columns if col not in existing_columns]
    final_order = existing_columns + remaining_columns
    
    return trades_df[final_order]
