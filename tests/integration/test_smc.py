import os
import sys
import webbrowser
import pandas as pd
from datetime import datetime, timedelta, UTC
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time


# Add project root to the Python path   
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.data.mt5_client import MT5Client
from app.config.constants import DEFAULT_CONFIG_PATH, ALL_SYMBOLS
from app.strategy.indicators import SmartMoneyConcepts

def draw_diagonal_trendlines(df, fig):
    """
    Draw diagonal trendlines connecting swing points.
    
    Args:
        df (pd.DataFrame): DataFrame with swing points
        fig (plotly.graph_objects.Figure): The figure to add trendlines to
    """
    if 'swingpoint' not in df.columns:
        print("Warning: swingpoint column not found. Skipping diagonal trendlines.")
        return fig
    
    # Get column mapping for OHLC data
    column_mapping = {}
    for col in ['high', 'low']:
        if col in df.columns:
            column_mapping[col] = col
        elif col.capitalize() in df.columns:
            column_mapping[col] = col.capitalize()
        else:
            print(f"Warning: Required column '{col}' not found. Skipping diagonal trendlines.")
            return fig
    
    # Get swing points
    swing_points = df[df['swingpoint'].notna()].copy()
    
    if len(swing_points) < 2:
        print("Not enough swing points for diagonal trendlines.")
        return fig
    
    # Sort by index to ensure chronological order of swing points
    swing_points = swing_points.sort_index()
    
    # Find consecutive swing points for trendline drawing
    # We'll connect every 2-3 swing points to create meaningful trendlines
    swing_indices = swing_points.index.tolist()
    
    # Draw trendlines connecting consecutive swing points
    for i in range(len(swing_indices) - 1):
        idx1 = swing_indices[i]
        idx2 = swing_indices[i + 1]
        
        swing_point1 = swing_points.loc[idx1]
        swing_point2 = swing_points.loc[idx2]
        
        # Determine the price values to connect based on swing point type
        if swing_point1['swingpoint'] == 1:  # Bearish swing point (high)   
            price1 = swing_point1[column_mapping['high']]
        else:  # Bullish swing point (low)
            price1 = swing_point1[column_mapping['low']]
            
        if swing_point2['swingpoint'] == 1:  # Bearish swing point (high)
            price2 = swing_point2[column_mapping['high']]
        else:  # Bullish swing point (low)
            price2 = swing_point2[column_mapping['low']]
        
        # Determine line color based on trend direction
        if price2 > price1:
            line_color = "green"
            line_name = f"Uptrend {i+1}"
        else:
            line_color = "red"
            line_name = f"Downtrend {i+1}"
        
        # Add the trendline
        fig.add_trace(
            go.Scatter(
                x=[idx1, idx2],
                y=[price1, price2],
                mode='lines',
                line=dict(
                    color=line_color,
                    width=2,
                    dash='dot'
                ),
                name=line_name,
                showlegend=False,
                hovertemplate=f'<b>{line_name}</b><br>' +
                            f'Start: {price1:.5f}<br>' +
                            f'End: {price2:.5f}<extra></extra>'
            )
        )
    
    return fig

def draw_trendlines(df, fig):
    """
    Draw trendlines using Support and Resistance levels.
    
    Args:
        df (pd.DataFrame): DataFrame with Support and Resistance columns
        fig (plotly.graph_objects.Figure): The figure to add trendlines to
    """
    if 'Support' not in df.columns or 'Resistance' not in df.columns:
        print("Warning: Support or Resistance columns not found. Skipping trendlines.")
        return fig
    
    # Get unique support and resistance levels
    support_levels = df['Support'].dropna().unique()
    resistance_levels = df['Resistance'].dropna().unique()
    
    print(f"Found {len(support_levels)} unique support levels and {len(resistance_levels)} unique resistance levels")
    
    # Draw support trendlines
    for i, level in enumerate(support_levels):
        # Find all points where this support level is active
        support_points = df[df['Support'] == level]
        
        if len(support_points) > 1:
            # Create a horizontal line for this support level
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                opacity=0.7,
                line_width=2,
                annotation_text=f"Support {i+1}: {level:.5f}",
                annotation_position="bottom right"
            )
    
    # Draw resistance trendlines
    for i, level in enumerate(resistance_levels):
        # Find all points where this resistance level is active
        resistance_points = df[df['Resistance'] == level]
        
        if len(resistance_points) > 1:
            # Create a horizontal line for this resistance level
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                line_width=2,
                annotation_text=f"Resistance {i+1}: {level:.5f}",
                annotation_position="top right"
            )
    
    return fig

def plot_candlestick_with_smc_analysis(df, symbol="GBPUSD", timeframe="M5"):
    """
    Create a candlestick chart with swing points and market structure labels.
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data, swing point information, and structure_label
        symbol (str): Trading symbol
        timeframe (str): Timeframe of the data
    """
    # Check for required columns and handle different naming conventions
    required_columns = ['open', 'high', 'low', 'close']
    column_mapping = {}
    for col in required_columns:
        if col in df.columns:
            column_mapping[col] = col
        elif col.capitalize() in df.columns:
            column_mapping[col] = col.capitalize()
        else:
            print(f"Warning: Required column '{col}' not found in DataFrame")
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Required column '{col}' not found. Available columns: {df.columns.tolist()}")
    fig = go.Figure()
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df[column_mapping['open']],
            high=df[column_mapping['high']],
            low=df[column_mapping['low']],
            close=df[column_mapping['close']],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    )
    # Add swing points
    swing_points = df[df['swingpoint'].notna()].copy()
    if not swing_points.empty:
        # Bullish swing points (lows)
        bullish_swing_points = swing_points[swing_points['swingpoint'] == -1]
        if not bullish_swing_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=bullish_swing_points.index,
                    y=bullish_swing_points[column_mapping['low']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',  # Downward triangle for lows
                        size=14,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    name='Swing Lows',
                    legendgroup='swing_points',
                    showlegend=True,
                    hovertemplate='<b>Swing Low</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Low: %{y:.5f}<extra></extra>'
                )
            )
        # Bearish swing points (highs)
        bearish_swing_points = swing_points[swing_points['swingpoint'] == 1]
        if not bearish_swing_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=bearish_swing_points.index,
                    y=bearish_swing_points[column_mapping['high']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',  # Upward triangle for highs
                        size=14,
                        color='blue',
                        line=dict(color='darkblue', width=1)
                    ),
                    name='Swing Highs',
                    legendgroup='swing_points',
                    showlegend=True,
                    hovertemplate='<b>Swing High</b><br>' +
                                  'Time: %{x}<br>' +
                                  'High: %{y:.5f}<extra></extra>'
                )
            )
    # Stepwise Resistance (blue)
    if 'Resistance' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Resistance'],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Resistance',
                line_shape='hv',  # horizontal-vertical (stepwise)
                connectgaps=False,
                showlegend=True
            )
        )
    # Stepwise Support (red)
    if 'Support' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Support'],
                mode='lines',
                line=dict(color='red', width=2),
                name='Support',
                line_shape='hv',
                connectgaps=False,
                showlegend=True
            )
        )

    
    # Add order blocks visualization
    # Visualize bullish order blocks
    bullish_blocks = df[df['Bullish_Order_Block_Top'].notna()]
    bearish_blocks = df[df['Bearish_Order_Block_Top'].notna()]
    for idx, row in bullish_blocks.iterrows():
        top = row['Bullish_Order_Block_Top']
        bottom = row['Bullish_Order_Block_Bottom']
        mitigated = row['Bullish_Order_Block_Mitigated']
        color = 'rgba(0, 255, 0, 1)' if mitigated == 0 else 'rgba(0, 255, 0, 0.5)'
        border_color = 'green' if mitigated == 0 else 'darkgreen'
        fig.add_shape(
            type="rect",
            x0=idx,
            x1=idx,
            y0=bottom,
            y1=top,
            fillcolor=color,
            line=dict(color=border_color, width=2),
            layer="below"
        )
        ob_text = f"Bullish OB{' - Mitigated' if mitigated == 1 else ''}"
        fig.add_annotation(
            x=idx,
            y=top + (top - bottom) * 0.1,
            text=ob_text,
            showarrow=False,
            font=dict(size=10, color=border_color),
            bgcolor="white",
            bordercolor=border_color,
            borderwidth=1
        )
    # Visualize bearish order blocks
    for idx, row in bearish_blocks.iterrows():
        top = row['Bearish_Order_Block_Top']
        bottom = row['Bearish_Order_Block_Bottom']
        mitigated = row['Bearish_Order_Block_Mitigated']
        color = 'rgba(255, 0, 0, 1)' if mitigated == 0 else 'rgba(255, 0, 0, 0.5)'
        border_color = 'red' if mitigated == 0 else 'darkred'
        fig.add_shape(
            type="rect",
            x0=idx,
            x1=idx,
            y0=bottom,
            y1=top,
            fillcolor=color,
            line=dict(color=border_color, width=2),
            layer="below"
        )
        ob_text = f"Bearish OB{' - Mitigated' if mitigated == 1 else ''}"
        fig.add_annotation(
            x=idx,
            y=top + (top - bottom) * 0.1,
            text=ob_text,
            showarrow=False,
            font=dict(size=10, color=border_color),
            bgcolor="white",
            bordercolor=border_color,
            borderwidth=1
        )
    
    # Add BOS visualization
    bos_events = df[df['BOS'].notna()].copy()
    if not bos_events.empty:
        fig.add_trace(
            go.Scatter(
                x=bos_events.index,
                y=bos_events[column_mapping['close']],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color=bos_events['BOS'].map({1: 'blue', -1: 'red'}),
                    line=dict(color='black', width=1)
                ),
                name='Break of Structure',
                hovertemplate='<b>BOS</b><br>' +
                            'Type: %{customdata}<br>' +
                            'Price: %{y:.5f}<br>' +
                            'Time: %{x}<extra></extra>',
                customdata=bos_events['BOS'].map({1: 'Bullish', -1: 'Bearish'})
            )
        )
    
    # Add CHoCH visualization
    choch_events = df[df['CHoCH'].notna()].copy()
    if not choch_events.empty:
        fig.add_trace(
            go.Scatter(
                x=choch_events.index,
                y=choch_events[column_mapping['close']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=18,
                    color=choch_events['CHoCH'].map({1: 'purple', -1: 'brown'}),
                    line=dict(color='black', width=2)
                ),
                name='Change of Character',
                hovertemplate='<b>CHoCH</b><br>' +
                            'Type: %{customdata}<br>' +
                            'Price: %{y:.5f}<br>' +
                            'Time: %{x}<extra></extra>',
                customdata=choch_events['CHoCH'].map({1: 'Bullish', -1: 'Bearish'})
            )
        )
    
    fig.update_layout(
        title=f'{symbol} {timeframe} - SMC Analysis',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    return fig

if __name__ == '__main__':
    # Initialize MT5 client
    client = MT5Client()
    symbol = "EURGBP"
    timeframe = "M5"
    num_candles = 1440 # 1 Week

    # Set start and end of the specified day
    date_input = "2025-07-10"
    input_date = datetime.strptime(date_input, "%Y-%m-%d").date()
    start_of_day = datetime.combine(input_date, time.min)  # 00:00:00
    end_of_day = datetime.combine(input_date, time.max)    # 23:59:59.999999


    # # Get current date at midnight (beginning of the day)
    # current_date = datetime.now().date()
    # start_of_day = datetime.combine(current_date, time.min)  # 00:00:00
    # end_of_day = datetime.now()  # Current time

    # Fetch data
    #df = client.fetch_data(symbol, timeframe, start_pos=0, end_pos=num_candles)
    #df = client.fetch_data(symbol, timeframe, start_date=datetime(2025, 7, 1), end_date=datetime(2025, 7, 3))
    df = client.fetch_data(symbol, timeframe, start_date=start_of_day, end_date=end_of_day) # Fetch data for current day only

    print(f"Fetched {len(df)} candles for {symbol} {timeframe}")
    
    # Debug: Check DataFrame columns
    print(f"Original DataFrame columns: {df.columns.tolist()}")
    print(f"Original DataFrame head:\n{df.head()}")
    
    # Initialize SMC
    smc = SmartMoneyConcepts(symbol, min_swing_length=3, min_pip_range=2)
    # Run SMC analysis
    df_smc = smc.run_smc(df)

    # Display comprehensive statistics
    print("\n" + "="*50)
    print("COMPREHENSIVE SMC ANALYSIS STATISTICS")
    print("="*50)
    
    # Swing points statistics
    swing_points = df_smc[df_smc['swingpoint'].notna()]
    print(f"\n📊 SWING POINTS:")
    print(f"   Total Swing Points: {len(swing_points)}")
    if not swing_points.empty:
        bullish_swing_points = swing_points[swing_points['swingpoint'] == -1]
        bearish_swing_points = swing_points[swing_points['swingpoint'] == 1]
        print(f"   Bullish Swing Points: {len(bullish_swing_points)}")
        print(f"   Bearish Swing Points: {len(bearish_swing_points)}")
        if not bullish_swing_points.empty:
            print(f"   Bullish Swing Point Range: {bullish_swing_points['Low'].min():.5f} - {bullish_swing_points['Low'].max():.5f}")
        if not bearish_swing_points.empty:
            print(f"   Bearish Swing Point Range: {bearish_swing_points['High'].min():.5f} - {bearish_swing_points['High'].max():.5f}")
    
    # Support/Resistance statistics
    support_levels = df_smc['Support'].dropna().unique()
    resistance_levels = df_smc['Resistance'].dropna().unique()
    print(f"\n🏗️  SUPPORT/RESISTANCE LEVELS:")
    print(f"   Support Levels: {len(support_levels)}")
    print(f"   Resistance Levels: {len(resistance_levels)}")
    if len(support_levels) > 0:
        print(f"   Support Range: {support_levels.min():.5f} - {support_levels.max():.5f}")
    if len(resistance_levels) > 0:
        print(f"   Resistance Range: {resistance_levels.min():.5f} - {resistance_levels.max():.5f}")
    
    # BOS statistics
    bos_events = df_smc[df_smc['BOS'].notna()]
    print(f"\n🚀 BREAK OF STRUCTURE (BOS):")
    print(f"   Total BOS Events: {len(bos_events)}")
    if not bos_events.empty:
        bullish_bos = bos_events[bos_events['BOS'] == 1]
        bearish_bos = bos_events[bos_events['BOS'] == -1]
        print(f"   Bullish BOS: {len(bullish_bos)}")
        print(f"   Bearish BOS: {len(bearish_bos)}")
        
        # Show BOS details
        print(f"\n   BOS Event Details:")
        for idx, row in bos_events.head(3).iterrows():
            bos_type = "Bullish" if row['BOS'] == 1 else "Bearish"
            print(f"     {idx}: {bos_type} BOS")
    
    # CHoCH statistics
    choch_events = df_smc[df_smc['CHoCH'].notna()]
    print(f"\n🔄 CHANGE OF CHARACTER (CHoCH):")
    print(f"   Total CHoCH Events: {len(choch_events)}")
    if not choch_events.empty:
        bullish_choch = choch_events[choch_events['CHoCH'] == 1]
        bearish_choch = choch_events[choch_events['CHoCH'] == -1]
        print(f"   Bullish CHoCH: {len(bullish_choch)}")
        print(f"   Bearish CHoCH: {len(bearish_choch)}")
        
        # Show CHoCH details
        print(f"\n   CHoCH Event Details:")
        for idx, row in choch_events.head(3).iterrows():
            choch_type = "Bullish" if row['CHoCH'] == 1 else "Bearish"
            print(f"     {idx}: {choch_type} CHoCH")
    
    # Order blocks statistics
    bullish_blocks = df_smc[df_smc['Bullish_Order_Block_Top'].notna()]
    bearish_blocks = df_smc[df_smc['Bearish_Order_Block_Top'].notna()]
    print(f"\n📦 ORDER BLOCKS:")
    print(f"   Bullish Order Block Bars: {len(bullish_blocks)}")
    print(f"   Bearish Order Block Bars: {len(bearish_blocks)}")
    if not bullish_blocks.empty:
        mitigated_bullish = bullish_blocks[bullish_blocks['Bullish_Order_Block_Mitigated'] == 1]
        print(f"   Mitigated Bullish Order Block Bars: {len(mitigated_bullish)}")
        print(f"   Active Bullish Order Block Bars: {len(bullish_blocks) - len(mitigated_bullish)}")
        print(f"\n   Bullish Order Block Details:")
        for idx, row in bullish_blocks.head(3).iterrows():
            mitigated = "Yes" if row['Bullish_Order_Block_Mitigated'] == 1 else "No"
            print(f"     {idx}: Top: {row['Bullish_Order_Block_Top']:.5f}, Bottom: {row['Bullish_Order_Block_Bottom']:.5f}, Mitigated: {mitigated}")
    if not bearish_blocks.empty:
        mitigated_bearish = bearish_blocks[bearish_blocks['Bearish_Order_Block_Mitigated'] == 1]
        print(f"   Mitigated Bearish Order Block Bars: {len(mitigated_bearish)}")
        print(f"   Active Bearish Order Block Bars: {len(bearish_blocks) - len(mitigated_bearish)}")
        print(f"\n   Bearish Order Block Details:")
        for idx, row in bearish_blocks.head(3).iterrows():
            mitigated = "Yes" if row['Bearish_Order_Block_Mitigated'] == 1 else "No"
            print(f"     {idx}: Top: {row['Bearish_Order_Block_Top']:.5f}, Bottom: {row['Bearish_Order_Block_Bottom']:.5f}, Mitigated: {mitigated}")
    
    # Market structure statistics
    swing_changes = df_smc['swingline'].diff().fillna(0)
    swing_changes_count = (swing_changes != 0).sum()
    print(f"\n📈 MARKET STRUCTURE:")
    print(f"   Swing Direction Changes: {swing_changes_count}")
    print(f"   Final Swing Direction: {'Uptrend' if df_smc['swingline'].iloc[-1] == 1 else 'Downtrend'}")
    
    # Price range statistics
    print(f"\n💰 PRICE RANGE:")
    print(f"   High: {df_smc['High'].max():.5f}")
    print(f"   Low: {df_smc['Low'].min():.5f}")
    print(f"   Range: {df_smc['High'].max() - df_smc['Low'].min():.5f}")
    print(f"   Current Close: {df_smc['Close'].iloc[-1]:.5f}")
    
    print("\n" + "="*50)
    
    # Debug: Check SMC DataFrame columns
    print(f"SMC DataFrame columns: {df_smc.columns.tolist()}")
    print(f"SMC DataFrame head:\n{df_smc.head()}")
  
    # Create and display the chart
    #print("\nCreating a graphic display of the SMC analysis...")
    #fig = plot_candlestick_with_smc_analysis(df_smc, symbol, timeframe)

    # Save the chart
    # chart_filename = f"smc_analysis.html"
    # fig.write_html(chart_filename)
    # print(f"Chart saved as: {chart_filename}")

    # Save results
    df_smc.to_csv(f"smc_analysis.csv")
    print(f"Data saved to smc_analysis.csv")

    # Display sample of results
    print("\nSample of SMC analysis results:")
    print(df_smc.tail(10))

    # Open the chart in the browser
    # if os.path.exists(chart_filename):
    #     webbrowser.open(f"file://{os.path.abspath(chart_filename)}")
    #     print(f"Opening {chart_filename} in browser...")
    # else:
    #     print(f"Chart file {chart_filename} not found. Please run the SMC test script first.") 


    


