import sys
import os
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.data.mt5_client import MT5Client
from app.config.constants import (
    DEFAULT_TIMEFRAME, 
    DEFAULT_SYMBOL, 
    DEFAULT_START_CANDLE, 
    DEFAULT_END_CANDLE,
    CHART_BACKGROUND_COLOR,
    BULLISH_CANDLE_COLOR,
    BEARISH_CANDLE_COLOR
)

def create_candlestick_chart(df: pd.DataFrame, symbol: str, timeframe: str):
    """
    Creates a candlestick chart from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data and a 'time' index.
        symbol (str): The symbol for the chart title.
        timeframe (str): The timeframe for the chart title.
    """
    df["date"] = df.index
    
    inc = df.Close > df.Open
    dec = df.Open > df.Close
    w = 12 * 60 * 60 * 1000  # half day in ms

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, sizing_mode="stretch_both",
               title=f"{symbol} Candlestick Chart ({timeframe})",
               x_axis_label='Date', y_axis_label='Price')
    p.xaxis.major_label_orientation = 3.14 / 4
    p.grid.grid_line_alpha = 0.3
    p.background_fill_color = CHART_BACKGROUND_COLOR
    p.border_fill_color = CHART_BACKGROUND_COLOR
    p.title.text_color = "white"
    p.xaxis.axis_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"
    p.xaxis.major_tick_line_color = "white"
    p.yaxis.major_tick_line_color = "white"
    p.xaxis.minor_tick_line_color = "white"
    p.yaxis.minor_tick_line_color = "white"

    # Plot the wicks
    p.segment(df.date, df.High, df.date, df.Low, color="white")

    # Plot the bars
    p.vbar(df.date[inc], w, df.Open[inc], df.Close[inc], fill_color=BULLISH_CANDLE_COLOR, line_color=BULLISH_CANDLE_COLOR)
    p.vbar(df.date[dec], w, df.Open[dec], df.Close[dec], fill_color=BEARISH_CANDLE_COLOR, line_color=BEARISH_CANDLE_COLOR)

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ('Date', '@date{%F %T}'),
            ('Open', '@Open{0,0.00000}'),
            ('High', '@High{0,0.00000}'),
            ('Low', '@Low{0,0.00000}'),
            ('Close', '@Close{0,0.00000}'),
            ('Volume', '@Volume{0.00 a}')
        ],
        formatters={
            '@date': 'datetime',
        },
        mode='vline'
    )
    p.add_tools(hover)
    
    # Save the plot to an HTML file
    output_path = os.path.join(project_root, "candlestick_chart.html")
    output_file(output_path)
    show(p)
    print(f"Candlestick chart saved to {output_path}")

def main():
    """
    Main function to fetch data and create the chart.
    """
    print("Connecting to MT5...")
    try:
        with MT5Client() as mt5_client:
            if not mt5_client.is_connected():
                print("Failed to connect to MT5. Exiting.")
                return

            print("Fetching data...")
            # Fetch 100 bars of EURUSD H1 data
            data = mt5_client.fetch_data(
                symbol=DEFAULT_SYMBOL,
                timeframe="D1",
                start_pos=DEFAULT_START_CANDLE,
                end_pos=DEFAULT_END_CANDLE
            )

            if data is not None and not data.empty:
                print(f"Successfully fetched {len(data)} records.")
                # Create and show the chart
                create_candlestick_chart(data, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME)
            else:
                print("No data fetched. Cannot create chart.")

    except FileNotFoundError:
        print("Error: config.ini not found. Please ensure it is in the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
