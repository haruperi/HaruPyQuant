import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

dark_mode = True
if dark_mode:
    plt.style.use('dark_background')

def compute_ma(data, time_window):
    """Compute moving average for given data and window."""
    return data.rolling(window=time_window).mean()

def compute_rsi(data, time_window):
    """Compute RSI (Relative Strength Index) using Wilder's Smoothing Method."""
    diff = data.diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)

    # Using Wilder's Smoothing Method
    avg_gain = gain.rolling(window=time_window, min_periods=time_window).mean()[time_window-1:]
    avg_loss = loss.rolling(window=time_window, min_periods=time_window).mean()[time_window-1:]

    for i in range(len(avg_gain)):
        if i > 0:
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (time_window - 1) + gain.iloc[i + time_window - 1]) / time_window
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (time_window - 1) + loss.iloc[i + time_window - 1]) / time_window

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger(data, window, std):
    """Compute Bollinger Bands."""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std)
    lower_band = rolling_mean - (rolling_std * std)
    return upper_band, lower_band

def add_candlesticks(ax, df, color_map=None):
    """Add candlestick chart to the given axis."""
    for i in range(len(df)):
        close_price = df['Close'].iloc[i].item()
        open_price = df['Open'].iloc[i].item()
        
        if color_map is None:
            color = 'green' if close_price >= open_price else 'red'
        else:
            color = color_map(close_price, open_price)
            
        ax.plot([df.index[i], df.index[i]], [df['Low'].iloc[i], df['High'].iloc[i]], color=color, zorder=1)
        ax.plot([df.index[i], df.index[i]], [df['Open'].iloc[i], df['Close'].iloc[i]], color=color, linewidth=5, zorder=2)

def add_bollinger_bands(ax, df, window=20, std=2, color='blue', alpha=0.15):
    """Add Bollinger Bands to the chart."""
    upper_band, lower_band = compute_bollinger(df['Close'], window, std)
    # Robustly handle DataFrame or 2D output
    if hasattr(upper_band, 'ndim') and upper_band.ndim == 2:
        upper_band = upper_band.iloc[:, 0]
    if hasattr(lower_band, 'ndim') and lower_band.ndim == 2:
        lower_band = lower_band.iloc[:, 0]
    upper_band = pd.Series(upper_band, index=df.index).squeeze()
    lower_band = pd.Series(lower_band, index=df.index).squeeze()
    valid_mask = ~(upper_band.isna() | lower_band.isna())
    if valid_mask.any():
        valid_index = df.index[valid_mask]
        valid_upper = upper_band.loc[valid_mask]
        valid_lower = lower_band.loc[valid_mask]
        ax.fill_between(valid_index, valid_upper, valid_lower, color=color, alpha=alpha)
        ax.plot(valid_index, valid_upper, color=color, linewidth=1)
        ax.plot(valid_index, valid_lower, color=color, linewidth=1)

def add_moving_averages(ax, df, periods, colors=None):
    if colors is None:
        colors = ['orange', '#39D0FF', 'yellow', 'purple', 'cyan']
    for i, period in enumerate(periods):
        color = colors[i % len(colors)]
        ma_data = compute_ma(df['Close'], period)
        if hasattr(ma_data, 'ndim') and ma_data.ndim == 2:
            ma_data = ma_data.iloc[:, 0]
        ma_data = pd.Series(ma_data, index=df.index).squeeze()
        valid_mask = ~ma_data.isna()
        if valid_mask.any():
            valid_index = df.index[valid_mask]
            valid_ma = ma_data.loc[valid_mask]
            ax.plot(valid_index, valid_ma, color=color, linewidth=1, label=f'MA{period}')
    ax.legend(loc='upper left')

def add_volume_bars(ax, df, volume_scale, color_map=None):
    """Add volume bars to the chart."""
    for i in range(len(df)):
        close_price = df['Close'].iloc[i].item()
        open_price = df['Open'].iloc[i].item()
        
        if color_map is None:
            bar_color = 'green' if close_price >= open_price else 'red'
        else:
            bar_color = color_map(close_price, open_price)
            
        ax.bar(df.index[i], df['Volume'].iloc[i] * volume_scale, width=5, alpha=0.5, color=bar_color, zorder=0)

def add_rsi(ax, df, window=14, color='#FF86FF'):
    rsi_data = compute_rsi(df['Close'], window)
    if hasattr(rsi_data, 'ndim') and rsi_data.ndim == 2:
        rsi_data = rsi_data.iloc[:, 0]
    rsi_data = pd.Series(rsi_data, index=df.index).squeeze()
    valid_mask = ~rsi_data.isna()
    if valid_mask.any():
        valid_index = df.index[valid_mask]
        valid_rsi = rsi_data.loc[valid_mask]
        ax.plot(valid_index, valid_rsi, color=color, linewidth=1, label=f'RSI{window}')
        ax.fill_between(valid_index, 30, 70, alpha=0.15, color='#FFE2FF')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.set_yticks([10, 30, 50, 70, 90])

def add_earnings_dividends(ax, ticker, df, y_position, volume_scale):
    """Add earnings and dividend markers to the chart."""
    earnings_data = ticker.earnings_dates
    dividends_data = ticker.dividends

    # Filter earnings and dividend dates for the data period
    start_date = df.index.min().tz_localize(None)
    end_date = df.index.max().tz_localize(None)
    earnings_dates = earnings_data.loc[(earnings_data.index.tz_localize(None) >= start_date) & (earnings_data.index.tz_localize(None) <= end_date)].index
    dividend_dates = dividends_data.loc[(dividends_data.index.tz_localize(None) >= start_date) & (dividends_data.index.tz_localize(None) <= end_date)].index

    # Define ellipse parameters
    ellipse_width = 16
    max_volume = df['Volume'].max().item()
    ellipse_height = max_volume * volume_scale / 4.6

    # Plotting earnings 'E' markers
    for date in earnings_dates:
        try:
            color = 'red' if earnings_data.loc[date, 'Reported EPS'] < earnings_data.loc[date, 'EPS Estimate'] else ('green' if earnings_data.loc[date, 'Reported EPS'] >= earnings_data.loc[date, 'EPS Estimate'] else 'gray')
            ellipse = mpatches.Ellipse((date, y_position), width=ellipse_width, height=ellipse_height, color=color, alpha=.9, transform=ax.transData, zorder=5)
            ax.add_patch(ellipse)
            text_artist = ax.annotate(r'$\mathbf{E}$', xy=(date, y_position), color='white', fontsize=10, ha='center', va='center', zorder=6)
            ellipse.set_clip_box(ax.bbox)
            text_artist.set_clip_box(ax.bbox)
        except:
            print(f'Error plotting earnings marker.')

    # Plotting dividend 'D' markers
    for date in dividend_dates:
        ellipse = mpatches.Ellipse((date, y_position), width=ellipse_width, height=ellipse_height, color='orange', alpha=.9, transform=ax.transData, zorder=3)
        ax.add_patch(ellipse)
        text_artist = ax.annotate(r'$\mathbf{D}$', xy=(date, y_position), color='white', fontsize=10, ha='center', va='center', zorder=4)
        ellipse.set_clip_box(ax.bbox)
        text_artist.set_clip_box(ax.bbox)

def add_price_line(ax, df, offset_weeks=10):
    """Add horizontal line for the last close price."""
    last_close_price = df['Close'].iloc[-1].item()
    last_open_price = df['Open'].iloc[-1].item()
    line_color = 'green' if last_close_price >= last_open_price else 'red'

    # Add horizontal line
    ax.axhline(y=last_close_price, color=line_color, linestyle='--', linewidth=1)

    # Add text box
    text_box_properties = dict(facecolor=line_color, edgecolor=line_color)
    text_box_position = (df.index[-1] + pd.DateOffset(weeks=offset_weeks), last_close_price)
    ax.text(text_box_position[0], text_box_position[1], f' {round(last_close_price, 2)} ', 
             verticalalignment='bottom', horizontalalignment='right',
             color='white', bbox=text_box_properties, transform=ax.transData, zorder=6)

def setup_chart_style(fig, ax1, ax2, ax1_twin):
    """Setup chart styling and appearance."""
    # Move y-axis ticks to the right
    ax1.yaxis.tick_right()
    ax1_twin.yaxis.tick_right()
    ax2.yaxis.tick_right()

    # Remove subplot borders
    ax1.spines['right'].set_visible(False)
    ax1_twin.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    if dark_mode:
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')

def plot(symbol, indicators=None, min_candles=100):
    """
    Create a comprehensive chart with customizable indicators.
    
    Args:
        symbol (str): Stock symbol to plot
        indicators (dict): Dictionary of indicators to add
        min_candles (int): Minimum number of candles required
    """
    # Initialize variables
    ticker = yf.Ticker(symbol)
    
    # Default indicators if none provided
    if indicators is None:
        indicators = {
            'bollinger': {'window': 20, 'std': 2, 'enabled': True},
            'moving_averages': {'periods': [12, 26], 'enabled': True},
            'rsi': {'window': 14, 'enabled': True},
            'volume': {'enabled': True},
            'earnings_dividends': {'enabled': True},
            'price_line': {'enabled': True}
        }

    # Create subplots with different heights
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [2, 1]},
                                   sharex=True)

    # Create additional axis for volume bars
    ax1_twin = ax1.twinx()

    # Setup chart style
    setup_chart_style(fig, ax1, ax2, ax1_twin)

    # Download data
    df = yf.download(symbol, interval='1wk', period='max')
    if len(df) < min_candles:
        return False

    df_last_100 = df.tail(min_candles)

    # ===== MAIN CHART INDICATORS =====
    # Add candlesticks (always enabled)
    add_candlesticks(ax1, df_last_100)
    
    # Add Bollinger Bands
    if indicators.get('bollinger', {}).get('enabled', False):
        config = indicators['bollinger']
        add_bollinger_bands(ax1, df_last_100, config.get('window', 20), config.get('std', 2))
    
    # Add Moving Averages
    if indicators.get('moving_averages', {}).get('enabled', False):
        config = indicators['moving_averages']
        add_moving_averages(ax1, df_last_100, config.get('periods', [12, 26]))
    
    # Add Price Line
    if indicators.get('price_line', {}).get('enabled', False):
        add_price_line(ax1, df_last_100)

    # ===== VOLUME AND ANNOTATIONS =====
    # Calculate volume scale
    max_close = df_last_100['Close'].max().item()
    max_volume = df_last_100['Volume'].max().item()
    volume_scale = (max_close * 0.05) / max_volume

    # Add Volume Bars
    if indicators.get('volume', {}).get('enabled', False):
        add_volume_bars(ax1_twin, df_last_100, volume_scale)
        ax1_twin.set_ylim(0, max_close * 0.2)
        ax1_twin.set_yticks([])

    # Add Earnings and Dividends
    if indicators.get('earnings_dividends', {}).get('enabled', False):
        y_annotation_position_twin = (df_last_100['Volume'] * volume_scale).max().item() / 5
        add_earnings_dividends(ax1_twin, ticker, df_last_100, y_annotation_position_twin, volume_scale)

    # ===== SUBPLOT INDICATORS =====
    # Add RSI
    if indicators.get('rsi', {}).get('enabled', False):
        config = indicators['rsi']
        add_rsi(ax2, df_last_100, config.get('window', 14))

    # ===== FINAL CHART SETUP =====
    ax1.set_title(f"{ticker.info['longName']} ({symbol})", fontsize=16, pad=20)
    ax1.grid(True, alpha=.25)
    ax2.grid(True, alpha=.25)

    plt.tight_layout()
    plt.savefig(f"{symbol}_candlestick.png")
    plt.show()

def plot_with_explicit_calls(symbol, min_candles=100):
    """
    Alternative plotting function with explicit indicator calls that can be easily toggled.
    This makes it very easy to enable/disable indicators by commenting out the function calls.
    """
    # Initialize variables
    ticker = yf.Ticker(symbol)
    
    # Create subplots with different heights
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [2, 1]},
                                   sharex=True)

    # Create additional axis for volume bars
    ax1_twin = ax1.twinx()

    # Setup chart style
    setup_chart_style(fig, ax1, ax2, ax1_twin)

    # Download data
    df = yf.download(symbol, interval='1wk', period='max')
    if len(df) < min_candles:
        return False

    df_last_100 = df.tail(min_candles)

    # ===== MAIN CHART INDICATORS =====
    # Always add candlesticks
    add_candlesticks(ax1, df_last_100)
    
    # ===== TOGGLE THESE INDICATORS BY COMMENTING/UNCOMMENTING =====
    
    # Bollinger Bands
    add_bollinger_bands(ax1, df_last_100, window=20, std=2)
    
    # Moving Averages (can specify multiple periods)
    add_moving_averages(ax1, df_last_100, periods=[12, 26, 50])
    
    # Price Line
    add_price_line(ax1, df_last_100)

    # ===== VOLUME AND ANNOTATIONS =====
    # Calculate volume scale
    max_close = df_last_100['Close'].max().item()
    max_volume = df_last_100['Volume'].max().item()
    volume_scale = (max_close * 0.05) / max_volume

    # Volume Bars
    add_volume_bars(ax1_twin, df_last_100, volume_scale)
    ax1_twin.set_ylim(0, max_close * 0.2)
    ax1_twin.set_yticks([])

    # Earnings and Dividends
    y_annotation_position_twin = (df_last_100['Volume'] * volume_scale).max().item() / 5
    add_earnings_dividends(ax1_twin, ticker, df_last_100, y_annotation_position_twin, volume_scale)

    # ===== SUBPLOT INDICATORS =====
    # RSI
    add_rsi(ax2, df_last_100, window=14)

    # ===== FINAL CHART SETUP =====
    ax1.set_title(f"{ticker.info['longName']} ({symbol})", fontsize=16, pad=20)
    ax1.grid(True, alpha=.25)
    ax2.grid(True, alpha=.25)

    plt.tight_layout()
    plt.savefig(f"{symbol}_candlestick.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    symbol = "KO"
    
    # Method 1: Using configuration dictionary (original method)
    custom_indicators = {
        'bollinger': {'window': 20, 'std': 2, 'enabled': True},
        'moving_averages': {'periods': [12, 26, 50], 'enabled': True},
        'rsi': {'window': 14, 'enabled': True},
        'volume': {'enabled': True},
        'earnings_dividends': {'enabled': True},
        'price_line': {'enabled': True}
    }
    
    # Uncomment to use the original method
    # plot(symbol, custom_indicators)
    
    # Method 2: Using explicit function calls (new method)
    # This makes it very easy to toggle indicators by commenting out lines
    plot_with_explicit_calls(symbol)
