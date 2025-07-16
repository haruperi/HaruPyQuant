# Real-time Data Streaming

This document describes the real-time data streaming functionality in HaruPyQuant, which provides live market data feeds for trading strategies and analysis.

## Overview

The streaming system provides real-time access to:
- **Tick Data**: Live bid/ask prices and trade information
- **OHLCV Data**: Real-time candlestick data for technical analysis
- **WebSocket Data**: External data sources via WebSocket connections
- **Fundamental Data**: Economic news and events

## Architecture

### Core Components

1. **StreamingManager**: Central manager for all data streams
2. **DataStreamHandler**: Base class for all stream handlers
3. **MT5TickStreamHandler**: Handles MT5 tick data streaming
4. **MT5OHLCVStreamHandler**: Handles MT5 OHLCV data streaming
5. **WebSocketStreamHandler**: Handles external WebSocket data

### Data Structures

- **TickData**: Contains bid, ask, last price, volume, and spread
- **OHLCVData**: Contains open, high, low, close, volume, and timeframe
- **StreamConfig**: Configuration for stream behavior

## Quick Start

### Basic Usage

```python
from app.data import StreamingManager, MT5Client
from app.config.setup import DEFAULT_CONFIG_PATH

# Initialize MT5 client
mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH)

# Create streaming manager
streaming_manager = StreamingManager(mt5_client)

# Create tick stream
tick_stream_id = streaming_manager.create_tick_stream(
    symbols=["EURUSD", "GBPUSD"],
    callback=lambda data: print(f"Tick: {data.symbol} Bid: {data.bid} Ask: {data.ask}")
)

# Start the stream
streaming_manager.start_stream(tick_stream_id)

# Stop when done
streaming_manager.stop_stream(tick_stream_id)
```

### OHLCV Streaming

```python
# Create OHLCV stream
ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
    symbols=["EURUSD"],
    timeframe="H1",
    callback=lambda data: print(f"Candle: {data.symbol} Close: {data.close}")
)

# Start the stream
streaming_manager.start_stream(ohlcv_stream_id)
```

## Detailed Usage

### Stream Configuration

```python
from app.data import StreamConfig, StreamType

# Configure a stream
config = StreamConfig(
    stream_type=StreamType.TICK,
    symbols=["EURUSD", "GBPUSD"],
    callback=my_callback_function,
    buffer_size=1000,
    reconnect_interval=5,
    max_reconnect_attempts=10,
    enable_logging=True
)
```

### Custom Callbacks

```python
def my_tick_handler(tick_data):
    """Handle incoming tick data."""
    print(f"New tick for {tick_data.symbol}:")
    print(f"  Bid: {tick_data.bid}")
    print(f"  Ask: {tick_data.ask}")
    print(f"  Spread: {tick_data.spread}")
    print(f"  Volume: {tick_data.volume}")

def my_ohlcv_handler(ohlcv_data):
    """Handle incoming OHLCV data."""
    print(f"New candle for {ohlcv_data.symbol} {ohlcv_data.timeframe}:")
    print(f"  Open: {ohlcv_data.open}")
    print(f"  High: {ohlcv_data.high}")
    print(f"  Low: {ohlcv_data.low}")
    print(f"  Close: {ohlcv_data.close}")
    print(f"  Volume: {ohlcv_data.volume}")

# Create streams with custom callbacks
tick_stream_id = streaming_manager.create_tick_stream(
    symbols=["EURUSD"],
    callback=my_tick_handler
)

ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
    symbols=["EURUSD"],
    timeframe="H1",
    callback=my_ohlcv_handler
)
```

### Multiple Streams

```python
# Create multiple streams
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
stream_ids = []

for symbol in symbols:
    # Tick stream
    tick_id = streaming_manager.create_tick_stream([symbol])
    stream_ids.append(tick_id)
    
    # OHLCV stream
    ohlcv_id = streaming_manager.create_ohlcv_stream([symbol], "H1")
    stream_ids.append(ohlcv_id)

# Start all streams
streaming_manager.start_all_streams()

# Monitor status
status = streaming_manager.get_stream_status()
print(f"Active streams: {len([s for s in status.values() if s['running']])}")

# Stop all streams
streaming_manager.stop_all_streams()
```

### Data Processing Example

```python
class TradingDataProcessor:
    def __init__(self, symbols):
        self.symbols = symbols
        self.tick_buffer = {}
        self.ohlcv_buffer = {}
        
        for symbol in symbols:
            self.tick_buffer[symbol] = []
            self.ohlcv_buffer[symbol] = []
    
    def process_tick(self, tick_data):
        """Process incoming tick data."""
        symbol = tick_data.symbol
        
        # Add to buffer
        self.tick_buffer[symbol].append(tick_data)
        
        # Keep only last 100 ticks
        if len(self.tick_buffer[symbol]) > 100:
            self.tick_buffer[symbol].pop(0)
        
        # Analyze for trading opportunities
        self._analyze_tick_data(tick_data)
    
    def process_ohlcv(self, ohlcv_data):
        """Process incoming OHLCV data."""
        symbol = ohlcv_data.symbol
        
        # Add to buffer
        self.ohlcv_buffer[symbol].append(ohlcv_data)
        
        # Keep only last 50 candles
        if len(self.ohlcv_buffer[symbol]) > 50:
            self.ohlcv_buffer[symbol].pop(0)
        
        # Analyze for trading signals
        self._analyze_ohlcv_data(ohlcv_data)
    
    def _analyze_tick_data(self, tick_data):
        """Analyze tick data for opportunities."""
        # Example: Monitor spread
        if tick_data.spread > 0.0005:  # 5 pips
            print(f"High spread warning: {tick_data.symbol} = {tick_data.spread}")
    
    def _analyze_ohlcv_data(self, ohlcv_data):
        """Analyze OHLCV data for signals."""
        # Example: Simple moving average
        symbol = ohlcv_data.symbol
        if len(self.ohlcv_buffer[symbol]) >= 20:
            sma = sum(c.close for c in self.ohlcv_buffer[symbol][-20:]) / 20
            
            if ohlcv_data.close > sma:
                print(f"BUY signal: {symbol} above SMA")
            elif ohlcv_data.close < sma:
                print(f"SELL signal: {symbol} below SMA")

# Usage
processor = TradingDataProcessor(["EURUSD", "GBPUSD"])

tick_stream_id = streaming_manager.create_tick_stream(
    symbols=["EURUSD", "GBPUSD"],
    callback=processor.process_tick
)

ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
    symbols=["EURUSD", "GBPUSD"],
    timeframe="H1",
    callback=processor.process_ohlcv
)
```

## WebSocket Streaming

### External Data Sources

```python
# Create WebSocket stream
websocket_stream_id = streaming_manager.create_websocket_stream(
    url="wss://example.com/stream",
    callback=lambda data: print(f"WebSocket data: {data}")
)

# Start the stream
streaming_manager.start_stream(websocket_stream_id)
```

### Custom WebSocket Handler

```python
class CustomWebSocketHandler:
    def __init__(self, url):
        self.url = url
        self.websocket = None
    
    def on_message(self, data):
        """Handle incoming WebSocket messages."""
        try:
            # Parse JSON data
            parsed_data = json.loads(data)
            
            # Process the data
            self.process_data(parsed_data)
            
        except json.JSONDecodeError:
            print(f"Invalid JSON: {data}")
    
    def process_data(self, data):
        """Process parsed WebSocket data."""
        # Implement your data processing logic here
        print(f"Processing: {data}")

# Usage
handler = CustomWebSocketHandler("wss://example.com/stream")
websocket_stream_id = streaming_manager.create_websocket_stream(
    url="wss://example.com/stream",
    callback=handler.on_message
)
```

## Error Handling

### Connection Issues

The streaming system automatically handles:
- Connection timeouts
- Reconnection attempts
- Data validation errors
- MT5 connection failures

### Custom Error Handling

```python
def robust_callback(data):
    """Callback with error handling."""
    try:
        # Process the data
        process_data(data)
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        # Implement fallback logic here

# Use the robust callback
streaming_manager.create_tick_stream(
    symbols=["EURUSD"],
    callback=robust_callback
)
```

## Performance Considerations

### Buffer Management

- Default buffer size is 1000 items
- Adjust based on your processing speed
- Monitor queue sizes to prevent memory issues

### Threading

- Each stream runs in its own thread
- Callbacks are executed in the stream thread
- Use thread-safe data structures for shared state

### Memory Usage

```python
# Monitor memory usage
status = streaming_manager.get_stream_status()
for stream_id, info in status.items():
    print(f"{stream_id}: Queue size = {info['queue_size']}")
```

## Testing

### Running Tests

```bash
# Run streaming tests
python tests/tools/test_streaming.py

# Run example
python scripts/streaming_example.py --simple
python scripts/streaming_example.py --full
```

### Test Coverage

The test suite covers:
- Tick data streaming
- OHLCV data streaming
- Multiple streams
- Custom callbacks
- Stream management
- Error handling

## Integration with Trading Strategies

### Strategy Integration

```python
from app.strategy.base import Strategy

class StreamingStrategy(Strategy):
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.streaming_manager = None
    
    def initialize_streams(self, mt5_client):
        """Initialize data streams."""
        self.streaming_manager = StreamingManager(mt5_client)
        
        # Create streams for each symbol
        for symbol in self.symbols:
            # Tick stream for price monitoring
            self.streaming_manager.create_tick_stream(
                symbols=[symbol],
                callback=self.on_tick
            )
            
            # OHLCV stream for technical analysis
            self.streaming_manager.create_ohlcv_stream(
                symbols=[symbol],
                timeframe="H1",
                callback=self.on_ohlcv
            )
    
    def on_tick(self, tick_data):
        """Handle tick data."""
        # Implement tick-based logic
        pass
    
    def on_ohlcv(self, ohlcv_data):
        """Handle OHLCV data."""
        # Implement candle-based logic
        pass
    
    def start(self):
        """Start the strategy."""
        if self.streaming_manager:
            self.streaming_manager.start_all_streams()
    
    def stop(self):
        """Stop the strategy."""
        if self.streaming_manager:
            self.streaming_manager.stop_all_streams()
```

## Troubleshooting

### Common Issues

1. **No data received**
   - Check MT5 connection
   - Verify symbols are available
   - Check callback functions

2. **High memory usage**
   - Reduce buffer sizes
   - Process data faster
   - Monitor queue sizes

3. **Connection drops**
   - Check network stability
   - Verify MT5 terminal status
   - Review reconnection settings

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('app.data.streaming').setLevel(logging.DEBUG)

# Monitor stream status
status = streaming_manager.get_stream_status()
for stream_id, info in status.items():
    print(f"{stream_id}: {info}")
```

## API Reference

### StreamingManager

- `create_tick_stream(symbols, callback=None) -> str`
- `create_ohlcv_stream(symbols, timeframe, callback=None) -> str`
- `create_websocket_stream(url, callback=None) -> str`
- `start_stream(stream_id)`
- `stop_stream(stream_id)`
- `start_all_streams()`
- `stop_all_streams()`
- `get_stream_status() -> Dict`
- `remove_stream(stream_id)`

### Data Structures

#### TickData
- `symbol: str`
- `timestamp: datetime`
- `bid: float`
- `ask: float`
- `last: float`
- `volume: int`
- `spread: float`

#### OHLCVData
- `symbol: str`
- `timestamp: datetime`
- `open: float`
- `high: float`
- `low: float`
- `close: float`
- `volume: int`
- `timeframe: str`

#### StreamConfig
- `stream_type: StreamType`
- `symbols: List[str]`
- `timeframe: Optional[str]`
- `callback: Optional[Callable]`
- `buffer_size: int`
- `reconnect_interval: int`
- `max_reconnect_attempts: int`
- `enable_logging: bool` 