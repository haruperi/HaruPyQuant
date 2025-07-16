"""
Real-time data streaming handlers for HaruPyQuant.

This module provides real-time data streaming capabilities for:
- MT5 tick data streaming
- OHLCV data streaming
- WebSocket-based streaming
- Callback-based data processing
"""

import asyncio
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
try:
    import websocket
except ImportError:
    websocket = None
from concurrent.futures import ThreadPoolExecutor

from app.util.logger import get_logger
from app.data.mt5_client import MT5Client

logger = get_logger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    TICK = "tick"
    OHLCV = "ohlcv"
    FUNDAMENTAL = "fundamental"
    WEBSOCKET = "websocket"


@dataclass
class StreamConfig:
    """Configuration for data streams."""
    stream_type: StreamType
    symbols: List[str]
    timeframe: Optional[str] = None
    callback: Optional[Callable] = None
    buffer_size: int = 1000
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    enable_logging: bool = True


@dataclass
class TickData:
    """Tick data structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid


@dataclass
class OHLCVData:
    """OHLCV data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str


class DataStreamHandler:
    """
    Base class for data stream handlers.
    
    Provides common functionality for all streaming implementations.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=config.buffer_size)
        self.callbacks: List[Callable] = []
        self.reconnect_attempts = 0
        self.last_data_time = None
        
        if config.callback:
            self.register_callback(config.callback)
    
    def register_callback(self, callback: Callable):
        """Register a callback function to be called when new data arrives."""
        self.callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable):
        """Unregister a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Unregistered callback: {callback.__name__}")
    
    def _notify_callbacks(self, data: Any):
        """Notify all registered callbacks with new data."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
    
    def start(self):
        """Start the data stream."""
        if self.is_running:
            logger.warning("Stream is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting {self.config.stream_type.value} stream for symbols: {self.config.symbols}")
    
    def stop(self):
        """Stop the data stream."""
        self.is_running = False
        logger.info(f"Stopped {self.config.stream_type.value} stream")
    
    def get_latest_data(self) -> Optional[Any]:
        """Get the latest data from the queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.data_queue.qsize()


class MT5TickStreamHandler(DataStreamHandler):
    """
    Real-time tick data streaming from MT5.
    
    Provides live tick data for specified symbols.
    """
    
    def __init__(self, config: StreamConfig, mt5_client: MT5Client):
        super().__init__(config)
        self.mt5_client = mt5_client
        self.stream_thread = None
        self.last_tick_data: Dict[str, TickData] = {}
    
    def start(self):
        """Start the MT5 tick stream."""
        super().start()
        
        self.stream_thread = threading.Thread(
            target=self._tick_stream_loop,
            daemon=True,
            name="MT5TickStream"
        )
        self.stream_thread.start()
        logger.info("MT5 tick stream started")
    
    def stop(self):
        """Stop the MT5 tick stream."""
        super().stop()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        
        logger.info("MT5 tick stream stopped")
    
    def _tick_stream_loop(self):
        """Main tick streaming loop."""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    if not self.is_running:
                        break
                    
                    tick_data = self._get_tick_data(symbol)
                    if tick_data:
                        self._process_tick_data(tick_data)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in tick stream loop: {e}")
                time.sleep(1)
    
    def _get_tick_data(self, symbol: str) -> Optional[TickData]:
        """Get tick data for a symbol."""
        try:
            tick = self.mt5_client.get_tick(symbol)
            if tick:
                # Handle both dictionary and named tuple formats
                if hasattr(tick, '_asdict'):
                    # Named tuple - convert to dict
                    tick_dict = tick._asdict()  # type: ignore
                elif isinstance(tick, dict):
                    # Already a dictionary
                    tick_dict = tick
                else:
                    # Try to access as attributes
                    tick_dict = {
                        'time': getattr(tick, 'time', None),
                        'bid': getattr(tick, 'bid', None),
                        'ask': getattr(tick, 'ask', None),
                        'last': getattr(tick, 'last', None),
                        'volume': getattr(tick, 'volume', None)
                    }
                
                # Validate required fields
                required_fields = ['time', 'bid', 'ask', 'last', 'volume']
                if not all(field in tick_dict and tick_dict[field] is not None for field in required_fields):
                    logger.warning(f"Missing required fields in tick data for {symbol}: {tick_dict}")
                    return None
                
                # Convert values with proper type checking
                time_val = tick_dict['time']
                bid_val = tick_dict['bid']
                ask_val = tick_dict['ask']
                last_val = tick_dict['last']
                volume_val = tick_dict['volume']
                
                if not all(isinstance(val, (int, float)) for val in [time_val, bid_val, ask_val, last_val, volume_val]):
                    logger.warning(f"Invalid data types in tick data for {symbol}: {tick_dict}")
                    return None
                
                return TickData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(float(time_val), tz=timezone.utc),  # type: ignore
                    bid=float(bid_val),  # type: ignore
                    ask=float(ask_val),  # type: ignore
                    last=float(last_val),  # type: ignore
                    volume=int(volume_val)  # type: ignore
                )
        except Exception as e:
            logger.error(f"Error getting tick data for {symbol}: {e}")
        
        return None
    
    def _process_tick_data(self, tick_data: TickData):
        """Process incoming tick data."""
        # Check if this is new data
        last_tick = self.last_tick_data.get(tick_data.symbol)
        if last_tick and last_tick.timestamp >= tick_data.timestamp:
            return  # Skip old data
        
        # Update last tick data
        self.last_tick_data[tick_data.symbol] = tick_data
        
        # Add to queue
        try:
            self.data_queue.put_nowait(tick_data)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(tick_data)
            except queue.Empty:
                pass
        
        # Notify callbacks
        self._notify_callbacks(tick_data)
        
        if self.config.enable_logging:
            logger.debug(f"Tick: {tick_data.symbol} Bid: {tick_data.bid} Ask: {tick_data.ask} Spread: {tick_data.spread:.5f}")


class MT5OHLCVStreamHandler(DataStreamHandler):
    """
    Real-time OHLCV data streaming from MT5.
    
    Provides live OHLCV data for specified symbols and timeframes.
    """
    
    def __init__(self, config: StreamConfig, mt5_client: MT5Client):
        super().__init__(config)
        self.mt5_client = mt5_client
        self.stream_thread = None
        self.last_candle_data: Dict[str, OHLCVData] = {}
        self.candle_cache: Dict[str, List[Dict]] = {}
    
    def start(self):
        """Start the MT5 OHLCV stream."""
        super().start()
        
        self.stream_thread = threading.Thread(
            target=self._ohlcv_stream_loop,
            daemon=True,
            name="MT5OHLCVStream"
        )
        self.stream_thread.start()
        logger.info(f"MT5 OHLCV stream started for timeframe: {self.config.timeframe}")
    
    def stop(self):
        """Stop the MT5 OHLCV stream."""
        super().stop()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        
        logger.info("MT5 OHLCV stream stopped")
    
    def _ohlcv_stream_loop(self):
        """Main OHLCV streaming loop."""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    if not self.is_running:
                        break
                    
                    ohlcv_data = self._get_ohlcv_data(symbol)
                    if ohlcv_data:
                        self._process_ohlcv_data(ohlcv_data)
                
                # Wait for next candle
                time.sleep(self._get_candle_interval())
                
            except Exception as e:
                logger.error(f"Error in OHLCV stream loop: {e}")
                time.sleep(5)
    
    def _get_ohlcv_data(self, symbol: str) -> Optional[OHLCVData]:
        """Get OHLCV data for a symbol."""
        try:
            # Get the latest candle
            if not self.config.timeframe:
                return None
                
            df = self.mt5_client.fetch_data(
                symbol, 
                self.config.timeframe, 
                start_pos=0, 
                end_pos=1
            )
            
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                timestamp = df.index[-1]
                if isinstance(timestamp, datetime):
                    return OHLCVData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=latest['Open'],
                        high=latest['High'],
                        low=latest['Low'],
                        close=latest['Close'],
                        volume=latest['Volume'],
                        timeframe=self.config.timeframe
                    )
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
        
        return None
    
    def _process_ohlcv_data(self, ohlcv_data: OHLCVData):
        """Process incoming OHLCV data."""
        # Check if this is new data
        last_candle = self.last_candle_data.get(ohlcv_data.symbol)
        if last_candle and last_candle.timestamp >= ohlcv_data.timestamp:
            return  # Skip old data
        
        # Update last candle data
        self.last_candle_data[ohlcv_data.symbol] = ohlcv_data
        
        # Add to queue
        try:
            self.data_queue.put_nowait(ohlcv_data)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(ohlcv_data)
            except queue.Empty:
                pass
        
        # Notify callbacks
        self._notify_callbacks(ohlcv_data)
        
        if self.config.enable_logging:
            logger.debug(f"Candle: {ohlcv_data.symbol} {ohlcv_data.timeframe} O:{ohlcv_data.open:.5f} H:{ohlcv_data.high:.5f} L:{ohlcv_data.low:.5f} C:{ohlcv_data.close:.5f}")
    
    def _get_candle_interval(self) -> int:
        """Get the interval between candle checks in seconds."""
        if not self.config.timeframe:
            return 60
        
        timeframe = self.config.timeframe.upper()
        if timeframe.startswith('M'):
            return int(timeframe[1:]) * 60
        elif timeframe.startswith('H'):
            return int(timeframe[1:]) * 3600
        elif timeframe == 'D1':
            return 86400
        else:
            return 60


class WebSocketStreamHandler(DataStreamHandler):
    """
    WebSocket-based data streaming handler.
    
    Supports external WebSocket data sources.
    """
    
    def __init__(self, config: StreamConfig, websocket_url: str):
        super().__init__(config)
        self.websocket_url = websocket_url
        self.websocket = None
        self.stream_thread = None
    
    def start(self):
        """Start the WebSocket stream."""
        super().start()
        
        self.stream_thread = threading.Thread(
            target=self._websocket_stream_loop,
            daemon=True,
            name="WebSocketStream"
        )
        self.stream_thread.start()
        logger.info(f"WebSocket stream started: {self.websocket_url}")
    
    def stop(self):
        """Stop the WebSocket stream."""
        super().stop()
        
        if self.websocket:
            self.websocket.close()
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        
        logger.info("WebSocket stream stopped")
    
    def _websocket_stream_loop(self):
        """Main WebSocket streaming loop."""
        while self.is_running:
            try:
                self._connect_websocket()
                self._receive_messages()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    time.sleep(self.config.reconnect_interval)
    
    def _connect_websocket(self):
        """Connect to WebSocket."""
        if websocket is None:
            logger.error("WebSocket library not available. Install with: pip install websocket-client")
            return
            
        try:
            self.websocket = websocket.WebSocketApp(
                self.websocket_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.websocket.run_forever()
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            self._process_websocket_data(data)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info("WebSocket connection closed")
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        logger.info("WebSocket connection opened")
    
    def _receive_messages(self):
        """Receive messages from WebSocket."""
        if self.websocket:
            self.websocket.run_forever()
    
    def _process_websocket_data(self, data: Dict[str, Any]):
        """Process incoming WebSocket data."""
        # Add to queue
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
            except queue.Empty:
                pass
        
        # Notify callbacks
        self._notify_callbacks(data)
        
        if self.config.enable_logging:
            logger.debug(f"WebSocket data: {data}")


class StreamingManager:
    """
    Manages multiple data streams.
    
    Provides a centralized interface for managing different types of data streams.
    """
    
    def __init__(self, mt5_client: MT5Client):
        self.mt5_client = mt5_client
        self.streams: Dict[str, DataStreamHandler] = {}
        self.global_callbacks: List[Callable] = []
    
    def create_tick_stream(self, symbols: List[str], callback: Optional[Callable] = None) -> str:
        """Create a tick data stream."""
        config = StreamConfig(
            stream_type=StreamType.TICK,
            symbols=symbols,
            callback=callback
        )
        
        stream_id = f"tick_{'_'.join(symbols)}"
        self.streams[stream_id] = MT5TickStreamHandler(config, self.mt5_client)
        
        logger.info(f"Created tick stream: {stream_id}")
        return stream_id
    
    def create_ohlcv_stream(self, symbols: List[str], timeframe: str, callback: Optional[Callable] = None) -> str:
        """Create an OHLCV data stream."""
        config = StreamConfig(
            stream_type=StreamType.OHLCV,
            symbols=symbols,
            timeframe=timeframe,
            callback=callback
        )
        
        stream_id = f"ohlcv_{timeframe}_{'_'.join(symbols)}"
        self.streams[stream_id] = MT5OHLCVStreamHandler(config, self.mt5_client)
        
        logger.info(f"Created OHLCV stream: {stream_id}")
        return stream_id
    
    def create_websocket_stream(self, url: str, callback: Optional[Callable] = None) -> str:
        """Create a WebSocket data stream."""
        config = StreamConfig(
            stream_type=StreamType.WEBSOCKET,
            symbols=[],  # Not applicable for WebSocket
            callback=callback
        )
        
        stream_id = f"websocket_{hash(url) % 10000}"
        self.streams[stream_id] = WebSocketStreamHandler(config, url)
        
        logger.info(f"Created WebSocket stream: {stream_id}")
        return stream_id
    
    def start_stream(self, stream_id: str):
        """Start a specific stream."""
        if stream_id in self.streams:
            self.streams[stream_id].start()
        else:
            logger.error(f"Stream not found: {stream_id}")
    
    def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
        else:
            logger.error(f"Stream not found: {stream_id}")
    
    def start_all_streams(self):
        """Start all streams."""
        for stream_id in self.streams:
            self.start_stream(stream_id)
    
    def stop_all_streams(self):
        """Stop all streams."""
        for stream_id in self.streams:
            self.stop_stream(stream_id)
    
    def get_stream_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all streams."""
        status = {}
        for stream_id, stream in self.streams.items():
            status[stream_id] = {
                'running': stream.is_running,
                'queue_size': stream.get_queue_size(),
                'last_data_time': stream.last_data_time,
                'reconnect_attempts': stream.reconnect_attempts
            }
        return status
    
    def register_global_callback(self, callback: Callable):
        """Register a callback for all streams."""
        self.global_callbacks.append(callback)
        for stream in self.streams.values():
            stream.register_callback(callback)
    
    def remove_stream(self, stream_id: str):
        """Remove a stream."""
        if stream_id in self.streams:
            self.stop_stream(stream_id)
            del self.streams[stream_id]
            logger.info(f"Removed stream: {stream_id}")


# Utility functions for common streaming operations
def create_tick_callback(symbol: str):
    """Create a callback function for tick data."""
    def callback(tick_data: TickData):
        if tick_data.symbol == symbol:
            print(f"Tick: {tick_data.symbol} Bid: {tick_data.bid} Ask: {tick_data.ask}")
    return callback


def create_ohlcv_callback(symbol: str, timeframe: str):
    """Create a callback function for OHLCV data."""
    def callback(ohlcv_data: OHLCVData):
        if ohlcv_data.symbol == symbol and ohlcv_data.timeframe == timeframe:
            print(f"Candle: {ohlcv_data.symbol} {ohlcv_data.timeframe} Close: {ohlcv_data.close}")
    return callback


def create_data_logger_callback():
    """Create a callback function that logs all data."""
    def callback(data: Any):
        logger.info(f"Received data: {data}")
    return callback 