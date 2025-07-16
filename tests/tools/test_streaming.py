#!/usr/bin/env python3
"""
Test script for real-time data streaming functionality.

This script demonstrates the streaming features by creating and managing
various types of data streams.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.data.streaming import (
    StreamingManager, 
    StreamConfig, 
    StreamType,
    create_tick_callback, 
    create_ohlcv_callback,
    create_data_logger_callback
)
from app.data.mt5_client import MT5Client
from app.config.setup import DEFAULT_CONFIG_PATH, ALL_SYMBOLS, TEST_SYMBOL, DEFAULT_TIMEFRAME
from app.util.logger import get_logger

logger = get_logger(__name__)


def test_tick_streaming():
    """Test tick data streaming."""
    logger.info("=== Testing Tick Data Streaming ===")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create tick stream for test symbol
        tick_stream_id = streaming_manager.create_tick_stream(
            symbols=[TEST_SYMBOL],
            callback=create_tick_callback(TEST_SYMBOL)
        )
        
        logger.info(f"Created tick stream: {tick_stream_id}")
        
        # Start the stream
        streaming_manager.start_stream(tick_stream_id)
        
        # Let it run for a few seconds
        logger.info("Running tick stream for 10 seconds...")
        time.sleep(10)
        
        # Stop the stream
        streaming_manager.stop_stream(tick_stream_id)
        
        # Get status
        status = streaming_manager.get_stream_status()
        logger.info(f"Stream status: {status}")
        
        logger.info("Tick streaming test completed")
        
    except Exception as e:
        logger.error(f"Error in tick streaming test: {e}")


def test_ohlcv_streaming():
    """Test OHLCV data streaming."""
    logger.info("=== Testing OHLCV Data Streaming ===")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create OHLCV stream for test symbol
        ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
            symbols=[TEST_SYMBOL],
            timeframe=DEFAULT_TIMEFRAME,
            callback=create_ohlcv_callback(TEST_SYMBOL, DEFAULT_TIMEFRAME)
        )
        
        logger.info(f"Created OHLCV stream: {ohlcv_stream_id}")
        
        # Start the stream
        streaming_manager.start_stream(ohlcv_stream_id)
        
        # Let it run for a few seconds
        logger.info("Running OHLCV stream for 30 seconds...")
        time.sleep(30)
        
        # Stop the stream
        streaming_manager.stop_stream(ohlcv_stream_id)
        
        # Get status
        status = streaming_manager.get_stream_status()
        logger.info(f"Stream status: {status}")
        
        logger.info("OHLCV streaming test completed")
        
    except Exception as e:
        logger.error(f"Error in OHLCV streaming test: {e}")


def test_multiple_streams():
    """Test multiple streams simultaneously."""
    logger.info("=== Testing Multiple Streams ===")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create multiple streams
        symbols = [TEST_SYMBOL, "EURUSD", "GBPUSD"]
        
        # Create tick streams
        tick_streams = []
        for symbol in symbols:
            stream_id = streaming_manager.create_tick_stream(
                symbols=[symbol],
                callback=create_tick_callback(symbol)
            )
            tick_streams.append(stream_id)
        
        # Create OHLCV streams
        ohlcv_streams = []
        for symbol in symbols:
            stream_id = streaming_manager.create_ohlcv_stream(
                symbols=[symbol],
                timeframe=DEFAULT_TIMEFRAME,
                callback=create_ohlcv_callback(symbol, DEFAULT_TIMEFRAME)
            )
            ohlcv_streams.append(stream_id)
        
        logger.info(f"Created {len(tick_streams)} tick streams and {len(ohlcv_streams)} OHLCV streams")
        
        # Start all streams
        streaming_manager.start_all_streams()
        
        # Let them run for a while
        logger.info("Running multiple streams for 20 seconds...")
        time.sleep(20)
        
        # Stop all streams
        streaming_manager.stop_all_streams()
        
        # Get status
        status = streaming_manager.get_stream_status()
        logger.info(f"All streams status: {status}")
        
        logger.info("Multiple streams test completed")
        
    except Exception as e:
        logger.error(f"Error in multiple streams test: {e}")


def test_custom_callbacks():
    """Test custom callback functions."""
    logger.info("=== Testing Custom Callbacks ===")
    
    def custom_tick_handler(tick_data):
        """Custom tick data handler."""
        logger.info(f"Custom tick handler: {tick_data.symbol} - Bid: {tick_data.bid:.5f}, Ask: {tick_data.ask:.5f}, Spread: {tick_data.spread:.5f}")
    
    def custom_ohlcv_handler(ohlcv_data):
        """Custom OHLCV data handler."""
        logger.info(f"Custom OHLCV handler: {ohlcv_data.symbol} {ohlcv_data.timeframe} - Close: {ohlcv_data.close:.5f}, Volume: {ohlcv_data.volume}")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create streams with custom callbacks
        tick_stream_id = streaming_manager.create_tick_stream(
            symbols=[TEST_SYMBOL],
            callback=custom_tick_handler
        )
        
        ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
            symbols=[TEST_SYMBOL],
            timeframe=DEFAULT_TIMEFRAME,
            callback=custom_ohlcv_handler
        )
        
        logger.info("Created streams with custom callbacks")
        
        # Start streams
        streaming_manager.start_stream(tick_stream_id)
        streaming_manager.start_stream(ohlcv_stream_id)
        
        # Let them run
        logger.info("Running streams with custom callbacks for 15 seconds...")
        time.sleep(15)
        
        # Stop streams
        streaming_manager.stop_stream(tick_stream_id)
        streaming_manager.stop_stream(ohlcv_stream_id)
        
        logger.info("Custom callbacks test completed")
        
    except Exception as e:
        logger.error(f"Error in custom callbacks test: {e}")


def test_stream_management():
    """Test stream management features."""
    logger.info("=== Testing Stream Management ===")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create multiple streams
        stream_ids = []
        for i, symbol in enumerate([TEST_SYMBOL, "EURUSD"]):
            # Tick stream
            tick_id = streaming_manager.create_tick_stream([symbol])
            stream_ids.append(tick_id)
            
            # OHLCV stream
            ohlcv_id = streaming_manager.create_ohlcv_stream([symbol], DEFAULT_TIMEFRAME)
            stream_ids.append(ohlcv_id)
        
        logger.info(f"Created {len(stream_ids)} streams: {stream_ids}")
        
        # Test individual stream control
        for stream_id in stream_ids:
            logger.info(f"Starting stream: {stream_id}")
            streaming_manager.start_stream(stream_id)
            time.sleep(2)
            
            status = streaming_manager.get_stream_status()
            logger.info(f"Status after starting {stream_id}: {status[stream_id]}")
        
        # Test stopping individual streams
        for stream_id in stream_ids:
            logger.info(f"Stopping stream: {stream_id}")
            streaming_manager.stop_stream(stream_id)
            time.sleep(1)
        
        # Test removing streams
        for stream_id in stream_ids:
            logger.info(f"Removing stream: {stream_id}")
            streaming_manager.remove_stream(stream_id)
        
        logger.info("Stream management test completed")
        
    except Exception as e:
        logger.error(f"Error in stream management test: {e}")


def main():
    """Run all streaming tests."""
    logger.info("Starting Real-time Data Streaming Tests")
    logger.info("=" * 50)
    
    try:
        # Run tests
        test_tick_streaming()
        time.sleep(2)
        
        test_ohlcv_streaming()
        time.sleep(2)
        
        test_multiple_streams()
        time.sleep(2)
        
        test_custom_callbacks()
        time.sleep(2)
        
        test_stream_management()
        
        logger.info("=" * 50)
        logger.info("All streaming tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 