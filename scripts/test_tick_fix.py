#!/usr/bin/env python3
"""
Test script to verify the tick data streaming fix.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data import StreamingManager, MT5Client
from app.config.setup import DEFAULT_CONFIG_PATH, TEST_SYMBOL
from app.util.logger import get_logger

logger = get_logger(__name__)


def test_tick_streaming():
    """Test tick data streaming to verify the fix."""
    logger.info("Testing tick data streaming fix...")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=[TEST_SYMBOL])
        
        # Test direct tick access first
        logger.info("Testing direct tick access...")
        tick = mt5_client.get_tick(TEST_SYMBOL)
        if tick:
            logger.info(f"Direct tick access successful: {tick}")
            logger.info(f"Tick type: {type(tick)}")
            logger.info(f"Tick keys/attributes: {dir(tick) if hasattr(tick, '__dict__') else 'No dict'}")
        else:
            logger.error("Failed to get tick data directly")
            return
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create tick stream
        tick_stream_id = streaming_manager.create_tick_stream(
            symbols=[TEST_SYMBOL],
            callback=lambda data: logger.info(f"Tick received: {data.symbol} Bid: {data.bid} Ask: {data.ask} Spread: {data.spread:.5f}")
        )
        
        logger.info(f"Created tick stream: {tick_stream_id}")
        
        # Start the stream
        streaming_manager.start_stream(tick_stream_id)
        
        # Let it run for 10 seconds
        logger.info("Running tick stream for 10 seconds...")
        time.sleep(10)
        
        # Stop the stream
        streaming_manager.stop_stream(tick_stream_id)
        
        # Get status
        status = streaming_manager.get_stream_status()
        logger.info(f"Stream status: {status}")
        
        logger.info("Tick streaming test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in tick streaming test: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_symbols():
    """Test tick streaming with multiple symbols."""
    logger.info("Testing tick streaming with multiple symbols...")
    
    try:
        # Initialize MT5 client
        symbols = [TEST_SYMBOL, "EURUSD", "GBPUSD"]
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=symbols)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create tick streams for each symbol
        stream_ids = []
        for symbol in symbols:
            stream_id = streaming_manager.create_tick_stream(
                symbols=[symbol],
                callback=lambda data, sym=symbol: logger.info(f"Tick {sym}: Bid: {data.bid} Ask: {data.ask}")
            )
            stream_ids.append(stream_id)
        
        logger.info(f"Created {len(stream_ids)} tick streams")
        
        # Start all streams
        streaming_manager.start_all_streams()
        
        # Let them run for 15 seconds
        logger.info("Running multiple tick streams for 15 seconds...")
        time.sleep(15)
        
        # Stop all streams
        streaming_manager.stop_all_streams()
        
        logger.info("Multiple symbols test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in multiple symbols test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the tests."""
    logger.info("Starting tick data streaming fix verification...")
    
    # Test single symbol
    test_tick_streaming()
    
    time.sleep(2)
    
    # Test multiple symbols
    test_multiple_symbols()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    main() 