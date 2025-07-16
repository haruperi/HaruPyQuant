#!/usr/bin/env python3
"""
Example script demonstrating real-time data streaming functionality.

This script shows how to:
1. Set up real-time tick and OHLCV data streams
2. Process incoming data with custom callbacks
3. Integrate streaming with trading strategies
4. Handle multiple symbols simultaneously
"""

import os
import sys
import time
import signal
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.streaming import (
    StreamingManager, 
    TickData, 
    OHLCVData,
    create_tick_callback, 
    create_ohlcv_callback
)
from app.data.mt5_client import MT5Client
from app.config.setup import DEFAULT_CONFIG_PATH, ALL_SYMBOLS, TEST_SYMBOL, DEFAULT_TIMEFRAME
from app.util.logger import get_logger

logger = get_logger(__name__)


class TradingDataProcessor:
    """
    Example data processor that demonstrates how to handle real-time data.
    
    This class shows how to:
    - Process tick data for price monitoring
    - Process OHLCV data for technical analysis
    - Maintain data buffers for analysis
    - Generate trading signals
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.tick_data_buffer: Dict[str, List[TickData]] = {}
        self.ohlcv_data_buffer: Dict[str, List[OHLCVData]] = {}
        self.max_buffer_size = 100
        
        # Initialize buffers
        for symbol in symbols:
            self.tick_data_buffer[symbol] = []
            self.ohlcv_data_buffer[symbol] = []
        
        # Trading signals
        self.signals = []
        
        logger.info(f"Initialized TradingDataProcessor for symbols: {symbols}")
    
    def process_tick_data(self, tick_data: TickData):
        """Process incoming tick data."""
        symbol = tick_data.symbol
        
        # Add to buffer
        self.tick_data_buffer[symbol].append(tick_data)
        
        # Maintain buffer size
        if len(self.tick_data_buffer[symbol]) > self.max_buffer_size:
            self.tick_data_buffer[symbol].pop(0)
        
        # Analyze tick data
        self._analyze_tick_data(tick_data)
    
    def process_ohlcv_data(self, ohlcv_data: OHLCVData):
        """Process incoming OHLCV data."""
        symbol = ohlcv_data.symbol
        
        # Add to buffer
        self.ohlcv_data_buffer[symbol].append(ohlcv_data)
        
        # Maintain buffer size
        if len(self.ohlcv_data_buffer[symbol]) > self.max_buffer_size:
            self.ohlcv_data_buffer[symbol].pop(0)
        
        # Analyze OHLCV data
        self._analyze_ohlcv_data(ohlcv_data)
    
    def _analyze_tick_data(self, tick_data: TickData):
        """Analyze tick data for trading opportunities."""
        # Example: Monitor spread changes
        if tick_data.spread > 0.0005:  # 5 pips spread
            logger.warning(f"High spread detected for {tick_data.symbol}: {tick_data.spread:.5f}")
        
        # Example: Price movement detection
        if len(self.tick_data_buffer[tick_data.symbol]) >= 2:
            prev_tick = self.tick_data_buffer[tick_data.symbol][-2]
            price_change = tick_data.last - prev_tick.last
            
            if abs(price_change) > 0.001:  # 1 pip movement
                direction = "UP" if price_change > 0 else "DOWN"
                logger.info(f"Significant price movement for {tick_data.symbol}: {direction} {abs(price_change):.5f}")
    
    def _analyze_ohlcv_data(self, ohlcv_data: OHLCVData):
        """Analyze OHLCV data for trading signals."""
        symbol = ohlcv_data.symbol
        
        # Example: Simple moving average calculation
        if len(self.ohlcv_data_buffer[symbol]) >= 20:
            recent_data = self.ohlcv_data_buffer[symbol][-20:]
            sma = sum(candle.close for candle in recent_data) / len(recent_data)
            
            # Generate signal if price crosses SMA
            if ohlcv_data.close > sma * 1.001:  # 0.1% above SMA
                signal = {
                    'timestamp': ohlcv_data.timestamp,
                    'symbol': symbol,
                    'type': 'BUY',
                    'price': ohlcv_data.close,
                    'reason': 'Price above 20-period SMA'
                }
                self.signals.append(signal)
                logger.info(f"BUY signal generated for {symbol} at {ohlcv_data.close:.5f}")
            
            elif ohlcv_data.close < sma * 0.999:  # 0.1% below SMA
                signal = {
                    'timestamp': ohlcv_data.timestamp,
                    'symbol': symbol,
                    'type': 'SELL',
                    'price': ohlcv_data.close,
                    'reason': 'Price below 20-period SMA'
                }
                self.signals.append(signal)
                logger.info(f"SELL signal generated for {symbol} at {ohlcv_data.close:.5f}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        stats = {
            'symbols': self.symbols,
            'tick_buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.tick_data_buffer.items()},
            'ohlcv_buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.ohlcv_data_buffer.items()},
            'total_signals': len(self.signals),
            'recent_signals': self.signals[-5:] if self.signals else []
        }
        return stats


def create_processor_callbacks(processor: TradingDataProcessor):
    """Create callback functions for the data processor."""
    
    def tick_callback(tick_data: TickData):
        processor.process_tick_data(tick_data)
    
    def ohlcv_callback(ohlcv_data: OHLCVData):
        processor.process_ohlcv_data(ohlcv_data)
    
    return tick_callback, ohlcv_callback


def run_streaming_example():
    """Run the streaming example."""
    logger.info("Starting Real-time Data Streaming Example")
    logger.info("=" * 60)
    
    try:
        # Initialize MT5 client
        logger.info("Initializing MT5 client...")
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create data processor
        symbols = [TEST_SYMBOL, "EURUSD", "GBPUSD"]
        processor = TradingDataProcessor(symbols)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create callback functions
        tick_callback, ohlcv_callback = create_processor_callbacks(processor)
        
        # Create streams for each symbol
        stream_ids = []
        
        for symbol in symbols:
            # Create tick stream
            tick_stream_id = streaming_manager.create_tick_stream(
                symbols=[symbol],
                callback=tick_callback
            )
            stream_ids.append(tick_stream_id)
            
            # Create OHLCV stream
            ohlcv_stream_id = streaming_manager.create_ohlcv_stream(
                symbols=[symbol],
                timeframe=DEFAULT_TIMEFRAME,
                callback=ohlcv_callback
            )
            stream_ids.append(ohlcv_stream_id)
        
        logger.info(f"Created {len(stream_ids)} data streams")
        
        # Start all streams
        logger.info("Starting all data streams...")
        streaming_manager.start_all_streams()
        
        # Monitor and display statistics
        logger.info("Monitoring data streams...")
        logger.info("Press Ctrl+C to stop")
        
        start_time = time.time()
        while True:
            time.sleep(10)  # Update every 10 seconds
            
            # Display statistics
            stats = processor.get_statistics()
            elapsed = time.time() - start_time
            
            logger.info(f"\n--- Statistics (Elapsed: {elapsed:.1f}s) ---")
            logger.info(f"Tick buffer sizes: {stats['tick_buffer_sizes']}")
            logger.info(f"OHLCV buffer sizes: {stats['ohlcv_buffer_sizes']}")
            logger.info(f"Total signals generated: {stats['total_signals']}")
            
            if stats['recent_signals']:
                logger.info("Recent signals:")
                for signal in stats['recent_signals']:
                    logger.info(f"  {signal['timestamp']} {signal['symbol']} {signal['type']} @ {signal['price']:.5f}")
            
            # Display stream status
            stream_status = streaming_manager.get_stream_status()
            logger.info(f"Stream status: {len([s for s in stream_status.values() if s['running']])}/{len(stream_status)} streams running")
        
    except KeyboardInterrupt:
        logger.info("\nStopping data streams...")
        streaming_manager.stop_all_streams()
        
        # Final statistics
        stats = processor.get_statistics()
        logger.info(f"\n--- Final Statistics ---")
        logger.info(f"Total signals generated: {stats['total_signals']}")
        logger.info(f"Total tick data processed: {sum(stats['tick_buffer_sizes'].values())}")
        logger.info(f"Total OHLCV data processed: {sum(stats['ohlcv_buffer_sizes'].values())}")
        
        logger.info("Streaming example completed")
        
    except Exception as e:
        logger.error(f"Error in streaming example: {e}")
        raise


def run_simple_example():
    """Run a simple streaming example for testing."""
    logger.info("Starting Simple Streaming Example")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        
        # Create streaming manager
        streaming_manager = StreamingManager(mt5_client)
        
        # Create simple tick stream
        tick_stream_id = streaming_manager.create_tick_stream(
            symbols=[TEST_SYMBOL],
            callback=create_tick_callback(TEST_SYMBOL)
        )
        
        logger.info(f"Created tick stream: {tick_stream_id}")
        
        # Start stream
        streaming_manager.start_stream(tick_stream_id)
        
        # Run for 30 seconds
        logger.info("Running tick stream for 30 seconds...")
        time.sleep(30)
        
        # Stop stream
        streaming_manager.stop_stream(tick_stream_id)
        
        logger.info("Simple example completed")
        
    except Exception as e:
        logger.error(f"Error in simple example: {e}")
        raise


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Data Streaming Example")
    parser.add_argument("--simple", action="store_true", help="Run simple example")
    parser.add_argument("--full", action="store_true", help="Run full example with data processing")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_example()
    elif args.full:
        run_streaming_example()
    else:
        # Default to simple example
        run_simple_example()


if __name__ == "__main__":
    main() 