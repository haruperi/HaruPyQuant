#!/usr/bin/env python3
"""
Test script to verify MT5 symbol initialization fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.setup import *
from app.data.mt5_client import MT5Client
from app.util.logger import get_logger

logger = get_logger(__name__)

def test_symbol_initialization():
    """Test the improved symbol initialization functionality."""
    
    print("=== Testing MT5 Symbol Initialization ===")
    
    # Test with broker 2 (demo account)
    print(f"\nTesting with broker 2 (demo account)")
    print(f"Symbols to initialize: {FOREX_SYMBOLS}")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=2)
        
        # Print detailed symbol status
        mt5_client.print_symbol_status()
        
        # Get market watch symbols
        market_watch_symbols = mt5_client.get_market_watch_symbols()
        print(f"\nMarket watch symbols: {market_watch_symbols}")
        
        # Check availability
        availability = mt5_client.check_symbol_availability()
        print(f"\nSymbol availability: {availability}")
        
        # Count visible symbols
        visible_count = 0
        for symbol in FOREX_SYMBOLS:
            broker_symbol = mt5_client.map_symbol(symbol)
            if mt5_client.is_symbol_visible(broker_symbol):
                visible_count += 1
        
        print(f"\nResults:")
        print(f"  - Total symbols requested: {len(FOREX_SYMBOLS)}")
        print(f"  - Symbols visible in market watch: {visible_count}")
        print(f"  - Success rate: {visible_count/len(FOREX_SYMBOLS)*100:.1f}%")
        
        if visible_count == len(FOREX_SYMBOLS):
            print("✅ All symbols successfully initialized!")
        else:
            print(f"⚠️  Only {visible_count}/{len(FOREX_SYMBOLS)} symbols are visible")
        
        # Cleanup
        mt5_client.shutdown()
        
    except Exception as e:
        logger.error(f"Error during symbol initialization test: {e}")
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def test_broker_3_symbols():
    """Test symbol initialization with broker 3 (Purple Trader)."""
    
    print("\n=== Testing Broker 3 Symbol Initialization ===")
    
    # Test with a subset of symbols for broker 3
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    print(f"Test symbols: {test_symbols}")
    
    try:
        # Initialize MT5 client for broker 3
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=test_symbols, broker=3)
        
        # Print detailed symbol status
        mt5_client.print_symbol_status()
        
        # Count visible symbols
        visible_count = 0
        for symbol in test_symbols:
            broker_symbol = mt5_client.map_symbol(symbol)
            if mt5_client.is_symbol_visible(broker_symbol):
                visible_count += 1
        
        print(f"\nResults for broker 3:")
        print(f"  - Total symbols requested: {len(test_symbols)}")
        print(f"  - Symbols visible in market watch: {visible_count}")
        print(f"  - Success rate: {visible_count/len(test_symbols)*100:.1f}%")
        
        # Cleanup
        mt5_client.shutdown()
        
    except Exception as e:
        logger.error(f"Error during broker 3 symbol initialization test: {e}")
        print(f"❌ Broker 3 test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting MT5 Symbol Initialization Tests...")
    
    # Test broker 2
    success1 = test_symbol_initialization()
    
    # Test broker 3
    success2 = test_broker_3_symbols()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 