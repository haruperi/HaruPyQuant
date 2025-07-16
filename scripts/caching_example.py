#!/usr/bin/env python3

import pandas as pd
"""
Example script demonstrating data caching integration with MT5.

This script shows how to:
1. Integrate caching with MT5 data retrieval
2. Improve performance with cached data
3. Monitor cache effectiveness
4. Handle cache invalidation
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.caching import DataCache, create_cache_config, get_cache_stats_summary
from app.data.mt5_client import MT5Client
from app.config.setup import DEFAULT_CONFIG_PATH, TEST_SYMBOL, DEFAULT_TIMEFRAME
from app.util.logger import get_logger

logger = get_logger(__name__)


class CachedMT5Client:
    """
    MT5 client with integrated caching for improved performance.
    
    This class wraps the MT5 client and adds caching capabilities
    to reduce redundant data requests and improve response times.
    """
    
    def __init__(self, mt5_client: MT5Client, cache: DataCache):
        self.mt5_client = mt5_client
        self.cache = cache
        self.request_count = 0
        self.cache_hit_count = 0
    
    def fetch_data_with_cache(self, symbol: str, timeframe: str = "D1", 
                             start_pos: Optional[int] = None, end_pos: Optional[int] = None,
                             start_date=None, end_date=None, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with caching.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "M1", "H1", "D1")
            start_pos: Start position
            end_pos: End position
            start_date: Start date
            end_date: End date
            force_refresh: Force refresh from MT5 (ignore cache)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        self.request_count += 1
        
        # Try to get from cache first (unless force refresh)
        if not force_refresh:
            cached_data = self.cache.get_cached_ohlcv(
                symbol, timeframe, 
                start_pos=start_pos, end_pos=end_pos,
                start_date=start_date, end_date=end_date
            )
            
            if cached_data is not None:
                self.cache_hit_count += 1
                logger.debug(f"Cache HIT for {symbol} {timeframe}")
                return cached_data
        
        # Cache miss or force refresh - fetch from MT5
        logger.debug(f"Cache MISS for {symbol} {timeframe} - fetching from MT5")
        
        data = self.mt5_client.fetch_data(
            symbol, timeframe, start_pos, end_pos, start_date, end_date
        )
        
        if data is not None:
            # Cache the data
            success = self.cache.set_cached_ohlcv(
                symbol, timeframe, data,
                start_pos=start_pos, end_pos=end_pos,
                start_date=start_date, end_date=end_date
            )
            
            if success:
                logger.debug(f"Cached {symbol} {timeframe} data")
            else:
                logger.warning(f"Failed to cache {symbol} {timeframe} data")
        
        return data
    
    def get_tick_with_cache(self, symbol: str, force_refresh: bool = False) -> Optional[dict]:
        """
        Get tick data with caching.
        
        Args:
            symbol: Trading symbol
            force_refresh: Force refresh from MT5
            
        Returns:
            Tick data dictionary or None if failed
        """
        self.request_count += 1
        
        # Try to get from cache first
        if not force_refresh:
            cached_tick = self.cache.get_cached_tick(symbol)
            if cached_tick is not None:
                self.cache_hit_count += 1
                logger.debug(f"Cache HIT for {symbol} tick")
                return cached_tick
        
        # Cache miss - fetch from MT5
        logger.debug(f"Cache MISS for {symbol} tick - fetching from MT5")
        
        tick_data = self.mt5_client.get_tick(symbol)
        
        if tick_data is not None:
            # Cache the tick data
            success = self.cache.set_cached_tick(symbol, tick_data)
            if success:
                logger.debug(f"Cached {symbol} tick data")
            else:
                logger.warning(f"Failed to cache {symbol} tick data")
        
        return tick_data
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        cache_stats = self.cache.get_stats()
        
        total_requests = self.request_count
        cache_hits = self.cache_hit_count
        cache_misses = total_requests - cache_hits
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': hit_rate,
            'cache_stats': cache_stats
        }


def demonstrate_caching_performance():
    """Demonstrate the performance benefits of caching."""
    logger.info("=== Demonstrating Caching Performance ===")
    
    try:
        # Initialize MT5 client
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=[TEST_SYMBOL])
        
        # Create cache
        cache_config = create_cache_config(
            cache_type="hybrid",
            max_memory_mb=50,
            max_disk_gb=1,
            ttl_hours=1
        )
        cache = DataCache(cache_config)
        
        # Create cached client
        cached_client = CachedMT5Client(mt5_client, cache)
        
        # Test 1: First request (cache miss)
        logger.info("Test 1: First request (should be cache miss)")
        start_time = time.time()
        data1 = cached_client.fetch_data_with_cache(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=100)
        time1 = time.time() - start_time
        logger.info(f"First request took {time1:.4f} seconds")
        
        # Test 2: Second request (cache hit)
        logger.info("Test 2: Second request (should be cache hit)")
        start_time = time.time()
        data2 = cached_client.fetch_data_with_cache(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=100)
        time2 = time.time() - start_time
        logger.info(f"Second request took {time2:.4f} seconds")
        
        # Calculate performance improvement
        if time1 > 0:
            improvement = ((time1 - time2) / time1) * 100
            logger.info(f"Performance improvement: {improvement:.1f}%")
        
        # Test 3: Multiple requests
        logger.info("Test 3: Multiple requests to test cache effectiveness")
        symbols = [TEST_SYMBOL, "EURUSD", "GBPUSD"]
        timeframes = ["M1", "M5", "H1"]
        
        start_time = time.time()
        for symbol in symbols:
            for timeframe in timeframes:
                cached_client.fetch_data_with_cache(symbol, timeframe, start_pos=0, end_pos=50)
        total_time = time.time() - start_time
        
        logger.info(f"Multiple requests took {total_time:.4f} seconds")
        
        # Get cache statistics
        stats = cached_client.get_cache_stats()
        logger.info(f"Cache hit rate: {stats['hit_rate']:.2%}")
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Cache hits: {stats['cache_hits']}")
        logger.info(f"Cache misses: {stats['cache_misses']}")
        
        return cached_client
        
    except Exception as e:
        logger.error(f"Error in caching performance demonstration: {e}")
        return None


def demonstrate_cache_invalidation():
    """Demonstrate cache invalidation strategies."""
    logger.info("=== Demonstrating Cache Invalidation ===")
    
    try:
        # Initialize components
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=[TEST_SYMBOL])
        
        # Create cache with short TTL for demonstration
        cache_config = create_cache_config(
            cache_type="memory",
            ttl_hours=1  # Very short TTL (1 hour)
        )
        cache = DataCache(cache_config)
        
        cached_client = CachedMT5Client(mt5_client, cache)
        
        # Test 1: Cache data
        logger.info("Test 1: Caching data")
        data1 = cached_client.fetch_data_with_cache(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=10)
        logger.info(f"Data cached: {'SUCCESS' if data1 is not None else 'FAILED'}")
        
        # Test 2: Retrieve from cache
        logger.info("Test 2: Retrieving from cache")
        data2 = cached_client.fetch_data_with_cache(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=10)
        logger.info(f"Data retrieved: {'SUCCESS' if data2 is not None else 'FAILED'}")
        
        # Test 3: Wait for expiration
        logger.info("Test 3: Waiting for cache expiration...")
        time.sleep(5)
        
        # Test 4: Try to retrieve expired data
        logger.info("Test 4: Retrieving expired data")
        data3 = cached_client.fetch_data_with_cache(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=10)
        logger.info(f"Expired data retrieved: {'SUCCESS' if data3 is not None else 'FAILED'}")
        
        # Test 5: Force refresh
        logger.info("Test 5: Force refresh")
        data4 = cached_client.fetch_data_with_cache(
            TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=0, end_pos=10, force_refresh=True
        )
        logger.info(f"Force refresh: {'SUCCESS' if data4 is not None else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Error in cache invalidation demonstration: {e}")


def demonstrate_cache_monitoring():
    """Demonstrate cache monitoring capabilities."""
    logger.info("=== Demonstrating Cache Monitoring ===")
    
    try:
        # Initialize components
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=[TEST_SYMBOL])
        
        cache_config = create_cache_config(
            cache_type="hybrid",
            max_memory_mb=10,
            max_disk_gb=1
        )
        cache = DataCache(cache_config)
        
        cached_client = CachedMT5Client(mt5_client, cache)
        
        # Perform various operations
        logger.info("Performing cache operations...")
        
        # Fetch different data types
        for i in range(5):
            cached_client.fetch_data_with_cache(TEST_SYMBOL, "M1", start_pos=0, end_pos=100)
            cached_client.fetch_data_with_cache(TEST_SYMBOL, "H1", start_pos=0, end_pos=50)
            cached_client.get_tick_with_cache(TEST_SYMBOL)
        
        # Get detailed statistics
        stats = cached_client.get_cache_stats()
        
        logger.info("Cache Performance Summary:")
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Cache Hits: {stats['cache_hits']}")
        logger.info(f"Cache Misses: {stats['cache_misses']}")
        logger.info(f"Hit Rate: {stats['hit_rate']:.2%}")
        
        # Get detailed cache statistics
        cache_stats = stats['cache_stats']
        
        logger.info("\nDetailed Cache Statistics:")
        logger.info("Memory Cache:")
        logger.info(f"  Hit Rate: {cache_stats['memory']['hit_rate']:.2%}")
        logger.info(f"  Items: {cache_stats['memory']['item_count']}")
        logger.info(f"  Size: {cache_stats['memory']['total_size'] / (1024*1024):.2f} MB")
        logger.info(f"  Evictions: {cache_stats['memory']['evictions']}")
        
        logger.info("Disk Cache:")
        logger.info(f"  Hit Rate: {cache_stats['disk']['hit_rate']:.2%}")
        logger.info(f"  Items: {cache_stats['disk']['item_count']}")
        logger.info(f"  Size: {cache_stats['disk']['total_size'] / (1024*1024):.2f} MB")
        
        # Get summary
        summary = get_cache_stats_summary(cache)
        logger.info("\n" + summary)
        
    except Exception as e:
        logger.error(f"Error in cache monitoring demonstration: {e}")


def main():
    """Run all caching demonstrations."""
    logger.info("Starting Data Caching Demonstrations")
    logger.info("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_caching_performance()
        time.sleep(2)
        
        demonstrate_cache_invalidation()
        time.sleep(2)
        
        demonstrate_cache_monitoring()
        
        logger.info("=" * 60)
        logger.info("All caching demonstrations completed!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 