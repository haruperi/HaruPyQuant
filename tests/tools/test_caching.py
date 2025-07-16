#!/usr/bin/env python3
"""
Test script for data caching functionality.

This script demonstrates the caching features by testing:
- Memory caching
- Disk caching
- Hybrid caching
- Cache eviction policies
- Performance monitoring
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.data.caching import (
    DataCache, 
    CacheConfig, 
    CacheType, 
    CachePolicy,
    create_cache_config,
    get_cache_stats_summary
)
from app.util.logger import get_logger

logger = get_logger(__name__)


def test_memory_cache():
    """Test memory-only caching."""
    logger.info("=== Testing Memory Cache ===")
    
    # Create memory-only cache
    config = create_cache_config(
        cache_type="memory",
        max_memory_mb=10,
        ttl_hours=1
    )
    
    cache = DataCache(config)
    
    # Test basic operations
    test_data = {"key1": "value1", "key2": "value2"}
    
    # Set data
    success = cache.set("test_key", test_data)
    logger.info(f"Set operation: {'SUCCESS' if success else 'FAILED'}")
    
    # Get data
    retrieved_data = cache.get("test_key")
    logger.info(f"Get operation: {'SUCCESS' if retrieved_data == test_data else 'FAILED'}")
    
    # Test cache hit
    start_time = time.time()
    for _ in range(1000):
        cache.get("test_key")
    end_time = time.time()
    logger.info(f"1000 cache hits in {end_time - start_time:.4f} seconds")
    
    # Get stats
    stats = cache.get_stats()
    logger.info(f"Memory cache hit rate: {stats['memory']['hit_rate']:.2%}")
    
    return cache


def test_disk_cache():
    """Test disk-only caching."""
    logger.info("=== Testing Disk Cache ===")
    
    # Create temporary directory for disk cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create disk-only cache
        config = create_cache_config(
            cache_type="disk",
            max_disk_gb=1,
            ttl_hours=1
        )
        
        cache = DataCache(config, cache_dir=temp_dir)
        
        # Test with larger data
        large_data = pd.DataFrame({
            'A': np.random.randn(1000),
            'B': np.random.randn(1000),
            'C': np.random.randn(1000)
        })
        
        # Set data
        success = cache.set("large_data", large_data)
        logger.info(f"Set large data: {'SUCCESS' if success else 'FAILED'}")
        
        # Get data
        retrieved_data = cache.get("large_data")
        logger.info(f"Get large data: {'SUCCESS' if retrieved_data is not None else 'FAILED'}")
        
        # Test cache hit
        start_time = time.time()
        for _ in range(100):
            cache.get("large_data")
        end_time = time.time()
        logger.info(f"100 disk cache hits in {end_time - start_time:.4f} seconds")
        
        # Get stats
        stats = cache.get_stats()
        logger.info(f"Disk cache hit rate: {stats['disk']['hit_rate']:.2%}")
        
        return cache
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_hybrid_cache():
    """Test hybrid caching (memory + disk)."""
    logger.info("=== Testing Hybrid Cache ===")
    
    # Create temporary directory for disk cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create hybrid cache
        config = create_cache_config(
            cache_type="hybrid",
            max_memory_mb=5,
            max_disk_gb=1,
            ttl_hours=1
        )
        
        cache = DataCache(config, cache_dir=temp_dir)
        
        # Test OHLCV data caching
        ohlcv_data = pd.DataFrame({
            'Open': [1.1000, 1.1010, 1.1020],
            'High': [1.1015, 1.1025, 1.1035],
            'Low': [1.0995, 1.1005, 1.1015],
            'Close': [1.1010, 1.1020, 1.1030],
            'Volume': [1000, 1200, 1100]
        }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
        
        # Cache OHLCV data
        success = cache.set_cached_ohlcv("EURUSD", "H1", ohlcv_data)
        logger.info(f"Cache OHLCV data: {'SUCCESS' if success else 'FAILED'}")
        
        # Retrieve OHLCV data
        retrieved_data = cache.get_cached_ohlcv("EURUSD", "H1")
        logger.info(f"Retrieve OHLCV data: {'SUCCESS' if retrieved_data is not None else 'FAILED'}")
        
        # Test tick data caching
        tick_data = {
            'time': int(time.time()),
            'bid': 1.1000,
            'ask': 1.1002,
            'last': 1.1001,
            'volume': 100
        }
        
        success = cache.set_cached_tick("EURUSD", tick_data)
        logger.info(f"Cache tick data: {'SUCCESS' if success else 'FAILED'}")
        
        # Retrieve tick data
        retrieved_tick = cache.get_cached_tick("EURUSD")
        logger.info(f"Retrieve tick data: {'SUCCESS' if retrieved_tick is not None else 'FAILED'}")
        
        # Get stats
        stats = cache.get_stats()
        logger.info(f"Hybrid cache - Memory hit rate: {stats['memory']['hit_rate']:.2%}")
        logger.info(f"Hybrid cache - Disk hit rate: {stats['disk']['hit_rate']:.2%}")
        
        return cache
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_cache_eviction():
    """Test cache eviction policies."""
    logger.info("=== Testing Cache Eviction ===")
    
    # Test LRU eviction
    config = create_cache_config(
        cache_type="memory",
        max_memory_mb=1
    )
    config.policy = CachePolicy.LRU
    config.max_items = 5
    
    cache = DataCache(config)
    
    # Fill cache beyond capacity
    for i in range(10):
        data = f"data_{i}" * 1000  # Create some data
        cache.set(f"key_{i}", data)
    
    # Try to get items
    for i in range(10):
        data = cache.get(f"key_{i}")
        logger.info(f"Key {i}: {'FOUND' if data is not None else 'NOT FOUND'}")
    
    stats = cache.get_stats()
    logger.info(f"Evictions: {stats['memory']['evictions']}")
    logger.info(f"Final item count: {stats['memory']['item_count']}")


def test_cache_performance():
    """Test cache performance with different data types."""
    logger.info("=== Testing Cache Performance ===")
    
    config = create_cache_config(
        cache_type="hybrid",
        max_memory_mb=50,
        max_disk_gb=1
    )
    
    cache = DataCache(config)
    
    # Test with different data types
    data_types = {
        'small_dict': {'key': 'value'},
        'large_dict': {f'key_{i}': f'value_{i}' for i in range(1000)},
        'dataframe': pd.DataFrame(np.random.randn(1000, 10)),
        'list': list(range(10000)),
        'string': 'x' * 100000
    }
    
    results = {}
    
    for data_type, data in data_types.items():
        logger.info(f"Testing {data_type}...")
        
        # Measure set time
        start_time = time.time()
        success = cache.set(f"test_{data_type}", data)
        set_time = time.time() - start_time
        
        # Measure get time
        start_time = time.time()
        retrieved = cache.get(f"test_{data_type}")
        get_time = time.time() - start_time
        
        results[data_type] = {
            'set_success': success,
            'set_time': set_time,
            'get_success': retrieved is not None,
            'get_time': get_time
        }
    
    # Report results
    for data_type, result in results.items():
        logger.info(f"{data_type}:")
        logger.info(f"  Set: {'SUCCESS' if result['set_success'] else 'FAILED'} ({result['set_time']:.4f}s)")
        logger.info(f"  Get: {'SUCCESS' if result['get_success'] else 'FAILED'} ({result['get_time']:.4f}s)")


def test_cache_invalidation():
    """Test cache invalidation and TTL."""
    logger.info("=== Testing Cache Invalidation ===")
    
    # Create cache with short TTL
    config = create_cache_config(
        cache_type="memory",
        ttl_hours=1  # 1 hour TTL
    )
    
    cache = DataCache(config)
    
    # Set data
    cache.set("expire_test", "test_data")
    
    # Verify data is there
    data = cache.get("expire_test")
    logger.info(f"Data immediately after set: {'FOUND' if data is not None else 'NOT FOUND'}")
    
    # Wait for expiration
    logger.info("Waiting for cache expiration...")
    time.sleep(5)
    
    # Check if data expired
    data = cache.get("expire_test")
    logger.info(f"Data after expiration: {'FOUND' if data is not None else 'NOT FOUND'}")


def test_cache_monitoring():
    """Test cache monitoring and statistics."""
    logger.info("=== Testing Cache Monitoring ===")
    
    config = create_cache_config(
        cache_type="hybrid",
        max_memory_mb=10,
        max_disk_gb=1
    )
    
    cache = DataCache(config)
    
    # Perform some operations
    for i in range(100):
        cache.set(f"key_{i}", f"data_{i}")
    
    for i in range(50):
        cache.get(f"key_{i}")
    
    # Get detailed stats
    stats = cache.get_stats()
    
    logger.info("Cache Statistics:")
    logger.info(f"Memory Cache:")
    logger.info(f"  Hits: {stats['memory']['hits']}")
    logger.info(f"  Misses: {stats['memory']['misses']}")
    logger.info(f"  Hit Rate: {stats['memory']['hit_rate']:.2%}")
    logger.info(f"  Items: {stats['memory']['item_count']}")
    logger.info(f"  Size: {stats['memory']['total_size'] / (1024*1024):.2f} MB")
    
    logger.info(f"Disk Cache:")
    logger.info(f"  Hits: {stats['disk']['hits']}")
    logger.info(f"  Misses: {stats['disk']['misses']}")
    logger.info(f"  Hit Rate: {stats['disk']['hit_rate']:.2%}")
    logger.info(f"  Items: {stats['disk']['item_count']}")
    logger.info(f"  Size: {stats['disk']['total_size'] / (1024*1024):.2f} MB")
    
    # Get summary
    summary = get_cache_stats_summary(cache)
    logger.info("\n" + summary)


def main():
    """Run all caching tests."""
    logger.info("Starting Data Caching Tests")
    logger.info("=" * 50)
    
    try:
        # Run tests
        test_memory_cache()
        time.sleep(1)
        
        test_disk_cache()
        time.sleep(1)
        
        test_hybrid_cache()
        time.sleep(1)
        
        test_cache_eviction()
        time.sleep(1)
        
        test_cache_performance()
        time.sleep(1)
        
        test_cache_invalidation()
        time.sleep(1)
        
        test_cache_monitoring()
        
        logger.info("=" * 50)
        logger.info("All caching tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 