# Data Caching System

This document describes the efficient data caching mechanisms in HaruPyQuant, which provide significant performance improvements for data retrieval operations.

## Overview

The caching system provides:
- **Memory Caching**: Fast in-memory storage for frequently accessed data
- **Disk Caching**: Persistent storage for larger datasets
- **Hybrid Caching**: Combination of memory and disk for optimal performance
- **Intelligent Eviction**: Multiple eviction policies (LRU, LFU, FIFO, TTL)
- **Performance Monitoring**: Comprehensive statistics and monitoring

## Architecture

### Core Components

1. **DataCache**: Main cache manager that coordinates memory and disk caches
2. **MemoryCache**: In-memory cache with configurable size limits
3. **DiskCache**: SQLite-based persistent cache
4. **CacheConfig**: Configuration for cache behavior
5. **CacheEntry**: Individual cache entry with metadata

### Cache Types

- **Memory**: Fastest access, limited by RAM
- **Disk**: Persistent storage, limited by disk space
- **Hybrid**: Best of both worlds (memory + disk)

### Eviction Policies

- **LRU**: Least Recently Used
- **LFU**: Least Frequently Used
- **FIFO**: First In, First Out
- **TTL**: Time To Live

## Quick Start

### Basic Usage

```python
from app.data import DataCache, create_cache_config

# Create cache configuration
config = create_cache_config(
    cache_type="hybrid",
    max_memory_mb=100,
    max_disk_gb=1,
    ttl_hours=1
)

# Initialize cache
cache = DataCache(config)

# Store data
cache.set("my_key", my_data)

# Retrieve data
data = cache.get("my_key")

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['memory']['hit_rate']:.2%}")
```

### MT5 Data Caching

```python
from app.data import DataCache, create_cache_config
from app.data.mt5_client import MT5Client

# Initialize components
mt5_client = MT5Client(config_path="config.ini")
cache = DataCache(create_cache_config("hybrid"))

# Cache OHLCV data
ohlcv_data = mt5_client.fetch_data("EURUSD", "H1", start_pos=0, end_pos=100)
cache.set_cached_ohlcv("EURUSD", "H1", ohlcv_data, start_pos=0, end_pos=100)

# Retrieve cached data
cached_data = cache.get_cached_ohlcv("EURUSD", "H1", start_pos=0, end_pos=100)

# Cache tick data
tick_data = mt5_client.get_tick("EURUSD")
cache.set_cached_tick("EURUSD", tick_data)

# Retrieve cached tick
cached_tick = cache.get_cached_tick("EURUSD")
```

## Detailed Usage

### Cache Configuration

```python
from app.data import CacheConfig, CacheType, CachePolicy

# Custom configuration
config = CacheConfig(
    cache_type=CacheType.HYBRID,
    max_memory_size=50 * 1024 * 1024,  # 50MB
    max_disk_size=2 * 1024 * 1024 * 1024,  # 2GB
    max_items=5000,
    ttl_seconds=7200,  # 2 hours
    cleanup_interval=600,  # 10 minutes
    policy=CachePolicy.LRU,
    enable_compression=True,
    enable_monitoring=True
)

cache = DataCache(config)
```

### Cache Operations

```python
# Basic operations
cache.set("key1", "value1")
data = cache.get("key1")
cache.delete("key1")

# With metadata
metadata = {
    'source': 'mt5',
    'symbol': 'EURUSD',
    'timeframe': 'H1',
    'timestamp': datetime.now().isoformat()
}
cache.set("ohlcv_data", data, metadata=metadata)

# Clear cache
cache.clear()  # Clear all
cache.clear(CacheType.MEMORY)  # Clear only memory
cache.clear(CacheType.DISK)    # Clear only disk
```

### OHLCV Data Caching

```python
# Cache OHLCV data with parameters
success = cache.set_cached_ohlcv(
    symbol="EURUSD",
    timeframe="H1",
    data=ohlcv_dataframe,
    start_pos=0,
    end_pos=100,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# Retrieve cached OHLCV data
cached_data = cache.get_cached_ohlcv(
    symbol="EURUSD",
    timeframe="H1",
    start_pos=0,
    end_pos=100,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
```

### Tick Data Caching

```python
# Cache tick data
tick_data = {
    'time': int(time.time()),
    'bid': 1.1000,
    'ask': 1.1002,
    'last': 1.1001,
    'volume': 100
}

success = cache.set_cached_tick("EURUSD", tick_data)
cached_tick = cache.get_cached_tick("EURUSD")
```

## Integration with MT5 Client

### CachedMT5Client Wrapper

```python
from app.data import DataCache, create_cache_config
from app.data.mt5_client import MT5Client

class CachedMT5Client:
    def __init__(self, mt5_client: MT5Client, cache: DataCache):
        self.mt5_client = mt5_client
        self.cache = cache
        self.request_count = 0
        self.cache_hit_count = 0
    
    def fetch_data_with_cache(self, symbol: str, timeframe: str = "D1", 
                             start_pos=None, end_pos=None, force_refresh=False):
        self.request_count += 1
        
        # Try cache first (unless force refresh)
        if not force_refresh:
            cached_data = self.cache.get_cached_ohlcv(
                symbol, timeframe, start_pos=start_pos, end_pos=end_pos
            )
            if cached_data is not None:
                self.cache_hit_count += 1
                return cached_data
        
        # Cache miss - fetch from MT5
        data = self.mt5_client.fetch_data(symbol, timeframe, start_pos, end_pos)
        
        if data is not None:
            # Cache the data
            self.cache.set_cached_ohlcv(symbol, timeframe, data, 
                                       start_pos=start_pos, end_pos=end_pos)
        
        return data
    
    def get_cache_stats(self):
        total_requests = self.request_count
        cache_hits = self.cache_hit_count
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': cache_hits,
            'cache_misses': total_requests - cache_hits,
            'hit_rate': hit_rate
        }

# Usage
mt5_client = MT5Client(config_path="config.ini")
cache = DataCache(create_cache_config("hybrid"))
cached_client = CachedMT5Client(mt5_client, cache)

# Fetch data (will use cache if available)
data = cached_client.fetch_data_with_cache("EURUSD", "H1", start_pos=0, end_pos=100)

# Get performance stats
stats = cached_client.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

## Performance Monitoring

### Cache Statistics

```python
# Get comprehensive statistics
stats = cache.get_stats()

# Memory cache stats
memory_stats = stats['memory']
print(f"Memory hit rate: {memory_stats['hit_rate']:.2%}")
print(f"Memory items: {memory_stats['item_count']}")
print(f"Memory size: {memory_stats['total_size'] / (1024*1024):.2f} MB")
print(f"Memory evictions: {memory_stats['evictions']}")

# Disk cache stats
disk_stats = stats['disk']
print(f"Disk hit rate: {disk_stats['hit_rate']:.2%}")
print(f"Disk items: {disk_stats['item_count']}")
print(f"Disk size: {disk_stats['total_size'] / (1024*1024*1024):.2f} GB")

# Configuration
config_stats = stats['config']
print(f"Cache type: {config_stats['cache_type']}")
print(f"Eviction policy: {config_stats['policy']}")
print(f"TTL: {config_stats['ttl_seconds']} seconds")
```

### Performance Summary

```python
from app.data import get_cache_stats_summary

# Get human-readable summary
summary = get_cache_stats_summary(cache)
print(summary)
```

## Cache Invalidation

### Time-Based Expiration

```python
# Create cache with short TTL for testing
config = create_cache_config(
    cache_type="memory",
    ttl_hours=0.001  # 3.6 seconds
)
cache = DataCache(config)

# Set data
cache.set("test_key", "test_data")

# Data is immediately available
data = cache.get("test_key")  # Returns "test_data"

# Wait for expiration
time.sleep(5)

# Data has expired
data = cache.get("test_key")  # Returns None
```

### Manual Invalidation

```python
# Delete specific items
cache.delete("specific_key")

# Clear entire cache
cache.clear()

# Clear specific cache type
cache.clear(CacheType.MEMORY)
cache.clear(CacheType.DISK)
```

### Force Refresh

```python
# Force refresh from source (ignore cache)
data = cached_client.fetch_data_with_cache(
    "EURUSD", "H1", start_pos=0, end_pos=100, force_refresh=True
)
```

## Best Practices

### Cache Configuration

```python
# For high-frequency trading
config = create_cache_config(
    cache_type="hybrid",
    max_memory_mb=200,      # Large memory cache
    max_disk_gb=5,          # Large disk cache
    ttl_hours=0.1,          # Short TTL for fresh data
    policy="lru"            # LRU for recent data
)

# For historical analysis
config = create_cache_config(
    cache_type="hybrid",
    max_memory_mb=50,       # Moderate memory
    max_disk_gb=10,         # Large disk for historical data
    ttl_hours=24,           # Long TTL for historical data
    policy="lfu"            # LFU for frequently used data
)

# For development/testing
config = create_cache_config(
    cache_type="memory",
    max_memory_mb=10,       # Small memory cache
    ttl_hours=1,            # Moderate TTL
    policy="fifo"           # Simple FIFO
)
```

### Key Generation

```python
# The cache automatically generates keys for OHLCV data
# Format: hash(ohlcv_symbol_timeframe_start_pos_end_pos_start_date_end_date)

# For custom data, use descriptive keys
cache.set(f"symbol_info_{symbol}", symbol_info)
cache.set(f"account_info_{account_id}", account_info)
cache.set(f"strategy_params_{strategy_name}", parameters)
```

### Memory Management

```python
# Monitor memory usage
stats = cache.get_stats()
memory_usage = stats['memory']['total_size'] / (1024*1024)
print(f"Memory usage: {memory_usage:.2f} MB")

# Clear cache if memory usage is high
if memory_usage > 80:  # 80% of max memory
    cache.clear(CacheType.MEMORY)
    print("Memory cache cleared due to high usage")
```

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   - Check TTL settings
   - Verify key generation
   - Monitor cache size limits

2. **High Memory Usage**
   - Reduce max_memory_size
   - Use disk cache for large data
   - Implement manual cleanup

3. **Slow Performance**
   - Use memory cache for frequently accessed data
   - Optimize key generation
   - Monitor eviction policy effectiveness

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('app.data.caching').setLevel(logging.DEBUG)

# Monitor cache operations
cache.set("debug_key", "debug_data")
data = cache.get("debug_key")
print(f"Debug data: {data}")

# Check cache status
stats = cache.get_stats()
print(f"Cache status: {stats}")
```

## Testing

### Running Tests

```bash
# Run caching tests
python tests/tools/test_caching.py

# Run caching example
python scripts/caching_example.py
```

### Test Coverage

The test suite covers:
- Memory caching
- Disk caching
- Hybrid caching
- Cache eviction policies
- Performance testing
- Cache invalidation
- Monitoring and statistics

## API Reference

### DataCache

- `get(key, cache_type=None) -> Optional[Any]`
- `set(key, data, cache_type=None, metadata=None) -> bool`
- `delete(key, cache_type=None) -> bool`
- `clear(cache_type=None)`
- `get_stats() -> Dict[str, Any]`
- `get_cached_ohlcv(symbol, timeframe, **kwargs) -> Optional[pd.DataFrame]`
- `set_cached_ohlcv(symbol, timeframe, data, **kwargs) -> bool`
- `get_cached_tick(symbol) -> Optional[Dict[str, Any]]`
- `set_cached_tick(symbol, data) -> bool`

### CacheConfig

- `cache_type: CacheType`
- `max_memory_size: int`
- `max_disk_size: int`
- `max_items: int`
- `ttl_seconds: int`
- `cleanup_interval: int`
- `policy: CachePolicy`
- `enable_compression: bool`
- `enable_monitoring: bool`

### Utility Functions

- `create_cache_config(cache_type, max_memory_mb, max_disk_gb, ttl_hours) -> CacheConfig`
- `get_cache_stats_summary(cache) -> str` 