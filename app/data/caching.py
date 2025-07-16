"""
Efficient data caching mechanisms for HaruPyQuant.

This module provides:
- Memory-based caching for frequently accessed data
- Disk-based caching for historical data
- Cache invalidation strategies
- Performance monitoring and optimization
"""

import os
import json
import pickle
import hashlib
import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
import pandas as pd
from collections import OrderedDict, defaultdict
import logging

from app.util.logger import get_logger

logger = get_logger(__name__)


class CacheType(Enum):
    """Types of cache storage."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"   # Time To Live


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    cache_type: CacheType = CacheType.HYBRID
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_disk_size: int = 1024 * 1024 * 1024   # 1GB
    max_items: int = 10000
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    policy: CachePolicy = CachePolicy.LRU
    enable_compression: bool = True
    enable_monitoring: bool = True


@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    key: str
    data: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = datetime.now(timezone.utc)
        if self.size_bytes == 0:
            self.size_bytes = self._estimate_size()
    
    def _estimate_size(self) -> int:
        """Estimate the size of the cached data in bytes."""
        try:
            if isinstance(self.data, pd.DataFrame):
                return self.data.memory_usage(deep=True).sum()
            elif isinstance(self.data, (dict, list)):
                return len(pickle.dumps(self.data))
            elif isinstance(self.data, str):
                return len(self.data.encode('utf-8'))
            else:
                return len(pickle.dumps(self.data))
        except Exception:
            return 1024  # Default estimate
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache entry has expired."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() > ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc)


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.total_size = 0
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_changes': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired(self.config.ttl_seconds):
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.update_access()
                self.access_frequency[key] += 1
                self._update_access_order(key)
                
                self.stats['hits'] += 1
                return entry.data
            else:
                self.stats['misses'] += 1
                return None
    
    def set(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data in cache."""
        with self.lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size -= old_entry.size_bytes
            
            # Check if we need to evict items
            while (self.total_size + entry.size_bytes > self.config.max_memory_size or 
                   len(self.cache) >= self.config.max_items):
                if not self._evict_item():
                    return False  # Cannot make space
            
            # Add entry
            self.cache[key] = entry
            self.total_size += entry.size_bytes
            self._update_access_order(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete an item from cache."""
        with self.lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) 
                       if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'total_size': self.total_size,
                'item_count': len(self.cache),
                'max_size': self.config.max_memory_size,
                'max_items': self.config.max_items
            }
    
    def _evict_item(self) -> bool:
        """Evict an item based on the cache policy."""
        if not self.cache:
            return False
        
        if self.config.policy == CachePolicy.LRU:
            # Remove least recently used
            if self.access_order:
                key_to_remove = self.access_order[0]
            else:
                key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
        
        elif self.config.policy == CachePolicy.LFU:
            # Remove least frequently used
            key_to_remove = min(self.access_frequency.keys(), key=lambda k: self.access_frequency[k])
        
        elif self.config.policy == CachePolicy.FIFO:
            # Remove oldest
            key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        
        else:  # TTL
            # Remove expired items first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired(self.config.ttl_seconds)]
            if expired_keys:
                key_to_remove = expired_keys[0]
            else:
                key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        
        return self._remove_entry(key_to_remove)
    
    def _remove_entry(self, key: str) -> bool:
        """Remove an entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size -= entry.size_bytes
            del self.cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_frequency:
                del self.access_frequency[key]
            
            self.stats['evictions'] += 1
            return True
        return False
    
    def _update_access_order(self, key: str):
        """Update the access order for LRU policy."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items() 
                if entry.is_expired(self.config.ttl_seconds)
            ]
            for key in expired_keys:
                self._remove_entry(key)


class DiskCache:
    """Disk-based cache implementation using SQLite."""
    
    def __init__(self, config: CacheConfig, cache_dir: str = "cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'deletes': 0
        }
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_access TEXT,
                    size_bytes INTEGER,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON cache_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON cache_entries(last_access)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from disk cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, timestamp, access_count FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    data_blob, timestamp_str, access_count = row
                    
                    # Check if expired
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if (datetime.now(timezone.utc) - timestamp).total_seconds() > self.config.ttl_seconds:
                        self.delete(key)
                        self.stats['misses'] += 1
                        return None
                    
                    # Deserialize data
                    data = pickle.loads(data_blob)
                    
                    # Update access statistics
                    conn.execute(
                        "UPDATE cache_entries SET access_count = ?, last_access = ? WHERE key = ?",
                        (access_count + 1, datetime.now(timezone.utc).isoformat(), key)
                    )
                    
                    self.stats['hits'] += 1
                    return data
                else:
                    self.stats['misses'] += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data in disk cache."""
        try:
            # Serialize data
            data_blob = pickle.dumps(data)
            
            # Check disk space
            if self._get_total_size() + len(data_blob) > self.config.max_disk_size:
                self._cleanup_old_entries()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, timestamp, access_count, last_access, size_bytes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    key,
                    data_blob,
                    datetime.now(timezone.utc).isoformat(),
                    0,
                    datetime.now(timezone.utc).isoformat(),
                    len(data_blob),
                    json.dumps(metadata or {})
                ))
            
            self.stats['writes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete an item from disk cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.stats['deletes'] += 1
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")
            return False
    
    def clear(self):
        """Clear all cached data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                row = cursor.fetchone()
                item_count = row[0] or 0
                total_size = row[1] or 0
                
                hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) 
                           if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
                
                return {
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'hit_rate': hit_rate,
                    'writes': self.stats['writes'],
                    'deletes': self.stats['deletes'],
                    'total_size': total_size,
                    'item_count': item_count,
                    'max_size': self.config.max_disk_size
                }
        except Exception as e:
            logger.error(f"Error getting disk cache stats: {e}")
            return {}
    
    def _get_total_size(self) -> int:
        """Get total size of cached data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()
                return result[0] or 0
        except Exception:
            return 0
    
    def _cleanup_old_entries(self):
        """Remove old entries to free space."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove expired entries first
                conn.execute(
                    "DELETE FROM cache_entries WHERE datetime(timestamp) < datetime('now', '-{} seconds')".format(
                        self.config.ttl_seconds
                    )
                )
                
                # If still over limit, remove oldest entries
                if self._get_total_size() > self.config.max_disk_size:
                    conn.execute("""
                        DELETE FROM cache_entries WHERE key IN (
                            SELECT key FROM cache_entries 
                            ORDER BY last_access ASC 
                            LIMIT 100
                        )
                    """)
        except Exception as e:
            logger.error(f"Error cleaning up disk cache: {e}")


class DataCache:
    """
    Main data cache manager that combines memory and disk caching.
    
    Provides intelligent caching strategies for different types of data.
    """
    
    def __init__(self, config: CacheConfig, cache_dir: str = "cache"):
        self.config = config
        self.memory_cache = MemoryCache(config)
        self.disk_cache = DiskCache(config, cache_dir)
        
        # Cache key generators
        self.key_generators = {
            'ohlcv': self._generate_ohlcv_key,
            'tick': self._generate_tick_key,
            'symbol_info': self._generate_symbol_info_key,
            'account_info': self._generate_account_info_key
        }
        
        logger.info(f"Initialized DataCache with {config.cache_type.value} strategy")
    
    def get(self, key: str, cache_type: Optional[CacheType] = None) -> Optional[Any]:
        """Get data from cache."""
        cache_type = cache_type or self.config.cache_type
        
        if cache_type == CacheType.MEMORY:
            return self.memory_cache.get(key)
        elif cache_type == CacheType.DISK:
            return self.disk_cache.get(key)
        else:  # HYBRID
            # Try memory first, then disk
            data = self.memory_cache.get(key)
            if data is not None:
                return data
            
            data = self.disk_cache.get(key)
            if data is not None:
                # Promote to memory cache
                self.memory_cache.set(key, data)
                return data
            
            return None
    
    def set(self, key: str, data: Any, cache_type: Optional[CacheType] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data in cache."""
        cache_type = cache_type or self.config.cache_type
        
        if cache_type == CacheType.MEMORY:
            return self.memory_cache.set(key, data, metadata)
        elif cache_type == CacheType.DISK:
            return self.disk_cache.set(key, data, metadata)
        else:  # HYBRID
            # Store in both caches
            memory_success = self.memory_cache.set(key, data, metadata)
            disk_success = self.disk_cache.set(key, data, metadata)
            return memory_success and disk_success
    
    def delete(self, key: str, cache_type: Optional[CacheType] = None) -> bool:
        """Delete data from cache."""
        cache_type = cache_type or self.config.cache_type
        
        if cache_type == CacheType.MEMORY:
            return self.memory_cache.delete(key)
        elif cache_type == CacheType.DISK:
            return self.disk_cache.delete(key)
        else:  # HYBRID
            memory_success = self.memory_cache.delete(key)
            disk_success = self.disk_cache.delete(key)
            return memory_success or disk_success
    
    def clear(self, cache_type: Optional[CacheType] = None):
        """Clear cache."""
        cache_type = cache_type or self.config.cache_type
        
        if cache_type == CacheType.MEMORY:
            self.memory_cache.clear()
        elif cache_type == CacheType.DISK:
            self.disk_cache.clear()
        else:  # HYBRID
            self.memory_cache.clear()
            self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        return {
            'memory': memory_stats,
            'disk': disk_stats,
            'config': {
                'cache_type': self.config.cache_type.value,
                'policy': self.config.policy.value,
                'ttl_seconds': self.config.ttl_seconds
            }
        }
    
    # Cache key generators
    def _generate_ohlcv_key(self, symbol: str, timeframe: str, 
                           start_pos: Optional[int] = None, end_pos: Optional[int] = None,
                           start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
        """Generate cache key for OHLCV data."""
        components = ['ohlcv', symbol, timeframe]
        
        if start_pos is not None:
            components.extend(['pos', str(start_pos)])
        if end_pos is not None:
            components.extend(['end_pos', str(end_pos)])
        if start_date is not None:
            components.extend(['start', start_date.isoformat()])
        if end_date is not None:
            components.extend(['end', end_date.isoformat()])
        
        return hashlib.md5('_'.join(components).encode()).hexdigest()
    
    def _generate_tick_key(self, symbol: str) -> str:
        """Generate cache key for tick data."""
        return hashlib.md5(f'tick_{symbol}'.encode()).hexdigest()
    
    def _generate_symbol_info_key(self, symbol: str) -> str:
        """Generate cache key for symbol info."""
        return hashlib.md5(f'symbol_info_{symbol}'.encode()).hexdigest()
    
    def _generate_account_info_key(self) -> str:
        """Generate cache key for account info."""
        return hashlib.md5('account_info'.encode()).hexdigest()
    
    def get_cached_ohlcv(self, symbol: str, timeframe: str, **kwargs) -> Optional[pd.DataFrame]:
        """Get cached OHLCV data."""
        key = self._generate_ohlcv_key(symbol, timeframe, **kwargs)
        return self.get(key)
    
    def set_cached_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame, **kwargs) -> bool:
        """Cache OHLCV data."""
        key = self._generate_ohlcv_key(symbol, timeframe, **kwargs)
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_type': 'ohlcv',
            'shape': data.shape,
            'columns': list(data.columns)
        }
        return self.set(key, data, metadata=metadata)
    
    def get_cached_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached tick data."""
        key = self._generate_tick_key(symbol)
        return self.get(key)
    
    def set_cached_tick(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Cache tick data."""
        key = self._generate_tick_key(symbol)
        metadata = {
            'symbol': symbol,
            'data_type': 'tick',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        return self.set(key, data, metadata=metadata)


# Utility functions for cache management
def create_cache_config(cache_type: str = "hybrid", 
                       max_memory_mb: int = 100,
                       max_disk_gb: int = 1,
                       ttl_hours: int = 1) -> CacheConfig:
    """Create a cache configuration."""
    return CacheConfig(
        cache_type=CacheType(cache_type),
        max_memory_size=max_memory_mb * 1024 * 1024,
        max_disk_size=max_disk_gb * 1024 * 1024 * 1024,
        ttl_seconds=ttl_hours * 3600
    )


def get_cache_stats_summary(cache: DataCache) -> str:
    """Get a human-readable cache statistics summary."""
    stats = cache.get_stats()
    
    memory = stats['memory']
    disk = stats['disk']
    
    summary = f"""
Cache Statistics Summary:
=======================

Memory Cache:
  Hit Rate: {memory.get('hit_rate', 0):.2%}
  Items: {memory.get('item_count', 0)} / {memory.get('max_items', 0)}
  Size: {memory.get('total_size', 0) / (1024*1024):.1f} MB / {memory.get('max_size', 0) / (1024*1024):.1f} MB
  Hits: {memory.get('hits', 0)}, Misses: {memory.get('misses', 0)}
  Evictions: {memory.get('evictions', 0)}

Disk Cache:
  Hit Rate: {disk.get('hit_rate', 0):.2%}
  Items: {disk.get('item_count', 0)}
  Size: {disk.get('total_size', 0) / (1024*1024*1024):.2f} GB / {disk.get('max_size', 0) / (1024*1024*1024):.2f} GB
  Hits: {disk.get('hits', 0)}, Misses: {disk.get('misses', 0)}
  Writes: {disk.get('writes', 0)}, Deletes: {disk.get('deletes', 0)}

Configuration:
  Type: {stats['config']['cache_type']}
  Policy: {stats['config']['policy']}
  TTL: {stats['config']['ttl_seconds']} seconds
"""
    return summary 