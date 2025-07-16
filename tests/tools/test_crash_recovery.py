#!/usr/bin/env python3
"""
Test script for crash recovery mechanisms.

This script demonstrates the crash recovery features by simulating
various failure scenarios and showing how the system handles them.
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.util.crash_recovery import CrashRecoveryManager, get_recovery_manager
from app.util.logger import get_logger

logger = get_logger(__name__)


def test_basic_recovery():
    """Test basic crash recovery functionality."""
    logger.info("=== Testing Basic Crash Recovery ===")
    
    # Initialize recovery manager
    recovery_manager = CrashRecoveryManager(
        state_file="test_system_state.json",
        max_restarts=3,
        restart_delay=2,
        health_check_interval=5
    )
    
    # Register test callbacks
    recovery_manager.register_recovery_callback(test_recovery_callback)
    recovery_manager.register_cleanup_callback(test_cleanup_callback)
    
    # Start health monitoring
    recovery_manager.start_health_monitoring()
    
    logger.info("Recovery manager initialized")
    logger.info(f"System status: {recovery_manager.state.status}")
    
    # Simulate some work
    for i in range(5):
        logger.info(f"Working... {i+1}/5")
        time.sleep(1)
    
    # Test exception handling
    logger.info("Testing exception handling...")
    try:
        with recovery_manager.exception_handler("test_operation"):
            # Simulate an error
            raise ValueError("Test error for recovery demonstration")
    except ValueError:
        logger.info("Exception caught and handled by recovery manager")
    
    # Check system info
    system_info = recovery_manager.get_system_info()
    logger.info(f"System info: {system_info}")
    
    # Cleanup
    recovery_manager.graceful_shutdown()
    logger.info("Basic recovery test completed")


def test_recovery_callback():
    """Test recovery callback functionality."""
    logger.info("Recovery callback executed")
    time.sleep(1)  # Simulate recovery work


def test_cleanup_callback():
    """Test cleanup callback functionality."""
    logger.info("Cleanup callback executed")
    time.sleep(1)  # Simulate cleanup work


def test_health_monitoring():
    """Test health monitoring functionality."""
    logger.info("=== Testing Health Monitoring ===")
    
    recovery_manager = CrashRecoveryManager(
        state_file="test_health_state.json",
        health_check_interval=2
    )
    
    recovery_manager.start_health_monitoring()
    
    # Let health monitoring run for a while
    logger.info("Health monitoring started, running for 10 seconds...")
    time.sleep(10)
    
    # Get system info
    system_info = recovery_manager.get_system_info()
    logger.info(f"System info after health monitoring: {system_info}")
    
    recovery_manager.graceful_shutdown()
    logger.info("Health monitoring test completed")


def test_signal_handling():
    """Test signal handling functionality."""
    logger.info("=== Testing Signal Handling ===")
    
    recovery_manager = CrashRecoveryManager(
        state_file="test_signal_state.json"
    )
    
    recovery_manager.start_health_monitoring()
    
    logger.info("Signal handling test started")
    logger.info("Press Ctrl+C to test signal handling...")
    
    try:
        # Keep running until interrupted
        while recovery_manager.state.status == "running":
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    
    logger.info("Signal handling test completed")


def test_state_persistence():
    """Test state persistence functionality."""
    logger.info("=== Testing State Persistence ===")
    
    state_file = "test_persistence_state.json"
    
    # Create first instance
    recovery_manager1 = CrashRecoveryManager(state_file=state_file)
    recovery_manager1.state.status = "running"
    recovery_manager1.state.active_connections = 5
    recovery_manager1.state.active_positions = 2
    recovery_manager1._save_state()
    
    logger.info(f"Saved state: {recovery_manager1.state}")
    
    # Create second instance (should load state)
    recovery_manager2 = CrashRecoveryManager(state_file=state_file)
    
    logger.info(f"Loaded state: {recovery_manager2.state}")
    
    # Verify state was loaded correctly
    assert recovery_manager2.state.status == "running"
    assert recovery_manager2.state.active_connections == 5
    assert recovery_manager2.state.active_positions == 2
    
    logger.info("State persistence test completed")
    
    # Cleanup
    if Path(state_file).exists():
        Path(state_file).unlink()


def test_memory_pressure():
    """Test memory pressure handling."""
    logger.info("=== Testing Memory Pressure Handling ===")
    
    recovery_manager = CrashRecoveryManager(
        state_file="test_memory_state.json",
        health_check_interval=1
    )
    
    recovery_manager.start_health_monitoring()
    
    # Simulate memory pressure
    logger.info("Simulating memory pressure...")
    large_list = []
    
    try:
        for i in range(1000000):  # Allocate memory
            large_list.append(f"data_{i}" * 100)
            if i % 100000 == 0:
                logger.info(f"Allocated {i} items")
                time.sleep(0.1)
    except MemoryError:
        logger.warning("Memory error occurred")
    finally:
        del large_list  # Free memory
    
    # Let health monitoring detect the issue
    time.sleep(5)
    
    system_info = recovery_manager.get_system_info()
    logger.info(f"System info after memory pressure: {system_info}")
    
    recovery_manager.graceful_shutdown()
    logger.info("Memory pressure test completed")


def test_concurrent_operations():
    """Test concurrent operations with recovery manager."""
    logger.info("=== Testing Concurrent Operations ===")
    
    recovery_manager = CrashRecoveryManager(
        state_file="test_concurrent_state.json",
        health_check_interval=1
    )
    
    recovery_manager.start_health_monitoring()
    
    # Start multiple threads
    threads = []
    
    def worker_thread(thread_id):
        """Worker thread function."""
        try:
            for i in range(10):
                with recovery_manager.exception_handler(f"worker_{thread_id}"):
                    logger.info(f"Thread {thread_id} working... {i+1}/10")
                    time.sleep(0.5)
                    
                    # Simulate occasional errors
                    if i == 5 and thread_id == 1:
                        raise RuntimeError(f"Simulated error in thread {thread_id}")
        except Exception as e:
            logger.error(f"Thread {thread_id} error: {e}")
    
    # Start worker threads
    for i in range(3):
        thread = threading.Thread(target=worker_thread, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    system_info = recovery_manager.get_system_info()
    logger.info(f"System info after concurrent operations: {system_info}")
    
    recovery_manager.graceful_shutdown()
    logger.info("Concurrent operations test completed")


def main():
    """Run all crash recovery tests."""
    logger.info("Starting Crash Recovery Tests")
    logger.info("=" * 50)
    
    try:
        # Run tests
        test_basic_recovery()
        time.sleep(2)
        
        test_health_monitoring()
        time.sleep(2)
        
        test_state_persistence()
        time.sleep(2)
        
        test_memory_pressure()
        time.sleep(2)
        
        test_concurrent_operations()
        time.sleep(2)
        
        logger.info("=" * 50)
        logger.info("All crash recovery tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 