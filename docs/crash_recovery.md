# Crash Recovery System

The HaruPyQuant crash recovery system provides comprehensive error handling, automatic recovery, and system monitoring to ensure the trading application remains stable and operational.

## Overview

The crash recovery system consists of two main components:

1. **CrashRecoveryManager** - Handles exception handling, state persistence, and recovery within the application
2. **WatchdogService** - External process monitor that can restart the application if it crashes

## Features

### CrashRecoveryManager Features

- **Exception Handling**: Automatic capture and logging of exceptions with context
- **State Persistence**: Save and restore system state across restarts
- **Health Monitoring**: Continuous monitoring of system resources (CPU, memory, disk)
- **Recovery Callbacks**: Custom recovery procedures for different components
- **Graceful Shutdown**: Proper cleanup of resources on shutdown
- **Signal Handling**: Respond to system signals (SIGINT, SIGTERM)
- **Automatic Restart**: Restart the application after crashes

### WatchdogService Features

- **Process Monitoring**: Monitor the main application process
- **Automatic Restart**: Restart the application if it crashes or becomes unresponsive
- **Resource Limits**: Restart if memory or CPU usage exceeds limits
- **Log Monitoring**: Monitor application output for errors
- **Configuration Management**: Persistent configuration storage
- **Status Reporting**: Get current status of the watchdog and application

## Usage

### Basic Usage

```python
from app.util.crash_recovery import get_recovery_manager

# Get the global recovery manager
recovery_manager = get_recovery_manager()

# Use exception handler for critical operations
with recovery_manager.exception_handler("trading_operation"):
    # Your trading logic here
    execute_trade()
```

### Advanced Usage

```python
from app.util.crash_recovery import initialize_recovery_manager

# Initialize with custom configuration
recovery_manager = initialize_recovery_manager(
    state_file="custom_state.json",
    max_restarts=10,
    restart_delay=60,
    health_check_interval=30
)

# Register custom callbacks
def my_recovery_callback():
    """Custom recovery procedure."""
    reconnect_to_broker()
    restore_positions()

def my_cleanup_callback():
    """Custom cleanup procedure."""
    close_all_positions()
    save_trading_state()

recovery_manager.register_recovery_callback(my_recovery_callback)
recovery_manager.register_cleanup_callback(my_cleanup_callback)

# Start health monitoring
recovery_manager.start_health_monitoring()
```

### Watchdog Service Usage

```bash
# Start watchdog service
python scripts/run_watchdog.py

# Start with custom configuration
python scripts/run_watchdog.py \
    --app-script main.py \
    --max-restarts 15 \
    --restart-delay 45 \
    --max-memory 2048 \
    --max-cpu 85

# Check status
python scripts/run_watchdog.py --status

# Stop watchdog service
python scripts/run_watchdog.py --stop
```

## Configuration

### CrashRecoveryManager Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_file` | "system_state.json" | File to persist system state |
| `max_restarts` | 5 | Maximum number of restart attempts |
| `restart_delay` | 30 | Delay between restart attempts (seconds) |
| `health_check_interval` | 60 | Interval for health checks (seconds) |

### WatchdogService Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `app_script` | "main.py" | Application script to monitor |
| `max_restarts` | 10 | Maximum restart attempts |
| `restart_delay` | 30 | Delay between restarts (seconds) |
| `health_check_interval` | 30 | Health check interval (seconds) |
| `max_memory_mb` | 1024 | Maximum memory usage (MB) |
| `max_cpu_percent` | 90 | Maximum CPU usage (%) |

## System State

The system state is persisted in JSON format and includes:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "status": "running",
  "uptime": 3600.5,
  "memory_usage": 512.3,
  "cpu_usage": 25.7,
  "active_connections": 2,
  "active_positions": 1,
  "last_error": null,
  "restart_count": 0,
  "max_restarts": 5,
  "recovery_attempts": 0,
  "max_recovery_attempts": 3
}
```

## Health Monitoring

The health monitoring system checks:

- **Memory Usage**: Warns if memory usage exceeds 80%
- **CPU Usage**: Warns if CPU usage exceeds 90%
- **Disk Space**: Warns if disk usage exceeds 90%
- **Process Status**: Monitors if the process is running and responsive

## Recovery Procedures

### Automatic Recovery

1. **Exception Detection**: Exceptions are caught and logged with full context
2. **State Update**: System state is updated to reflect the error
3. **Recovery Attempt**: Recovery callbacks are executed
4. **State Reset**: If recovery succeeds, error state is cleared

### Manual Recovery

```python
# Force recovery attempt
recovery_manager._attempt_recovery()

# Check recovery status
if recovery_manager.state.recovery_attempts > 0:
    logger.info(f"Recovery attempts: {recovery_manager.state.recovery_attempts}")
```

## Best Practices

### 1. Register Appropriate Callbacks

```python
# Register recovery callbacks for critical components
recovery_manager.register_recovery_callback(reconnect_database)
recovery_manager.register_recovery_callback(reconnect_broker)
recovery_manager.register_recovery_callback(restore_positions)

# Register cleanup callbacks for proper shutdown
recovery_manager.register_cleanup_callback(close_positions)
recovery_manager.register_cleanup_callback(save_state)
recovery_manager.register_cleanup_callback(close_connections)
```

### 2. Use Exception Handlers for Critical Operations

```python
# Wrap critical operations in exception handlers
with recovery_manager.exception_handler("order_execution"):
    place_order(symbol, side, quantity, price)

with recovery_manager.exception_handler("data_processing"):
    process_market_data(data)
```

### 3. Monitor System Resources

```python
# Get system information
system_info = recovery_manager.get_system_info()
logger.info(f"Memory usage: {system_info['memory_usage_mb']:.1f} MB")
logger.info(f"CPU usage: {system_info['cpu_percent']:.1f}%")
```

### 4. Configure Appropriate Limits

```python
# Set conservative limits for production
recovery_manager = initialize_recovery_manager(
    max_restarts=3,  # Fewer restarts in production
    restart_delay=60,  # Longer delay between restarts
    health_check_interval=30  # More frequent health checks
)
```

### 5. Use Watchdog for Production

```bash
# Start watchdog as a service
nohup python scripts/run_watchdog.py \
    --max-restarts 5 \
    --restart-delay 60 \
    --max-memory 2048 \
    --max-cpu 80 \
    > watchdog.log 2>&1 &
```

## Testing

Run the crash recovery tests:

```bash
# Run all tests
python tests/tools/test_crash_recovery.py

# Run specific test
python -c "
from tests.tools.test_crash_recovery import test_basic_recovery
test_basic_recovery()
"
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in your code
   - Increase `max_memory_mb` limit
   - Implement memory cleanup in recovery callbacks

2. **Frequent Restarts**
   - Check application logs for errors
   - Increase `restart_delay` to prevent rapid restarts
   - Implement proper error handling in your code

3. **State Not Persisting**
   - Check file permissions for state file
   - Verify disk space is available
   - Check for JSON serialization errors

4. **Watchdog Not Starting**
   - Check if another watchdog is already running
   - Verify the application script path
   - Check log files for errors

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Monitoring

Monitor the system in real-time:

```python
# Get real-time system info
while True:
    info = recovery_manager.get_system_info()
    print(f"Status: {info['status']}, Memory: {info['memory_usage_mb']:.1f}MB")
    time.sleep(5)
```

## Integration with Trading System

### Trading-Specific Recovery

```python
def trading_recovery_callback():
    """Recovery callback for trading system."""
    try:
        # Reconnect to MT5
        mt5_client.reconnect()
        
        # Restore position tracking
        position_manager.restore_positions()
        
        # Resume data feeds
        data_manager.resume_feeds()
        
        logger.info("Trading system recovery completed")
    except Exception as e:
        logger.error(f"Trading recovery failed: {e}")

def trading_cleanup_callback():
    """Cleanup callback for trading system."""
    try:
        # Close all positions
        position_manager.close_all_positions()
        
        # Save trading state
        trading_state.save()
        
        # Disconnect from MT5
        mt5_client.disconnect()
        
        logger.info("Trading system cleanup completed")
    except Exception as e:
        logger.error(f"Trading cleanup failed: {e}")

# Register callbacks
recovery_manager.register_recovery_callback(trading_recovery_callback)
recovery_manager.register_cleanup_callback(trading_cleanup_callback)
```

## Security Considerations

1. **State File Security**: Ensure state files are not accessible to unauthorized users
2. **Log Security**: Secure log files to prevent information leakage
3. **Process Isolation**: Run the watchdog with appropriate user permissions
4. **Network Security**: Secure any network connections used for monitoring

## Performance Impact

The crash recovery system has minimal performance impact:

- **Memory**: ~5-10 MB additional memory usage
- **CPU**: <1% CPU usage for health monitoring
- **Disk**: Minimal I/O for state persistence
- **Network**: No network overhead unless using remote monitoring

## Future Enhancements

1. **Remote Monitoring**: Web-based monitoring dashboard
2. **Alerting**: Email/SMS notifications for critical events
3. **Metrics**: Integration with monitoring systems (Prometheus, Grafana)
4. **Distributed Recovery**: Multi-node recovery coordination
5. **Machine Learning**: Predictive failure detection 