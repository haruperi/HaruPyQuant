# Live Trading Module

## Overview

The Live Trading Module is a comprehensive system for real-time automated trading that integrates all the components of the HaruPyQuant platform. It provides a complete solution for live trading with the SwingTrendMomentum strategy, including risk management, system monitoring, and notification capabilities.

## Architecture

The Live Trading Module consists of the following key components:

### Core Components

1. **LiveTrader** - Main orchestrator that coordinates all other components
2. **ExecutionEngine** - Handles trade execution and order management
3. **StrategyRunner** - Manages strategy execution and signal generation
4. **PositionManager** - Tracks and manages open positions
5. **RiskMonitor** - Monitors portfolio risk and generates alerts
6. **SystemMonitor** - Monitors system health and resources
7. **PerformanceTracker** - Tracks trading performance metrics
8. **TradingScheduler** - Manages trading hours and market sessions

### Configuration

The module uses a comprehensive configuration system (`LiveTradingConfig`) that includes:

- **Strategy Configuration** - Settings for each trading strategy
- **Execution Configuration** - Trade execution parameters
- **Risk Configuration** - Risk management settings
- **Monitoring Configuration** - System monitoring parameters
- **Notification Configuration** - Alert and notification settings
- **Schedule Configuration** - Trading hours and market sessions

## Features

### Real-time Execution
- Continuous market data monitoring
- Automated signal generation and execution
- Order retry and error handling
- Slippage and deviation management

### Risk Management
- Position sizing based on account balance
- Portfolio-level risk monitoring
- Correlation analysis
- Maximum drawdown protection
- Trailing stop management

### System Monitoring
- CPU, memory, and disk usage monitoring
- Connection health checks
- Performance metrics tracking
- Automatic alert generation

### Notification System
- Email notifications for critical events
- Telegram bot integration
- SMS notifications (optional)
- Configurable notification levels

### Trading Schedule
- Market hours management
- Weekend and holiday trading control
- Timezone support
- Market open/close event handling

## Usage

### Basic Setup

```python
from app.live_trading import LiveTrader, LiveTradingConfig, StrategyConfig, RiskLevel

# Create configuration
config = LiveTradingConfig()
config.mode = config.mode.DEMO
config.account_name = "Demo Account"

# Configure strategy
strategy_config = StrategyConfig(
    name="SwingTrendMomentum",
    enabled=True,
    symbols=["USDJPY", "EURUSD"],
    parameters={'timeframe': 'M5', 'bars': 300},
    risk_level=RiskLevel.MEDIUM
)
config.add_strategy(strategy_config)

# Create and start live trader
live_trader = LiveTrader(config)
live_trader.start()
```

### Configuration Options

#### Strategy Configuration
```python
strategy_config = StrategyConfig(
    name="SwingTrendMomentum",
    enabled=True,
    symbols=["USDJPY", "EURUSD", "GBPUSD"],
    parameters={
        'timeframe': 'M5',
        'bars': 300,
        'update_interval': 60.0
    },
    risk_level=RiskLevel.MEDIUM,
    max_positions=3,
    max_daily_trades=10,
    max_daily_loss=100.0,
    max_drawdown=5.0
)
```

#### Risk Configuration
```python
config.risk.max_risk_per_trade = 1.0  # 1% risk per trade
config.risk.max_portfolio_risk = 5.0  # 5% max portfolio risk
config.risk.max_correlation = 0.7
config.risk.max_positions_per_symbol = 1
config.risk.max_total_positions = 10
config.risk.stop_loss_atr_multiplier = 2.0
config.risk.trailing_stop_enabled = True
config.risk.trailing_stop_atr_multiplier = 1.5
```

#### Execution Configuration
```python
config.execution.max_slippage = 3
config.execution.max_deviation = 5
config.execution.retry_attempts = 3
config.execution.retry_delay = 1.0
config.execution.execution_timeout = 30.0
config.execution.use_market_orders = True
config.execution.confirm_trades = False
```

#### Monitoring Configuration
```python
config.monitoring.health_check_interval = 60.0
config.monitoring.performance_update_interval = 300.0
config.monitoring.position_update_interval = 30.0
config.monitoring.connection_check_interval = 30.0
config.monitoring.max_cpu_usage = 80.0
config.monitoring.max_memory_usage = 80.0
config.monitoring.max_disk_usage = 90.0
```

#### Notification Configuration
```python
config.notifications.enabled = True
config.notifications.services = ["email", "telegram"]
config.notifications.trade_notifications = True
config.notifications.error_notifications = True
config.notifications.performance_notifications = True
config.notifications.system_notifications = True
config.notifications.notification_levels = ["WARNING", "ERROR", "CRITICAL"]
```

### Running the System

```python
# Start the live trading system
if live_trader.start():
    print("Live trading system started successfully")
    
    # Monitor the system
    while live_trader.get_status().value in ['running', 'paused']:
        stats = live_trader.get_statistics()
        print(f"Status: {stats.status.value}, PnL: ${stats.total_pnl:.2f}")
        time.sleep(10)
    
    # Stop the system
    live_trader.stop()
```

## Integration with Main Application

The Live Trading Module is integrated into the main application through `main.py`:

```python
def run_live_trading():
    """Run the live trading system."""
    config = create_live_trading_config()
    live_trader = LiveTrader(config)
    
    if live_trader.start():
        while live_trader.get_status().value in ['running', 'paused']:
            time.sleep(10)
            stats = live_trader.get_statistics()
            logger.info(f"Live Trading Status: {stats.status.value}")
        
        live_trader.stop()
        return True
    return False
```

## Strategy Integration

The module is specifically designed to work with the SwingTrendMomentum strategy:

### Strategy Features
- Multi-timeframe analysis (M5 and H1)
- Smart Money Concepts (SMC) integration
- Swing line and swing value calculations
- Trend strength analysis
- Signal generation with confidence levels

### Strategy Configuration
```python
swing_strategy_config = StrategyConfig(
    name="SwingTrendMomentum",
    enabled=True,
    symbols=["USDJPY", "EURUSD", "GBPUSD"],
    parameters={
        'timeframe': 'M5',
        'bars': 300,
        'update_interval': 60.0
    },
    risk_level=RiskLevel.MEDIUM
)
```

## Error Handling and Recovery

The module includes comprehensive error handling:

### Crash Recovery
- Integration with the existing crash recovery system
- Automatic restart capabilities
- State persistence across restarts
- Health monitoring and alerts

### Error Handling
- Connection failure recovery
- Order execution retry logic
- Strategy error isolation
- Graceful degradation

### Monitoring and Alerts
- Real-time system health monitoring
- Performance degradation detection
- Resource usage alerts
- Trading performance tracking

## Testing

The module includes comprehensive testing capabilities:

### Unit Tests
```bash
python scripts/test_live_trading.py
```

### Example Usage
```bash
python scripts/live_trading_example.py
```

## Configuration Files

The module uses the existing configuration system:

- `config.ini` - Main configuration file
- `config.ini.example` - Example configuration template

## Dependencies

The module depends on the following components:

- **MT5Client** - MetaTrader 5 connection
- **Trader** - Core trading functionality
- **RiskManager** - Risk management
- **NotificationManager** - Notification system
- **SwingTrendMomentumStrategy** - Trading strategy

## Performance Considerations

### Optimization
- Efficient data processing
- Minimal memory usage
- Fast signal generation
- Optimized order execution

### Monitoring
- Real-time performance tracking
- Resource usage monitoring
- Latency measurement
- Throughput analysis

## Security

### Risk Management
- Position size limits
- Portfolio risk controls
- Maximum drawdown protection
- Correlation analysis

### System Security
- Secure credential management
- Connection encryption
- Access control
- Audit logging

## Troubleshooting

### Common Issues

1. **MT5 Connection Issues**
   - Verify MT5 terminal is running
   - Check configuration file path
   - Ensure API access is enabled

2. **Strategy Errors**
   - Check symbol availability
   - Verify data feed
   - Review strategy parameters

3. **Performance Issues**
   - Monitor system resources
   - Check update intervals
   - Review logging levels

### Debug Mode

Enable debug mode for detailed logging:

```python
config.enable_debug = True
config.log_level = "DEBUG"
```

## Future Enhancements

### Planned Features
- Multi-account support
- Advanced risk models
- Machine learning integration
- Web-based monitoring dashboard
- Mobile notifications

### Extensibility
- Plugin architecture
- Custom strategy support
- Third-party integrations
- API endpoints

## Conclusion

The Live Trading Module provides a complete, production-ready solution for automated trading with comprehensive risk management, monitoring, and notification capabilities. It integrates seamlessly with the existing HaruPyQuant platform and provides a solid foundation for live trading operations. 