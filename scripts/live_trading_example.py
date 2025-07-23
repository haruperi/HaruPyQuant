#!/usr/bin/env python3
"""
Live Trading Module Example

This script demonstrates how to use the Live Trading Module
with the SwingTrendMomentum strategy.
"""

import sys
import os
import time
import signal
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.live_trading import LiveTrader, LiveTradingConfig, StrategyConfig, RiskLevel
from app.util import get_logger

logger = get_logger(__name__)


def create_demo_config() -> LiveTradingConfig:
    """Create a demo configuration for live trading."""
    config = LiveTradingConfig()
    
    # Set basic configuration
    config.mode = config.mode.DEMO
    config.account_name = "Demo Account"
    config.broker_name = "MT5"
    
    # Configure Swing Trend Momentum strategy
    swing_strategy_config = StrategyConfig(
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
    config.add_strategy(swing_strategy_config)
    
    # Configure execution settings
    config.execution.max_slippage = 3
    config.execution.max_deviation = 5
    config.execution.retry_attempts = 3
    config.execution.retry_delay = 1.0
    config.execution.execution_timeout = 30.0
    config.execution.use_market_orders = True
    config.execution.confirm_trades = False
    
    # Configure risk management
    config.risk.max_risk_per_trade = 1.0
    config.risk.max_portfolio_risk = 5.0
    config.risk.max_correlation = 0.7
    config.risk.max_positions_per_symbol = 1
    config.risk.max_total_positions = 10
    config.risk.stop_loss_atr_multiplier = 2.0
    config.risk.trailing_stop_enabled = True
    config.risk.trailing_stop_atr_multiplier = 1.5
    
    # Configure monitoring
    config.monitoring.health_check_interval = 60.0
    config.monitoring.performance_update_interval = 300.0
    config.monitoring.position_update_interval = 30.0
    config.monitoring.connection_check_interval = 30.0
    config.monitoring.max_cpu_usage = 80.0
    config.monitoring.max_memory_usage = 80.0
    config.monitoring.max_disk_usage = 90.0
    
    # Configure notifications
    config.notifications.enabled = True
    config.notifications.services = ["email", "telegram"]
    config.notifications.trade_notifications = True
    config.notifications.error_notifications = True
    config.notifications.performance_notifications = True
    config.notifications.system_notifications = True
    config.notifications.notification_levels = ["WARNING", "ERROR", "CRITICAL"]
    
    # Configure schedule
    config.schedule.enabled = True
    config.schedule.timezone = "UTC"
    config.schedule.weekend_trading = False
    config.schedule.holiday_trading = False
    
    # Configure data updates
    config.data_update_interval = 5.0
    config.strategy_update_interval = 60.0
    config.log_level = "INFO"
    config.enable_debug = False
    config.save_trades = True
    config.save_performance = True
    
    return config


def print_status_header():
    """Print status header."""
    print("\n" + "=" * 100)
    print(f"{'Time':<20} {'Status':<10} {'Signals':<8} {'Trades':<8} {'PnL':<12} {'Positions':<10} {'Health':<10} {'Market':<10}")
    print("=" * 100)


def print_status(stats):
    """Print current status."""
    time_str = stats.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    status_str = stats.status.value
    signals_str = str(stats.total_signals)
    trades_str = str(stats.total_trades)
    pnl_str = f"${stats.total_pnl:.2f}"
    positions_str = str(stats.active_positions)
    health_str = stats.system_health
    market_str = stats.market_status
    
    print(f"{time_str:<20} {status_str:<10} {signals_str:<8} {trades_str:<8} {pnl_str:<12} {positions_str:<10} {health_str:<10} {market_str:<10}")


def main():
    """Main function to run the live trading example."""
    logger.info("Starting Live Trading Module Example")
    
    # Create configuration
    config = create_demo_config()
    logger.info("Configuration created successfully")
    
    # Create live trader
    live_trader = LiveTrader(config)
    logger.info("LiveTrader created successfully")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping live trader...")
        live_trader.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the live trading system
        logger.info("Starting live trading system...")
        try:
            if not live_trader.start():
                logger.error("Failed to start live trading system")
                return False
        except Exception as e:
            logger.error(f"Error starting live trading system: {e}")
            logger.info("This might be due to MT5 not being available or configuration issues.")
            logger.info("Please ensure MT5 is running and config.ini is properly configured.")
            return False
        
        logger.info("Live trading system started successfully")
        
        # Print status header
        print_status_header()
        
        # Main monitoring loop with timeout
        update_count = 0
        max_runtime = 300  # 5 minutes maximum runtime for demo
        start_time = time.time()
        
        while live_trader.get_status().value in ['running', 'paused']:
            try:
                # Check if we've exceeded the maximum runtime
                if time.time() - start_time > max_runtime:
                    logger.info(f"Demo completed after {max_runtime} seconds")
                    break
                
                # Get current statistics
                stats = live_trader.get_statistics()
                
                # Print status every 10 seconds
                if update_count % 10 == 0:
                    print_status(stats)
                
                # Log detailed status every minute
                if update_count % 60 == 0:
                    logger.info(f"Live Trading Status: {stats.status.value}, "
                              f"Signals: {stats.total_signals}, "
                              f"Trades: {stats.total_trades}, "
                              f"PnL: ${stats.total_pnl:.2f}, "
                              f"Positions: {stats.active_positions}")
                
                update_count += 1
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
        
        # Stop the live trading system
        logger.info("Stopping live trading system...")
        live_trader.stop()
        logger.info("Live trading system stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live trading example: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Live Trading Example completed successfully")
    else:
        logger.error("Live Trading Example failed")
        sys.exit(1) 