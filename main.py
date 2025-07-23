from app.config.setup import *
from app.util.crash_recovery import get_recovery_manager, initialize_recovery_manager
from app.live_trading import LiveTrader, LiveTradingConfig, StrategyConfig, RiskLevel
import time
import signal
import sys
import threading
import os
from app.data import *
from app.strategy.indicators import *
from app.strategy.naive_trend import NaiveTrendStrategy
from datetime import datetime, timedelta

logger = get_logger(__name__)

# Global shutdown flag
shutdown_requested = False
shutdown_event = threading.Event()
SHUTDOWN_FILE = "shutdown_requested.txt"

def signal_handler(signum, frame):
    """Global signal handler for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_requested = True
    shutdown_event.set()
    
    # Force exit after a short delay if graceful shutdown doesn't work
    def force_exit():
        time.sleep(5)  # Wait 5 seconds for graceful shutdown
        logger.warning("Force exiting due to timeout...")
        sys.exit(1)
    
    # Start force exit thread
    exit_thread = threading.Thread(target=force_exit, daemon=True)
    exit_thread.start()

def check_shutdown_file():
    """Check if shutdown file exists."""
    return os.path.exists(SHUTDOWN_FILE)

def create_shutdown_file():
    """Create shutdown file to request shutdown."""
    with open(SHUTDOWN_FILE, 'w') as f:
        f.write(f"Shutdown requested at {datetime.now()}\n")

def remove_shutdown_file():
    """Remove shutdown file."""
    if os.path.exists(SHUTDOWN_FILE):
        os.remove(SHUTDOWN_FILE)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_live_trading_config() -> LiveTradingConfig:
    """Create live trading configuration."""
    config = LiveTradingConfig()
    
    # Set basic configuration
    config.mode = config.mode.DEMO  # Use demo mode for safety
    config.account_name = "Demo Account"
    config.broker_name = "MT5"
    
    # Configure Swing Trend Momentum strategy
    swing_strategy_config = StrategyConfig(
        name="SwingTrendMomentum",
        enabled=True,
        symbols=["USDJPY", "EURUSD", "GBPUSD"],  # Add more symbols as needed
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


def initialize_system():
    """Initialize the trading system with crash recovery."""
    logger.info("Initializing HaruPyQuant trading system...")
    
    # Initialize crash recovery manager
    recovery_manager = initialize_recovery_manager(
        state_file="system_state.json",
        max_restarts=5,
        restart_delay=30,
        health_check_interval=60
    )
    
    # Register cleanup callbacks
    recovery_manager.register_cleanup_callback(cleanup_trading_system)
    recovery_manager.register_recovery_callback(recover_trading_system)
    
    # Start health monitoring
    recovery_manager.start_health_monitoring()
    
    # Update system status
    recovery_manager.state.status = "running"
    recovery_manager.state.active_connections = 0
    recovery_manager.state.active_positions = 0
    
    logger.info("Trading system initialized successfully")
    return recovery_manager


def cleanup_trading_system():
    """Cleanup trading system resources."""
    logger.info("Cleaning up trading system resources...")
    
    try:
        # Close any open connections
        # Close any open positions
        # Save any pending data
        # Close database connections
        # Stop any running threads
        
        logger.info("Trading system cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during trading system cleanup: {e}")


def recover_trading_system():
    """Recover trading system after a crash."""
    logger.info("Recovering trading system...")
    
    try:
        # Reload configuration
        # Reconnect to data sources
        # Restore any saved state
        # Reinitialize components
        
        logger.info("Trading system recovery completed")
        
    except Exception as e:
        logger.error(f"Error during trading system recovery: {e}")


def run_live_trading():
    """Run the live trading system."""
    logger.info("Starting live trading system...")
    
    try:
        # Create live trading configuration
        config = create_live_trading_config()
        
        # Create and start live trader
        live_trader = LiveTrader(config)
        
        # Start the live trading system
        if not live_trader.start():
            logger.error("Failed to start live trading system")
            return False
        
        logger.info("Live trading system started successfully")
        
        # Keep the system running
        try:
            # Remove any existing shutdown file
            remove_shutdown_file()
            
            while live_trader.get_status().value in ['running', 'paused'] and not shutdown_requested:
                # Use event.wait instead of time.sleep for better interrupt handling
                if shutdown_event.wait(timeout=1):
                    logger.info("Shutdown event triggered")
                    break
                
                # Check for shutdown file (backup method)
                if check_shutdown_file():
                    logger.info("Shutdown file detected, shutting down...")
                    break
                
                # Log status periodically (every 10 seconds)
                if int(time.time()) % 10 == 0:
                    stats = live_trader.get_statistics()
                    logger.info(f"Live Trading Status: {stats.status.value}, "
                              f"Signals: {stats.total_signals}, "
                              f"Trades: {stats.total_trades}, "
                              f"PnL: ${stats.total_pnl:.2f}, "
                              f"Positions: {stats.active_positions}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal in main loop, shutting down...")
        
        # Stop the live trading system
        logger.info("Stopping live trading system...")
        
        # Set a timeout for stopping
        stop_timeout = 30  # seconds
        start_time = time.time()
        
        live_trader.stop()
        
        # Check if stop took too long
        stop_duration = time.time() - start_time
        if stop_duration > stop_timeout:
            logger.warning(f"Stop operation took {stop_duration:.1f}s (longer than {stop_timeout}s timeout)")
        
        logger.info("Live trading system stopped")
        
        # Clean up shutdown file
        remove_shutdown_file()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live trading system: {e}", exc_info=True)
        return False


def main():
    """
    Main function to run the HaruPyQuant application with crash recovery.
    """
    recovery_manager = None
    live_trader = None
    
    try:
        # Initialize the system
        recovery_manager = initialize_system()
        
        logger.info("HaruPyQuant application started successfully")
        
        # Main application loop
        with recovery_manager.exception_handler("main_loop"):
            # Run live trading system
            success = run_live_trading()
            
            if not success:
                logger.error("Live trading system failed")
                return
        
        logger.info("HaruPyQuant application finished normally")
    
    # TODO: Add sending email and telegram to admin if the application is crashed
    # TODO: Test the crash recovery system with a simulated crash when app is done
    except KeyboardInterrupt:
        logger.info("Received interrupt signal in main function")
        # Ensure live trader is stopped if it exists
        if live_trader:
            live_trader.stop()
    except SystemExit:
        logger.info("System exit requested")
        if live_trader:
            live_trader.stop()
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        if recovery_manager:
            recovery_manager._handle_exception(e, "main")
    finally:
        if recovery_manager:
            recovery_manager.graceful_shutdown()


if __name__ == "__main__":
    main() 


  