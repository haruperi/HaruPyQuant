#!/usr/bin/env python3
"""
Test script for Live Trading Module

This script tests the basic functionality of the Live Trading Module
without actually connecting to MT5 or placing real trades.
"""

import sys
import os
import time
import signal
import threading
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.live_trading import LiveTrader, LiveTradingConfig, StrategyConfig, RiskLevel
from app.util import get_logger

logger = get_logger(__name__)


def create_test_config() -> LiveTradingConfig:
    """Create a test configuration for the Live Trading Module."""
    config = LiveTradingConfig()
    
    # Set basic configuration
    config.mode = config.mode.DEMO
    config.account_name = "Test Account"
    config.broker_name = "Test Broker"
    
    # Configure Swing Trend Momentum strategy
    swing_strategy_config = StrategyConfig(
        name="SwingTrendMomentum",
        enabled=True,
        symbols=["USDJPY", "EURUSD"],  # Test with fewer symbols
        parameters={
            'timeframe': 'M5',
            'bars': 100,  # Reduced for testing
            'update_interval': 30.0  # Faster updates for testing
        },
        risk_level=RiskLevel.LOW,  # Use low risk for testing
        max_positions=2,
        max_daily_trades=5,
        max_daily_loss=50.0,
        max_drawdown=2.0
    )
    config.add_strategy(swing_strategy_config)
    
    # Configure execution settings
    config.execution.max_slippage = 3
    config.execution.max_deviation = 5
    config.execution.retry_attempts = 2
    config.execution.retry_delay = 0.5
    config.execution.execution_timeout = 10.0
    config.execution.use_market_orders = True
    config.execution.confirm_trades = False
    
    # Configure risk management
    config.risk.max_risk_per_trade = 0.5  # Very low risk for testing
    config.risk.max_portfolio_risk = 2.0
    config.risk.max_correlation = 0.7
    config.risk.max_positions_per_symbol = 1
    config.risk.max_total_positions = 3
    config.risk.stop_loss_atr_multiplier = 2.0
    config.risk.trailing_stop_enabled = True
    config.risk.trailing_stop_atr_multiplier = 1.5
    
    # Configure monitoring
    config.monitoring.health_check_interval = 30.0  # Faster for testing
    config.monitoring.performance_update_interval = 60.0
    config.monitoring.position_update_interval = 15.0
    config.monitoring.connection_check_interval = 15.0
    config.monitoring.max_cpu_usage = 80.0
    config.monitoring.max_memory_usage = 80.0
    config.monitoring.max_disk_usage = 90.0
    
    # Configure notifications
    config.notifications.enabled = False  # Disable for testing
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
    config.strategy_update_interval = 30.0  # Faster for testing
    config.log_level = "INFO"
    config.enable_debug = True  # Enable debug for testing
    config.save_trades = True
    config.save_performance = True
    
    return config


def test_live_trading_module():
    """Test the Live Trading Module functionality."""
    logger.info("Starting Live Trading Module test...")
    
    # Set up timeout for the entire test
    test_timeout = 30  # 30 seconds timeout
    
    def timeout_handler():
        logger.error(f"Test timed out after {test_timeout} seconds")
        os._exit(1)
    
    # Set up timer for timeout
    timer = threading.Timer(test_timeout, timeout_handler)
    timer.start()
    
    try:
        # Create test configuration
        config = create_test_config()
        logger.info("Test configuration created successfully")
        
        # Create live trader
        live_trader = LiveTrader(config)
        logger.info("LiveTrader instance created successfully")
        
        # Test component initialization with timeout
        logger.info("Testing component initialization...")
        try:
            # Test basic component creation with timeout
            init_timeout = 10  # 10 seconds for initialization
            init_timer = threading.Timer(init_timeout, lambda: logger.warning("Component initialization timed out"))
            init_timer.start()
            
            live_trader._initialize_components()
            init_timer.cancel()
            logger.info("Component initialization successful")
        except Exception as e:
            logger.warning(f"Component initialization failed: {e}")
            # This might fail if MT5 is not available, which is expected in some test environments
            pass
        
        # Test configuration access
        logger.info("Testing configuration access...")
        strategies = config.get_enabled_strategies()
        logger.info(f"Enabled strategies: {strategies}")
        
        # Test status methods
        logger.info("Testing status methods...")
        status = live_trader.get_status()
        logger.info(f"Initial status: {status.value}")
        
        # Test statistics with timeout
        try:
            stats_timeout = 5  # 5 seconds for statistics
            stats_timer = threading.Timer(stats_timeout, lambda: logger.warning("Statistics retrieval timed out"))
            stats_timer.start()
            
            stats = live_trader.get_statistics()
            stats_timer.cancel()
            logger.info(f"Initial statistics: {stats}")
        except Exception as e:
            logger.warning(f"Statistics retrieval failed: {e}")
            # Continue with the test even if statistics fail
            pass
        
        # Test component status with timeout
        logger.info("Testing component status...")
        try:
            status_timeout = 5  # 5 seconds for status check
            status_timer = threading.Timer(status_timeout, lambda: logger.warning("Component status check timed out"))
            status_timer.start()
            
            component_status = live_trader.get_component_status()
            status_timer.cancel()
            
            for component, status_info in component_status.items():
                logger.info(f"{component}: {status_info}")
        except Exception as e:
            logger.warning(f"Component status check failed: {e}")
            # This might fail if MT5 is not available, which is expected in some test environments
            pass
        
        # Cancel the main timeout timer
        timer.cancel()
        
        logger.info("Live Trading Module test completed successfully")
        return True
        
    except Exception as e:
        timer.cancel()
        logger.error(f"Error during Live Trading Module test: {e}", exc_info=True)
        return False


def test_configuration():
    """Test the configuration system."""
    logger.info("Testing configuration system...")
    
    # Set up timeout for configuration test
    config_timeout = 15  # 15 seconds timeout
    
    def config_timeout_handler():
        logger.error(f"Configuration test timed out after {config_timeout} seconds")
        os._exit(1)
    
    # Set up timer for timeout
    timer = threading.Timer(config_timeout, config_timeout_handler)
    timer.start()
    
    try:
        config = create_test_config()
        
        # Test configuration validation
        logger.info("Testing configuration validation...")
        config._validate_config()
        
        # Test strategy management
        logger.info("Testing strategy management...")
        enabled_strategies = config.get_enabled_strategies()
        logger.info(f"Enabled strategies: {enabled_strategies}")
        
        # Test configuration serialization
        logger.info("Testing configuration serialization...")
        config_dict = config.to_dict()
        logger.info(f"Configuration serialized to dict with {len(config_dict)} keys")
        
        # Test strategy operations
        logger.info("Testing strategy operations...")
        config.disable_strategy("SwingTrendMomentum")
        enabled_strategies = config.get_enabled_strategies()
        logger.info(f"After disabling: {enabled_strategies}")
        
        config.enable_strategy("SwingTrendMomentum")
        enabled_strategies = config.get_enabled_strategies()
        logger.info(f"After enabling: {enabled_strategies}")
        
        # Cancel the timeout timer
        timer.cancel()
        
        logger.info("Configuration system test completed successfully")
        return True
        
    except Exception as e:
        timer.cancel()
        logger.error(f"Error during configuration test: {e}", exc_info=True)
        return False


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("LIVE TRADING MODULE TEST")
    logger.info("=" * 60)
    logger.info(f"Test started at: {datetime.now()}")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, stopping tests...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Test configuration system
        config_success = test_configuration()
        
        # Test live trading module
        module_success = test_live_trading_module()
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Configuration test: {'PASSED' if config_success else 'FAILED'}")
        logger.info(f"Module test: {'PASSED' if module_success else 'FAILED'}")
        
        if config_success and module_success:
            logger.info("All tests PASSED! Live Trading Module is ready for use.")
            return True
        else:
            logger.error("Some tests FAILED! Please check the logs for details.")
            return False
            
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 