from app.config.setup import *
from app.util.crash_recovery import get_recovery_manager, initialize_recovery_manager
import time
from app.data import *

logger = get_logger(__name__)


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


def main():
    """
    Main function to run the HaruPyQuant application with crash recovery.
    """
    recovery_manager = None
    
    try:
        # Initialize the system
        recovery_manager = initialize_system()
        
        logger.info("HaruPyQuant application started successfully")
        
        # Main application loop
        with recovery_manager.exception_handler("main_loop"):
            # TODO: Add main trading logic here
            # For now, just keep the application running
            while recovery_manager.state.status == "running":
                # Simulate some work
                time.sleep(1)
                
                # Check if shutdown is requested
                if recovery_manager.is_shutting_down:
                    break
        
        logger.info("HaruPyQuant application finished normally")
    
    # TODO: Add sending email and telegrame to admin if the application is crashed
    # TODO: Test the crash recovery system with a simulated crash when app is done
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        if recovery_manager:
            recovery_manager._handle_exception(e, "main")
    finally:
        if recovery_manager:
            recovery_manager.graceful_shutdown()



def test_mt5_client():
    try:
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS)
        df = mt5_client.fetch_data(TEST_SYMBOL, DEFAULT_TIMEFRAME, start_pos=START_POS, end_pos=END_POS)
        
        if df is not None and validate_ohlcv_data(df):
            logger.info(f"Successfully fetched {len(df)} rows of data")
            print("DataFrame:")
            print(df.head())





           
        else:
            logger.error("Failed to fetch data - DataFrame is None")
            
    except Exception as e:
        logger.error(f"Error in test_mt5_client: {e}")
        print(f"Error: {e}")








if __name__ == "__main__":
    #main() 
    test_mt5_client()