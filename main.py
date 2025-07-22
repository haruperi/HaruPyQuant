from app.config.setup import *
from app.util.crash_recovery import get_recovery_manager, initialize_recovery_manager
import time
from app.data import *
from app.strategy.indicators import *
from app.strategy.naive_trend import NaiveTrendStrategy
from datetime import datetime, timedelta

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
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, demo=True)
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
    #test_mt5_client()

    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
    start_date = datetime(2020, 5, 18)
    end_date = datetime(2025, 7, 21)

    for symbol in FOREX_SYMBOLS:
        print(f"Processing symbol: {symbol}")

        # Get Data
        df = mt5_client.fetch_data(symbol, DEFAULT_TIMEFRAME, start_date=start_date, end_date=end_date)
        if df is None:
            print(f"Failed to fetch data for {symbol} with {DEFAULT_TIMEFRAME}, skipping...")
            continue
            
        df_H1 = mt5_client.fetch_data(symbol, "H1", start_date=start_date, end_date=end_date)
        if df_H1 is None:
            print(f"Failed to fetch H1 data for {symbol}, skipping...")
            continue
            
        df_core = mt5_client.fetch_data(symbol, CORE_TIMEFRAME, start_date=start_date-timedelta(days=ADR_PERIOD), end_date=end_date)
        if df_core is None:
            print(f"Failed to fetch {CORE_TIMEFRAME} data for {symbol}, skipping...")
            continue
            
        print(f"Successfully fetched data: {len(df)} rows for {DEFAULT_TIMEFRAME}, {len(df_H1)} rows for H1, {len(df_core)} rows for {CORE_TIMEFRAME}")

        # Get Targets Data using ADR with Core Timeframe and start_date being ADR_PERIOD days before start_date
        symbol_info = mt5_client.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {symbol}, skipping...")
            continue
            
        df_core = get_adr(df_core, symbol_info)
        if df_core is None or "SL" not in df_core.columns:
            print(f"Failed to calculate ADR for {symbol}, skipping...")
            continue
            
        df_core = df_core[["SL"]]

        # Get H1 Bias
        smc = SmartMoneyConcepts(mt5_client, symbol)
        df_H1 = smc.calculate_swingline(df_H1)
        if df_H1 is None or "swingline" not in df_H1.columns or "swingvalue" not in df_H1.columns:
            print(f"Failed to calculate swingline for H1 data of {symbol}, skipping...")
            continue
            
        df_H1 = df_H1[["swingline", "swingvalue"]]
        df_H1 = df_H1.rename(columns={"swingline": "swingline_H1", "swingvalue": "swingvalue_H1"})

        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        print(f"Processing symbol: {symbol}")
        print(f"Base currency: {base_currency}, Quote currency: {quote_currency}")

        # Read currency index data
        try:
            df_base = pd.read_csv(f"{DATA_DIR}/{base_currency}X.csv")
            df_quote = pd.read_csv(f"{DATA_DIR}/{quote_currency}X.csv")
            
            print(f"Loaded {base_currency}X.csv with {len(df_base)} rows")
            print(f"Loaded {quote_currency}X.csv with {len(df_quote)} rows")
            
            # Convert datetime column if needed
            if 'datetime' in df_base.columns:
                df_base['datetime'] = pd.to_datetime(df_base['datetime'])
                df_base.set_index('datetime', inplace=True)
            if 'datetime' in df_quote.columns:
                df_quote['datetime'] = pd.to_datetime(df_quote['datetime'])
                df_quote.set_index('datetime', inplace=True)
                
            df_base = smc.calculate_swingline(df_base)
            df_quote = smc.calculate_swingline(df_quote)

            df_base = df_base[["swingline", "swingvalue"]]
            df_quote = df_quote[["swingline", "swingvalue"]]

            df_base = df_base.rename(columns={"swingline": "swingline_base", "swingvalue": "swingvalue_base"})
            df_quote = df_quote.rename(columns={"swingline": "swingline_quote", "swingvalue": "swingvalue_quote"})
            
        except Exception as e:
            print(f"Error loading currency index data: {e}")
            # Create empty DataFrames as fallback
            df_base = pd.DataFrame(index=df.index, columns=["swingline_base", "swingvalue_base"])
            df_quote = pd.DataFrame(index=df.index, columns=["swingline_quote", "swingvalue_quote"])
            df_base.fillna(0, inplace=True)
            df_quote.fillna(0, inplace=True)


        # Merge df and df_core using the index (datetime index)
        df = df.merge(df_core, left_index=True, right_index=True, how='left')
        df = df.merge(df_H1, left_index=True, right_index=True, how='left')
        df = df.merge(df_base, left_index=True, right_index=True, how='left')
        df = df.merge(df_quote, left_index=True, right_index=True, how='left')

        df["SL"] = df["SL"].ffill()
        df["swingline_H1"] = df["swingline_H1"].ffill()
        df["swingvalue_H1"] = df["swingvalue_H1"].ffill()
        df["swingline_base"] = df["swingline_base"].ffill()
        df["swingvalue_base"] = df["swingvalue_base"].ffill()
        df["swingline_quote"] = df["swingline_quote"].ffill()
        df["swingvalue_quote"] = df["swingvalue_quote"].ffill()

        # Get Signal
        df = smc.calculate_swingline(df)
        if df is None or "swingline" not in df.columns or "swingvalue" not in df.columns:
            print(f"Failed to calculate final swingline for {symbol}, skipping...")
            continue

        # Generate Signal column based on the given conditions
        # Long: if close[-2] < swingvalue and close[-1] > swingvalue and swingline == -1 and swingline_H1 == 1 and swingline_base == 1 and swingline_quote == -1 then Signal = 1
        # Short: if close[-2] > swingvalue and close[-1] < swingvalue and swingline == 1 and swingline_H1 == -1 and swingline_base == -1 and swingline_quote == 1 then Signal = -1

        df["Signal"] = 0  # default to 0

        # Debug: Check data availability
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns available: {list(df.columns)}")
        print(f"Sample of swingline values: {df['swingline'].value_counts().head()}")
        print(f"Sample of swingline_H1 values: {df['swingline_H1'].value_counts().head()}")
        print(f"Sample of swingline_base values: {df['swingline_base'].value_counts().head()}")
        print(f"Sample of swingline_quote values: {df['swingline_quote'].value_counts().head()}")

        # Check for NaN values
        print(f"NaN values in swingline: {df['swingline'].isna().sum()}")
        print(f"NaN values in swingline_H1: {df['swingline_H1'].isna().sum()}")
        print(f"NaN values in swingline_base: {df['swingline_base'].isna().sum()}")
        print(f"NaN values in swingline_quote: {df['swingline_quote'].isna().sum()}")

        long_cond = (
            (df["Close"].shift(1) < df["swingvalue"]) &
            (df["Close"] > df["swingvalue"]) &
            (df["swingline"] == -1) &
            (df["swingline_H1"] == 1) &
            (df["swingline_base"] == 1) &
            (df["swingline_quote"] == -1)
        )

        short_cond = (
            (df["Close"].shift(1) > df["swingvalue"]) &
            (df["Close"] < df["swingvalue"]) &
            (df["swingline"] == 1) &
            (df["swingline_H1"] == -1) &
            (df["swingline_base"] == -1) &
            (df["swingline_quote"] == 1)
        )

        print(f"Long condition matches: {long_cond.sum()}")
        print(f"Short condition matches: {short_cond.sum()}")

        df.loc[long_cond, "Signal"] = 1
        df.loc[short_cond, "Signal"] = -1

        signal_df = df[df["Signal"]!=0]
        print(f"Total signals generated: {len(signal_df)}")
        if len(signal_df) > 0:
            print("Signal details:")
            print(signal_df[["Close", "swingline", "swingline_H1", "swingline_base", "swingline_quote", "Signal"]].head())
        else:
            print("No signals generated. Let's try a simpler condition...")
            
            # Try a simpler condition for debugging
            simple_long = (
                (df["swingline"] == -1) &
                (df["swingline_H1"] == 1)
            )
            simple_short = (
                (df["swingline"] == 1) &
                (df["swingline_H1"] == -1)
            )
            
            print(f"Simple long condition matches: {simple_long.sum()}")
            print(f"Simple short condition matches: {simple_short.sum()}")
            
            df.loc[simple_long, "Signal"] = 1
            df.loc[simple_short, "Signal"] = -1
            
            simple_signals = df[df["Signal"]!=0]
            print(f"Simple signals generated: {len(simple_signals)}")
            if len(simple_signals) > 0:
                print("Simple signal details:")
                print(simple_signals[["Close", "swingline", "swingline_H1", "Signal"]].head())

        signal_df.to_csv(f"{DATA_DIR}/{symbol}-signal.csv")
        df.to_csv(f"{DATA_DIR}/{symbol}-data.csv")




























  
    # if df is not None:
    #     strategy = NaiveTrendStrategy(parameters={"fast_ema_period": 12, "slow_ema_period": 24, "bias_ema_period": 72})
    #     signals = strategy.get_signals(df)
    #     print(signals)

    #     smc = SmartMoneyConcepts(mt5_client, "EURUSD")
    #     df = smc.run_smc(df)
    #     print(df)
    # else:
    #     print("Failed to fetch data")
    #  