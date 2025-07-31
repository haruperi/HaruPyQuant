#!/usr/bin/env python3
"""
Live Signals Bot

This script runs the SwingTrendMomentum strategy continuously, checking for trading signals
every 5 minutes at bar opening times and sending notifications when signals are found.

Features:
- Runs every 5 minutes at exact bar opening times
- Analyzes all configured forex symbols
- Sends notifications via Email, Telegram, and/or SMS when signals are found
- Provides startup, shutdown, and error notifications
- Handles different broker symbol formats

Notification Configuration:
To enable notifications, add the following sections to your config.ini file:

[EMAIL]
enabled = true
smtp_server = smtp.gmail.com
smtp_port = 587
username = your-email@gmail.com
password = your-app-password
use_tls = true
use_ssl = false
from_email = your-email@gmail.com
from_name = HaruPyQuant
recipients = recipient1@example.com,recipient2@example.com

[TELEGRAM]
enabled = true
token = your-bot-token
chat_ids = 123456789,-987654321
parse_mode = HTML
disable_web_page_preview = true
disable_notification = false
protect_content = false

[SMS]
enabled = false
account_sid = your-account-sid
auth_token = your-auth-token
from_number = +1234567890
recipients = +1234567890,+0987654321

[NOTIFICATIONS]
enable_all = true
default_levels = WARNING,ERROR,CRITICAL

Usage:
    python scripts/live_signals.py
"""

from app.strategy.swing_trend_momentum import SwingTrendMomentumStrategy
from app.data.mt5_client import MT5Client
from app.config.setup import *
from app.util import get_logger
from app.notifications import NotificationManager
from app.notifications.config import NotificationConfig, NotificationPresets
from app.notifications.manager import NotificationManagerConfig
from app.notifications.templates import NotificationTemplate
from app.strategy.trend_swingline_mtf import TrendSwinglineMTF
import time
from datetime import datetime, timedelta, timezone
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


def initialize_notification_manager() -> NotificationManager:
    """Initialize the notification manager with configuration from config.ini."""
    try:
        # Try to load configuration from config.ini
        config = NotificationConfig.from_ini(DEFAULT_CONFIG_PATH)
        
        manager_config = NotificationManagerConfig()
        
        if config.email_enabled:
            manager_config.email_config = config.get_email_config()
            logger.info("Email notifications enabled")
        
        if config.telegram_enabled:
            manager_config.telegram_config = config.get_telegram_config()
            logger.info("Telegram notifications enabled")
        
        if config.sms_enabled:
            manager_config.sms_config = config.get_sms_config()
            logger.info("SMS notifications enabled")
        
        manager_config.default_levels = config.get_default_levels()
        manager_config.enable_all = config.enable_all
        
        notification_manager = NotificationManager(manager_config)
        
        if notification_manager.notifiers:
            logger.info(f"Notification manager initialized with {len(notification_manager.notifiers)} services")
            return notification_manager
        else:
            logger.warning("No notification services configured, notifications will be disabled")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize notification manager: {e}")
        return None


def send_trading_signal_notification(notification_manager: NotificationManager, signal_data: dict, symbol: str):
    """Send a trading signal notification using the trading_signal template."""
    if not notification_manager:
        return
    
    try:
        # Initialize template system
        template_system = NotificationTemplate()
        
        # Format the signal data for notification
        action = "BUY" if signal_data.get("Signal") == 1 else "SELL"
        entry_price = signal_data.get("Price", "N/A")
        stop_loss = signal_data.get("SL Price", "N/A")
        take_profit = signal_data.get("TP Price", "N/A")
        lots = signal_data.get("Lots", "N/A")
        adr = signal_data.get("ADR", "N/A")
        current_var = signal_data.get('CurrVAR', 'N/A')
        proposed_var = signal_data.get('PropVAR', 'N/A')
        
        # Get pip values from signal_data (already calculated by the strategy)
        stop_loss_pips = signal_data.get("SL Pips", "N/A")
        take_profit_pips = signal_data.get("TP Pips", "N/A")
        
        # Calculate VAR difference percentage
        try:
            if current_var != "N/A" and proposed_var != "N/A":
                current_var_float = float(str(current_var).replace('$', '').replace(',', ''))
                proposed_var_float = float(str(proposed_var).replace('$', '').replace(',', ''))
                if current_var_float != 0:
                    var_difference = ((proposed_var_float - current_var_float) / current_var_float) * 100
                else:
                    var_difference = "N/A"
            else:
                var_difference = "N/A"
        except (ValueError, TypeError):
            var_difference = "N/A"
        
        # Format time consistently
        timestamp = signal_data.get('Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Prepare template variables
        template_vars = {
            'symbol': symbol,
            'signal_type': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_loss_pips': f"{stop_loss_pips:.1f}" if stop_loss_pips != "N/A" else "N/A",
            'take_profit': take_profit,
            'take_profit_pips': f"{take_profit_pips:.1f}" if take_profit_pips != "N/A" else "N/A",
            'lots': lots,
            'strategy': signal_data.get('Strategy Name', 'Swing Trend Momentum'),
            'strength': signal_data.get("Strength", "N/A"),  # Default value, could be calculated based on signal strength
            'adr': adr,
            'range': signal_data.get("Range", "N/A"),  # Get range from signal_data
            'current_var': current_var,
            'proposed_var': proposed_var,
            'var_difference': signal_data.get("DiffVAR", "N/A"),  # Get VAR difference from signal_data
            'timestamp': timestamp
        }
        
        # Render the trading_signal template
        notification_message = template_system.render("trading_signal", **template_vars)
        
        # Send the notification
        results = notification_manager.send_custom_message(
            title=notification_message.title,
            body=notification_message.body,
            level="INFO",
            services=["telegram", "email"]  # Send to both Telegram and Email if available
        )
        
        # Log the results
        for service, result in results.items():
            if result.success:
                logger.info(f"Signal notification sent successfully via {service}")
            else:
                logger.error(f"Failed to send signal notification via {service}: {result.error_message}")
                
    except Exception as e:
        logger.error(f"Error sending trading signal notification: {e}")


def countdown_to_next_bar() -> str:
    """
    Continuously counts down to the opening of the next bar based on the server time and interval.

    This function retrieves the trading server time, determines the next bar time,
    and continuously sleeps until the new bar time is reached, at which point a task
    (e.g., run_signal_analysis) can be executed.
    
    Args:
        interval_min (int): The time interval in minutes for the next bar's start. Default is 5.
        timeShift (int): The number of hours to adjust the server time. Default is 0.
        
    Returns:
        int: Seconds to wait until next bar opening
    """
    # Get the current server time
    mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)

    def get_server_time():
        tick = mt5_client.get_tick(TEST_SYMBOL)
        if tick is None:
            logger.error(f"Failed to get tick for {TEST_SYMBOL}")
            return None
        server_time = datetime.fromtimestamp(tick['time'])

        # Apply the time shift
        adjusted_time = server_time + timedelta(hours=TIME_SHIFT)
        return adjusted_time

    server_time = get_server_time()
    if server_time is None:
        logger.error("Failed to get server time")
        return "Failed to get server time."
    
    # Calculate the next bar time
    next_bar_time = server_time.replace(second=0, microsecond=0) + timedelta(minutes=INTERVAL_MINUTES)
    next_bar_time = next_bar_time - timedelta(minutes=server_time.minute % INTERVAL_MINUTES)

    # Format the time until the next bar
    logger.info(f"Next bar at: {next_bar_time}")

    while get_server_time() < next_bar_time:
        time.sleep(1)

    mt5_client.shutdown()
    
    return f"\n\n\nNew bar opened at {next_bar_time}. Running the task..."


def run_signal_analysis():
    """Run the signal analysis for all symbols."""
    
    logger.info(f"Starting signal analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize notification manager
    notification_manager = initialize_notification_manager()
    
    try:
    #     # Initialize secondary MT5 client for index data (broker 3 - Purple Trading)
    #     mt5_client_indices = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=INDEX_SYMBOLS, broker=3)

    #     # Check secondary MT5 connection for indices
    #     indices_account_info = mt5_client_indices.get_account_info()
    #     if indices_account_info is not None:
    #         logger.info("Indices MT5 initialized successfully")
    #         logger.info(f"Indices Login: {indices_account_info['login']} \tserver: {indices_account_info['server']}")

    #         # Store index dataframes in a dictionary
    #         index_dataframes = {}
    #         for index in INDEX_SYMBOLS:
    #             index_dataframes[index] = mt5_client_indices.fetch_data(index, "H1", start_pos=START_POS, end_pos=END_POS_HTF)
    #             logger.info(f"Fetched {index} data: {index_dataframes[index].shape if index_dataframes[index] is not None else 'None'}")

    #         if index_dataframes is not None:
    #             use_indices = True
    #         else:
    #             use_indices = False

    #     else:
    #         logger.warning(f"Failed to connect to indices MT5 terminal. Error code: {mt5_client_indices.mt5.last_error()}")
    #         logger.warning("Index data will not be available. Using main MT5 for all data.")
    #         index_dataframes = {}
    #         use_indices = False

    #     mt5_client_indices.shutdown()
    #     if mt5_client_indices._connected:
    #         logger.error("Failed to close indices MT5 connection")  
    #     else:
    #         logger.info("Indices MT5 connection closed successfully")
            

        # Initialize main MT5 client for trading data (broker 1 - Pepperstone)
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=FOREX_SYMBOLS, broker=BROKER)

        signals_found = 0
        
        for symbol in FOREX_SYMBOLS:
            try:
                symbol_info = mt5_client.get_symbol_info(symbol)
                if not symbol_info or not hasattr(symbol_info, "trade_tick_size"):
                    logger.error(f"Error: Invalid symbol_info for {symbol}. Skipping...")
                    continue
                
                data = mt5_client.fetch_data(symbol, "M5", start_pos=START_POS, end_pos=END_POS)
                dfH1 = mt5_client.fetch_data(symbol, "H1", start_pos=START_POS, end_pos=END_POS_HTF)

                if data is not None and dfH1 is not None:
  
                    strategy = TrendSwinglineMTF(mt5_client, symbol_info, parameters={})
                    trigger_signal, data = strategy.get_trigger_signal(data)
                    if trigger_signal != 0:
                        data = strategy.get_features(data, dfH1)
                        entry_signal, entry_time = strategy.get_entry_signal(data)

                        if entry_signal != 0:
                            df_core = mt5_client.fetch_data(symbol, CORE_TIMEFRAME, start_pos=START_POS, end_pos=END_POS_D1)

                            if df_core is None:
                                logger.error(f"Failed to fetch {CORE_TIMEFRAME} data for {symbol}, skipping...")
                                continue

                            str_message, data = strategy.get_trade_parameters(data, df_core, entry_signal, symbol_info)

                            if str_message is not None:
                                logger.info(f"SIGNAL FOUND: {str_message}")
                                signals_found += 1
                                send_trading_signal_notification(notification_manager, str_message, symbol)
                            else:
                                logger.info(f"No Signal for {symbol}, at {entry_time} skipping...")
                    else:
                        logger.info(f"No Trigger Signal for {symbol}, skipping...")
                        continue
                else:
                    if data is None:
                        logger.error(f"Error: Failed to fetch M5 data for {symbol}. Skipping...")
                    if dfH1 is None:
                        logger.error(f"Error: Failed to fetch H1 data for {symbol}. Skipping...")
                    continue
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue
        
        logger.info(f"Signal analysis completed. Found {signals_found} signals.")
        
        # # Send completion notification if signals were found
        # if signals_found > 0 and notification_manager:
        #     try:
        #         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #         results = notification_manager.send_custom_message(
        #             title=f"Analysis Complete - {signals_found} Signal(s) Found",
        #             body=f"""Signal Analysis Summary
        #             Analysis Time: {current_time} UTC
        #             Signals Found: {signals_found}
        #             Strategy: Swing Trend Momentum
        #             Next Analysis: In {INTERVAL_MINUTES} minutes

        #             HaruPyQuant Bot - Live Trading Signals""",
        #                                 level="INFO")
        #         for service, result in results.items():
        #             if result.success:
        #                 logger.info(f"Completion notification sent via {service}")
        #     except Exception as e:
        #         logger.error(f"Error sending completion notification: {e}")
        
        mt5_client.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"Error in signal analysis: {e}")
        
        # Send error notification
        if notification_manager:
            try:
                results = notification_manager.send_error_alert(
                    error_type="Signal Analysis Error",
                    message=str(e),
                    component="Live Signals Bot"
                )
                for service, result in results.items():
                    if result.success:
                        logger.info(f"Error notification sent via {service}")
            except Exception as notify_error:
                logger.error(f"Error sending error notification: {notify_error}")
        
        return False


def main():
    """Main function that runs the signal analysis continuously at bar intervals."""
    
    logger.info("Starting Live Signals Bot...")
    logger.info(f"Running every {INTERVAL_MINUTES} minutes at bar opening times")
    
    # Initialize notification manager for startup notification
    notification_manager = initialize_notification_manager()
    
    # Send startup notification
    if notification_manager:
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results = notification_manager.send_custom_message(
                title="HaruPyQuant Bot Started",
                body=f"""Live Signals Bot Status

                Status: Started Successfully
                Start Time: {current_time} UTC
                Analysis Interval: Every {INTERVAL_MINUTES} minutes
                Strategy: Swing Trend Momentum
                Symbols: All configured forex pairs

                Bot is now monitoring for trading signals""",
                                level="INFO"
                            )
            for service, result in results.items():
                if result.success:
                    logger.info(f"Startup notification sent via {service}")
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
    
    try:
        while True:
            # Calculate time until next bar
            new_bar_message = countdown_to_next_bar()
            logger.info(new_bar_message)

            # Run signal analysis
            success = run_signal_analysis()
            
            if not success:
                logger.warning("Signal analysis failed, will retry at next interval")
            
            logger.info("=" * 80)
            
    except KeyboardInterrupt:
        logger.info("Live Signals Bot stopped by user")
        
        # Send shutdown notification
        if notification_manager:
            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                results = notification_manager.send_custom_message(
                    title="HaruPyQuant Bot Stopped",
                    body=f"""Live Signals Bot Status

                    Status: Stopped by User
                    Stop Time: {current_time} UTC
                    Reason: Manual shutdown (Ctrl+C)

                    Bot monitoring has been stopped""",
                                        level="INFO"
                                    )
                for service, result in results.items():
                    if result.success:
                        logger.info(f"Shutdown notification sent via {service}")
            except Exception as e:
                logger.error(f"Error sending shutdown notification: {e}")
                
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        
        # Send error notification
        if notification_manager:
            try:
                results = notification_manager.send_error_alert(
                    error_type="Main Loop Error",
                    message=str(e),
                    component="Live Signals Bot"
                )
                for service, result in results.items():
                    if result.success:
                        logger.info(f"Error notification sent via {service}")
            except Exception as notify_error:
                logger.error(f"Error sending error notification: {notify_error}")
                
    finally:
        logger.info("Live Signals Bot shutdown complete")





if __name__ == "__main__":
    main()



