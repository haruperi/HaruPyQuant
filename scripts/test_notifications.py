#!/usr/bin/env python3
"""
Test script for notification functionality in live signals bot.

This script tests the notification system without running the full trading analysis.
"""

import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.notifications import NotificationManager
from app.notifications.config import NotificationConfig, NotificationPresets
from app.notifications.manager import NotificationManagerConfig
from app.config.setup import DEFAULT_CONFIG_PATH
from app.util import get_logger

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


def send_test_trading_signal_notification(notification_manager: NotificationManager):
    """Send a test trading signal notification."""
    if not notification_manager:
        logger.warning("No notification manager available")
        return
    
    try:
        # Create test signal data
        test_signal_data = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Strategy Name": "Swing Trend Momentum",
            "Symbol": "EURUSD",
            "Signal": 1,  # 1 for BUY, -1 for SELL
            "Price": 1.0850,
            "SL Price": 1.0800,
            "TP Price": 1.0950,
            "Lots": 0.1,
            "ADR": 85,
            "CurrVAR": "$150.00",
            "PropVAR": "$250.00"
        }
        
        # Format the signal data for notification
        action = "BUY" if test_signal_data.get("Signal") == 1 else "SELL"
        price = test_signal_data.get("Price", "N/A")
        stop_loss = test_signal_data.get("SL Price", "N/A")
        take_profit = test_signal_data.get("TP Price", "N/A")
        lots = test_signal_data.get("Lots", "N/A")
        adr = test_signal_data.get("ADR", "N/A")
        risk_info = f"Current VaR: {test_signal_data.get('CurrVAR', 'N/A')}, Proposed VaR: {test_signal_data.get('PropVAR', 'N/A')}"
        
        # Create notification message
        title = f"TEST - Trading Signal: EURUSD {action}"
        
        body = f"""
<b>TEST - Trading Signal Detected</b>

<b>Symbol:</b> EURUSD
<b>Action:</b> {action}
<b>Price:</b> {price}
<b>Stop Loss:</b> {stop_loss}
<b>Take Profit:</b> {take_profit}
<b>Lots:</b> {lots}
<b>ADR:</b> {adr}
<b>Risk:</b> {risk_info}

<b>Time:</b> {test_signal_data.get('Time', 'N/A')}
<b>Strategy:</b> {test_signal_data.get('Strategy Name', 'Swing Trend Momentum')}

<i>This is a test notification from the Live Signals Bot.</i>
        """
        
        # Send the notification
        results = notification_manager.send_custom_message(
            title=title,
            body=body,
            level="INFO",
            services=["telegram", "email"]  # Send to both Telegram and Email if available
        )
        
        # Log the results
        for service, result in results.items():
            if result.success:
                logger.info(f"Test signal notification sent successfully via {service}")
            else:
                logger.error(f"Failed to send test signal notification via {service}: {result.error}")
                
    except Exception as e:
        logger.error(f"Error sending test trading signal notification: {e}")


def main():
    """Test the notification functionality."""
    logger.info("Testing notification functionality...")
    
    # Initialize notification manager
    notification_manager = initialize_notification_manager()
    
    if not notification_manager:
        logger.error("No notification services configured. Please check your config.ini file.")
        logger.info("See the documentation in live_signals.py for configuration examples.")
        return
    
    # Send test notifications
    logger.info("Sending test notifications...")
    
    # Test 1: System alert
    try:
        logger.info("Sending system alert...")
        results = notification_manager.send_system_alert(
            level="INFO",
            message="Test System Alert",
            details="This is a test system alert from the Live Signals Bot",
            component="Test Component",
            status="Testing"
        )
        logger.info(f"System alert results: {results}")
        for service, result in results.items():
            if result.success:
                logger.info(f"Test system alert sent via {service}")
            else:
                logger.error(f"Failed to send test system alert via {service}: {result.error}")
    except Exception as e:
        logger.error(f"Error sending test system alert: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Test 2: Trading signal notification
    send_test_trading_signal_notification(notification_manager)
    
    # Test 3: Custom message
    try:
        logger.info("Sending custom message...")
        results = notification_manager.send_custom_message(
            title="Test Complete",
            body="Notification test completed successfully!",
            level="INFO"
        )
        logger.info(f"Custom message results: {results}")
        for service, result in results.items():
            if result.success:
                logger.info(f"Test completion message sent via {service}")
            else:
                logger.error(f"Failed to send test completion message via {service}: {result.error}")
    except Exception as e:
        logger.error(f"Error sending test completion message: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("Notification test completed!")


if __name__ == "__main__":
    main() 