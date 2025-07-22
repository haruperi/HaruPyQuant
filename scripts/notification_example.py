#!/usr/bin/env python3
"""
Notification Service Example Script

This script demonstrates how to use the HaruPyQuant notification service
with various configuration options and notification types.

Usage:
    python scripts/notification_example.py [--config CONFIG_FILE] [--test-all]
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.notifications import NotificationManager, NotificationLevel
from app.notifications.config import NotificationConfig, NotificationPresets
from app.notifications.email import EmailConfig, EmailProviders
from app.notifications.telegram import TelegramConfig
from app.notifications.sms import SMSConfig
from app.notifications.manager import NotificationManagerConfig
from app.util import get_logger

logger = get_logger(__name__)


def create_basic_config():
    """Create a basic notification configuration for testing."""
    config = NotificationManagerConfig()
    
    # Email configuration (Gmail example)
    # Note: You'll need to set up an app password for Gmail
    email_config = EmailProviders.gmail(
        username="your-email@gmail.com",
        password="your-app-password",
        app_password="your-app-password"
    )
    email_config.default_recipients = ["recipient@example.com"]
    config.email_config = email_config
    
    # Telegram configuration
    # Note: You'll need to create a bot and get the token
    telegram_config = TelegramConfig(
        bot_token="your-bot-token",
        chat_ids=["your-chat-id"]
    )
    config.telegram_config = telegram_config
    
    # SMS configuration (Twilio example)
    # Note: You'll need a Twilio account
    sms_config = SMSConfig(
        account_sid="your-account-sid",
        auth_token="your-auth-token",
        from_number="+1234567890"
    )
    sms_config.default_recipients = ["+1234567890"]
    config.sms_config = sms_config
    
    return config


def create_ini_config():
    """Create configuration from INI file."""
    config = NotificationConfig.from_ini()
    
    manager_config = NotificationManagerConfig()
    
    if config.email_enabled:
        manager_config.email_config = config.get_email_config()
    
    if config.telegram_enabled:
        manager_config.telegram_config = config.get_telegram_config()
    
    if config.sms_enabled:
        manager_config.sms_config = config.get_sms_config()
    
    manager_config.default_levels = config.get_default_levels()
    manager_config.enable_all = config.enable_all
    
    return manager_config


def create_env_config():
    """Create configuration from environment variables."""
    config = NotificationConfig.from_env()
    
    manager_config = NotificationManagerConfig()
    
    if config.email_enabled:
        manager_config.email_config = config.get_email_config()
    
    if config.telegram_enabled:
        manager_config.telegram_config = config.get_telegram_config()
    
    if config.sms_enabled:
        manager_config.sms_config = config.get_sms_config()
    
    manager_config.default_levels = config.get_default_levels()
    manager_config.enable_all = config.enable_all
    
    return manager_config


def demo_trading_alerts(notifier: NotificationManager):
    """Demonstrate trading alert notifications."""
    print("\n=== Trading Alert Demonstrations ===")
    
    # Basic trading alert
    print("1. Sending basic trading alert...")
    results = notifier.send_trading_alert(
        symbol="EURUSD",
        action="BUY",
        price=1.0850,
        reason="RSI oversold condition",
        account="Demo Account",
        strategy="RSI Strategy",
        risk_level="Medium"
    )
    print_results(results)
    
    # Position opened alert
    print("\n2. Sending position opened alert...")
    message = notifier.template.render(
        "position_opened",
        symbol="GBPUSD",
        direction="BUY",
        size="0.1",
        entry_price="1.2650",
        stop_loss="1.2600",
        take_profit="1.2750",
        account="Live Account",
        strategy="Breakout Strategy",
        risk_amount="$50"
    )
    results = notifier.send_notification(message)
    print_results(results)
    
    # Position update
    print("\n3. Sending position update...")
    results = notifier.send_position_update(
        symbol="USDJPY",
        position_type="SELL",
        size=0.05,
        entry_price=150.50,
        current_price=150.25,
        pnl=12.50,
        pnl_percent=2.5
    )
    print_results(results)


def demo_system_alerts(notifier: NotificationManager):
    """Demonstrate system alert notifications."""
    print("\n=== System Alert Demonstrations ===")
    
    # System startup
    print("1. Sending system startup alert...")
    message = notifier.template.render(
        "system_startup",
        version="1.0.0",
        environment="Production",
        account="Demo Account",
        mt5_status="Connected",
        data_feed_status="Active",
        strategy_status="Running",
        risk_manager_status="Active"
    )
    results = notifier.send_notification(message)
    print_results(results)
    
    # Connection lost
    print("\n2. Sending connection lost alert...")
    results = notifier.send_system_alert(
        level="WARNING",
        message="MT5 connection lost",
        details="Network timeout after 30 seconds",
        component="MT5 Client",
        status="Disconnected"
    )
    print_results(results)
    
    # Error alert
    print("\n3. Sending error alert...")
    results = notifier.send_error_alert(
        error_type="StrategyError",
        message="Invalid signal generated",
        component="RSI Strategy",
        stack_trace="Traceback (most recent call last):\n  File 'strategy.py', line 45, in <module>\n    signal = generate_signal()\nValueError: Invalid parameters"
    )
    print_results(results)


def demo_performance_alerts(notifier: NotificationManager):
    """Demonstrate performance alert notifications."""
    print("\n=== Performance Alert Demonstrations ===")
    
    # Performance alert
    print("1. Sending performance alert...")
    message = notifier.template.render(
        "performance_alert",
        alert_type="High Win Rate",
        metric="Win Rate",
        value="75%",
        threshold="70%",
        period="Last 30 days",
        account="Live Account"
    )
    results = notifier.send_notification(message)
    print_results(results)
    
    # Drawdown alert
    print("\n2. Sending drawdown alert...")
    message = notifier.template.render(
        "drawdown_alert",
        drawdown_type="Current",
        current_drawdown=5.2,
        peak_drawdown=8.1,
        duration="3 days",
        account="Live Account",
        balance="$10,000",
        equity="$9,480"
    )
    results = notifier.send_notification(message)
    print_results(results)


def demo_market_alerts(notifier: NotificationManager):
    """Demonstrate market alert notifications."""
    print("\n=== Market Alert Demonstrations ===")
    
    # Market alert
    print("1. Sending market alert...")
    message = notifier.template.render(
        "market_alert",
        symbol="EURUSD",
        event="NFP Release",
        price="1.0850",
        impact="High",
        details="Non-Farm Payrolls data released, USD strengthening"
    )
    results = notifier.send_notification(message)
    print_results(results)
    
    # News alert
    print("\n2. Sending news alert...")
    message = notifier.template.render(
        "news_alert",
        headline="Fed Announces Rate Decision",
        source="Reuters",
        impact="High",
        summary="Federal Reserve maintains current interest rates",
        symbols="USD, EUR, GBP"
    )
    results = notifier.send_notification(message)
    print_results(results)


def demo_custom_messages(notifier: NotificationManager):
    """Demonstrate custom message notifications."""
    print("\n=== Custom Message Demonstrations ===")
    
    # Custom message
    print("1. Sending custom message...")
    results = notifier.send_custom_message(
        title="Custom Trading Alert",
        body="This is a custom message with specific trading information.\n\nSymbol: AUDUSD\nAction: SELL\nReason: Technical breakdown",
        level="INFO",
        metadata={
            "custom_field": "custom_value",
            "priority": "high"
        }
    )
    print_results(results)
    
    # Test message
    print("\n2. Sending test message...")
    message = notifier.template.render(
        "test_message",
        service="All Services",
        status="Testing"
    )
    results = notifier.send_notification(message)
    print_results(results)


def demo_service_management(notifier: NotificationManager):
    """Demonstrate service management features."""
    print("\n=== Service Management Demonstrations ===")
    
    # Test all services
    print("1. Testing all services...")
    test_results = notifier.test_all_services()
    for service, result in test_results.items():
        print(f"  {service}: {'PASS' if result else 'FAIL'}")
    
    # Get service status
    print("\n2. Service status:")
    status = notifier.get_service_status()
    for service, info in status.items():
        print(f"  {service}: {'Enabled' if info['enabled'] else 'Disabled'}")
        print(f"    Rate limit: {info['rate_limit']['max_requests']} requests per {info['rate_limit']['time_window']} seconds")
    
    # Get statistics
    print("\n3. Notification statistics:")
    stats = notifier.get_statistics()
    print(f"  Total sent: {stats['total_sent']}")
    print(f"  Total failed: {stats['total_failed']}")
    print(f"  By service: {stats['by_service']}")
    print(f"  By level: {stats['by_level']}")


def demo_template_management(notifier: NotificationManager):
    """Demonstrate template management features."""
    print("\n=== Template Management Demonstrations ===")
    
    # List templates
    print("1. Available templates:")
    templates = notifier.list_templates()
    for template in templates:
        print(f"  - {template}")
    
    # Add custom template
    print("\n2. Adding custom template...")
    notifier.add_template(
        "custom_alert",
        "Custom Alert: {alert_type}",
        """
🚨 Custom Alert

Type: {alert_type}
Message: {message}
Time: {timestamp}

This is a custom template for specific alerts.
        """.strip()
    )
    
    # Use custom template
    print("3. Using custom template...")
    message = notifier.template.render(
        "custom_alert",
        alert_type="Market Volatility",
        message="High volatility detected in EURUSD"
    )
    results = notifier.send_notification(message)
    print_results(results)
    
    # Get template info
    print("\n4. Template information:")
    template_info = notifier.template.get_template_info("trading_alert")
    print(f"  Name: {template_info['name']}")
    print(f"  Required variables: {template_info['required_variables']}")
    print(f"  Title length: {template_info['title_length']}")
    print(f"  Body length: {template_info['body_length']}")


def print_results(results: dict):
    """Print notification results in a formatted way."""
    if not results:
        print("  No results (no services configured or enabled)")
        return
    
    for service, result in results.items():
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"  {service}: {status}")
        if not result.success and result.error_message:
            print(f"    Error: {result.error_message}")
        if result.delivery_time_ms:
            print(f"    Delivery time: {result.delivery_time_ms}ms")


def main():
    """Main function to run the notification examples."""
    parser = argparse.ArgumentParser(description="Notification Service Example")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--test-all", action="store_true", help="Test all services")
    parser.add_argument("--env", action="store_true", help="Use environment variables for configuration")
    parser.add_argument("--ini", action="store_true", help="Use INI file for configuration (default)")
    parser.add_argument("--basic", action="store_true", help="Use basic configuration (requires manual setup)")
    
    args = parser.parse_args()
    
    print("HaruPyQuant Notification Service Example")
    print("=" * 50)
    
    # Create notification manager
    try:
        if args.config:
            # Load from file
            config = NotificationConfig.from_file(args.config)
            manager_config = NotificationManagerConfig()
            
            if config.email_enabled:
                manager_config.email_config = config.get_email_config()
            if config.telegram_enabled:
                manager_config.telegram_config = config.get_telegram_config()
            if config.sms_enabled:
                manager_config.sms_config = config.get_sms_config()
            
            manager_config.default_levels = config.get_default_levels()
            manager_config.enable_all = config.enable_all
            
        elif args.env:
            # Load from environment variables
            manager_config = create_env_config()
            
        elif args.basic:
            # Use basic configuration
            manager_config = create_basic_config()
            
        else:
            # Use INI file by default
            manager_config = create_ini_config()
        
        notifier = NotificationManager(manager_config)
        
    except Exception as e:
        logger.error(f"Failed to create notification manager: {str(e)}")
        print(f"Error: {str(e)}")
        print("\nPlease configure your notification services first.")
        print("You can:")
        print("1. Set environment variables (see docs/notifications.md)")
        print("2. Create a configuration file")
        print("3. Use --basic flag and modify the script")
        return
    
    # Check if any services are configured
    if not notifier.list_services():
        print("No notification services configured.")
        print("Please configure at least one service (email, telegram, or SMS).")
        return
    
    print(f"Configured services: {', '.join(notifier.list_services())}")
    
    if args.test_all:
        # Test all services
        print("\nTesting all services...")
        test_results = notifier.test_all_services()
        all_passed = all(test_results.values())
        
        for service, result in test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {service}: {status}")
        
        if not all_passed:
            print("\nSome services failed. Please check your configuration.")
            return
        
        print("\nAll services passed! Running demonstrations...")
    
    # Run demonstrations
    try:
        demo_trading_alerts(notifier)
        demo_system_alerts(notifier)
        demo_performance_alerts(notifier)
        demo_market_alerts(notifier)
        demo_custom_messages(notifier)
        demo_service_management(notifier)
        demo_template_management(notifier)
        
        print("\n=== All Demonstrations Complete ===")
        print("Check your configured notification channels for the messages.")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}", exc_info=True)
        print(f"\nError during demonstration: {str(e)}")


if __name__ == "__main__":
    main() 