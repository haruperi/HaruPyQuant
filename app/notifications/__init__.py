"""
Notification Service Module

This module provides comprehensive notification services for trading alerts and system notifications.
Supports email, Telegram, and SMS notifications with unified interface and configuration management.

Classes:
    NotificationManager: Main interface for sending notifications
    EmailNotifier: Email notification service using SMTP
    TelegramNotifier: Telegram bot notification service
    SMSNotifier: SMS notification service using Twilio
    NotificationTemplate: Template system for notification messages
    NotificationConfig: Configuration management for notification settings

Usage:
    from app.notifications import NotificationManager
    
    # Initialize notification manager
    notifier = NotificationManager()
    
    # Send trading alert
    notifier.send_trading_alert(
        symbol="EURUSD",
        action="BUY",
        price=1.0850,
        reason="RSI oversold condition"
    )
    
    # Send system notification
    notifier.send_system_alert(
        level="WARNING",
        message="MT5 connection lost",
        details="Attempting to reconnect..."
    )
"""

from .manager import NotificationManager
from .base import NotificationError, NotificationLevel
from .email import EmailNotifier
from .telegram import TelegramNotifier
from .sms import SMSNotifier
from .templates import NotificationTemplate
from .config import NotificationConfig

__version__ = "1.0.0"
__author__ = "HaruPyQuant Team"

__all__ = [
    "NotificationManager",
    "NotificationError", 
    "NotificationLevel",
    "EmailNotifier",
    "TelegramNotifier", 
    "SMSNotifier",
    "NotificationTemplate",
    "NotificationConfig"
] 