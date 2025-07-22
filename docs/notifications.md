# Notification Service Documentation

The HaruPyQuant notification service provides comprehensive alerting capabilities for trading systems, supporting multiple channels including email, Telegram, and SMS notifications.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Templates](#templates)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The notification service is designed to provide real-time alerts for:
- Trading signals and position updates
- System status and errors
- Performance metrics and drawdowns
- Market events and news
- Risk management alerts

### Supported Channels

- **Email**: SMTP-based email notifications with HTML formatting
- **Telegram**: Bot-based messaging with rich formatting
- **SMS**: Twilio-based SMS notifications (optional)

## Features

### Core Features

- **Multi-channel Support**: Send notifications through email, Telegram, and SMS
- **Template System**: Pre-defined templates for common notification types
- **Rate Limiting**: Built-in rate limiting to prevent spam
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Environment variables and file-based configuration
- **Statistics**: Track notification delivery and failure rates

### Advanced Features

- **Custom Templates**: Create and manage custom notification templates
- **Service Management**: Enable/disable individual services
- **Connection Testing**: Test service connectivity
- **Message Formatting**: Rich formatting for different channels
- **Metadata Support**: Attach additional data to notifications
- **Level-based Filtering**: Filter notifications by priority level

## Installation

### Dependencies

The notification service requires the following Python packages:

```bash
pip install requests
```

For email notifications, no additional packages are required (uses built-in `smtplib`).

For SMS notifications (optional), install Twilio:

```bash
pip install twilio
```

### Setup

1. **Email Setup**:
   - For Gmail: Enable 2-factor authentication and create an app password
   - For other providers: Use your SMTP credentials

2. **Telegram Setup**:
   - Create a bot using [@BotFather](https://t.me/botfather)
   - Get your bot token
   - Get your chat ID (send a message to your bot and check the chat ID)

3. **SMS Setup (Optional)**:
   - Create a Twilio account
   - Get your Account SID and Auth Token
   - Get a phone number for sending SMS

## Configuration

### INI File Configuration (Recommended)

The notification service is configured using the `config.ini` file, which is consistent with the rest of the HaruPyQuant project.

#### Email Configuration

```ini
[EMAIL]
enabled = false
smtp_server = smtp.gmail.com
smtp_port = 587
username = your-email@gmail.com
password = your-app-password
use_tls = true
use_ssl = false
from_email = your-email@gmail.com
from_name = HaruPyQuant
recipients = recipient1@example.com,recipient2@example.com
```

#### Telegram Configuration

```ini
[TELEGRAM]
enabled = true
token = your-bot-token
chat_ids = 123456789,-987654321
parse_mode = HTML
disable_web_page_preview = true
disable_notification = false
protect_content = false
```

#### SMS Configuration

```ini
[SMS]
enabled = false
account_sid = your-account-sid
auth_token = your-auth-token
from_number = +1234567890
recipients = +1234567890,+0987654321
webhook_url = 
status_callback = 
```

#### General Settings

```ini
[NOTIFICATIONS]
enable_all = true
default_levels = WARNING,ERROR,CRITICAL
```

### Environment Variables (Alternative)

The notification service can also be configured using environment variables as a fallback:

#### Email Configuration

```bash
# Enable email notifications
NOTIFICATION_EMAIL_ENABLED=true

# SMTP settings
NOTIFICATION_EMAIL_SMTP_SERVER=smtp.gmail.com
NOTIFICATION_EMAIL_SMTP_PORT=587
NOTIFICATION_EMAIL_USERNAME=your-email@gmail.com
NOTIFICATION_EMAIL_PASSWORD=your-app-password
NOTIFICATION_EMAIL_USE_TLS=true
NOTIFICATION_EMAIL_FROM_EMAIL=your-email@gmail.com
NOTIFICATION_EMAIL_FROM_NAME=HaruPyQuant

# Recipients (comma-separated)
NOTIFICATION_EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

#### Telegram Configuration

```bash
# Enable Telegram notifications
NOTIFICATION_TELEGRAM_ENABLED=true

# Bot settings
NOTIFICATION_TELEGRAM_BOT_TOKEN=your-bot-token
NOTIFICATION_TELEGRAM_CHAT_IDS=123456789,-987654321

# Optional settings
NOTIFICATION_TELEGRAM_PARSE_MODE=HTML
NOTIFICATION_TELEGRAM_DISABLE_WEB_PAGE_PREVIEW=true
```

#### SMS Configuration

```bash
# Enable SMS notifications
NOTIFICATION_SMS_ENABLED=true

# Twilio settings
NOTIFICATION_SMS_ACCOUNT_SID=your-account-sid
NOTIFICATION_SMS_AUTH_TOKEN=your-auth-token
NOTIFICATION_SMS_FROM_NUMBER=+1234567890
NOTIFICATION_SMS_RECIPIENTS=+1234567890,+0987654321
```

#### General Settings

```bash
# Enable all notifications
NOTIFICATION_ENABLE_ALL=true

# Default notification levels (comma-separated)
NOTIFICATION_DEFAULT_LEVELS=WARNING,ERROR,CRITICAL
```

### Configuration File

You can also use a JSON configuration file:

```json
{
  "email_enabled": true,
  "email_smtp_server": "smtp.gmail.com",
  "email_smtp_port": 587,
  "email_username": "your-email@gmail.com",
  "email_password": "your-app-password",
  "email_use_tls": true,
  "email_from_email": "your-email@gmail.com",
  "email_default_recipients": ["recipient@example.com"],
  
  "telegram_enabled": true,
  "telegram_bot_token": "your-bot-token",
  "telegram_chat_ids": ["123456789"],
  "telegram_parse_mode": "HTML",
  
  "sms_enabled": false,
  "sms_account_sid": "",
  "sms_auth_token": "",
  "sms_from_number": "",
  "sms_default_recipients": [],
  
  "enable_all": true,
  "default_levels": ["WARNING", "ERROR", "CRITICAL"]
}
```

## Usage Examples

### Basic Usage

```python
from app.notifications import NotificationManager, NotificationManagerConfig
from app.notifications.config import NotificationConfig

# Load configuration from INI file (default)
config = NotificationConfig.from_ini()
manager_config = NotificationManagerConfig()

if config.email_enabled:
    manager_config.email_config = config.get_email_config()
if config.telegram_enabled:
    manager_config.telegram_config = config.get_telegram_config()

# Create notification manager
notifier = NotificationManager(manager_config)

# Send trading alert
notifier.send_trading_alert(
    symbol="EURUSD",
    action="BUY",
    price=1.0850,
    reason="RSI oversold condition"
)

# Send system alert
notifier.send_system_alert(
    level="WARNING",
    message="MT5 connection lost",
    details="Network timeout after 30 seconds"
)
```

### Advanced Usage

```python
# Send position update
notifier.send_position_update(
    symbol="GBPUSD",
    position_type="BUY",
    size=0.1,
    entry_price=1.2650,
    current_price=1.2700,
    pnl=50.0,
    pnl_percent=5.0
)

# Send error alert
notifier.send_error_alert(
    error_type="StrategyError",
    message="Invalid signal generated",
    component="RSI Strategy",
    stack_trace="Traceback..."
)

# Send custom message
notifier.send_custom_message(
    title="Custom Alert",
    body="This is a custom notification message",
    level="INFO",
    metadata={"custom_field": "value"}
)
```

### Service Management

```python
# Test all services
test_results = notifier.test_all_services()
for service, result in test_results.items():
    print(f"{service}: {'PASS' if result else 'FAIL'}")

# Get service status
status = notifier.get_service_status()
for service, info in status.items():
    print(f"{service}: {'Enabled' if info['enabled'] else 'Disabled'}")

# Enable/disable services
notifier.enable_service('email')
notifier.disable_service('sms')

# Get statistics
stats = notifier.get_statistics()
print(f"Total sent: {stats['total_sent']}")
print(f"Total failed: {stats['total_failed']}")
```

### Template Management

```python
# List available templates
templates = notifier.list_templates()
print(f"Available templates: {templates}")

# Add custom template
notifier.add_template(
    "custom_alert",
    "Custom Alert: {alert_type}",
    """
🚨 Custom Alert

Type: {alert_type}
Message: {message}
Time: {timestamp}
    """.strip()
)

# Use custom template
message = notifier.template.render(
    "custom_alert",
    alert_type="Market Volatility",
    message="High volatility detected"
)
notifier.send_notification(message)
```

## API Reference

### NotificationManager

The main class for managing notifications.

#### Constructor

```python
NotificationManager(config: Optional[NotificationManagerConfig] = None)
```

#### Methods

##### send_notification(message, services=None)

Send a notification message through specified services.

- `message`: NotificationMessage object
- `services`: List of service names (email, telegram, sms) or None for all

Returns: Dictionary mapping service names to NotificationResult objects

##### send_trading_alert(symbol, action, price, reason, account="Demo", strategy="Unknown", risk_level="Medium", services=None)

Send a trading alert notification.

##### send_system_alert(level, message, details="", component="System", status="Active", services=None)

Send a system alert notification.

##### send_position_update(symbol, position_type, size, entry_price, current_price, pnl, pnl_percent, services=None)

Send a position update notification.

##### send_error_alert(error_type, message, component="Unknown", stack_trace="", services=None)

Send an error alert notification.

##### send_custom_message(title, body, level="INFO", metadata=None, recipients=None, services=None)

Send a custom notification message.

##### test_all_services()

Test all configured notification services.

Returns: Dictionary mapping service names to boolean results

##### enable_service(service_name)

Enable a specific notification service.

##### disable_service(service_name)

Disable a specific notification service.

##### get_service_status()

Get status of all notification services.

Returns: Dictionary with service status information

##### get_statistics()

Get notification statistics.

Returns: Dictionary with statistics

##### add_template(name, title_template, body_template)

Add a new notification template.

##### list_templates()

List all available template names.

Returns: List of template names

### NotificationMessage

Represents a notification message with metadata.

#### Attributes

- `title`: Message title
- `body`: Message body
- `level`: NotificationLevel (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `timestamp`: Message timestamp
- `metadata`: Additional metadata dictionary
- `recipients`: List of specific recipients
- `template_name`: Name of the template used

### NotificationResult

Represents the result of a notification attempt.

#### Attributes

- `success`: Boolean indicating success
- `message_id`: Optional message identifier
- `error_message`: Optional error message
- `timestamp`: Result timestamp
- `retry_count`: Number of retry attempts
- `delivery_time_ms`: Delivery time in milliseconds

## Templates

### Built-in Templates

The notification service includes pre-defined templates for common use cases:

#### Trading Templates

- `trading_alert`: Basic trading alert
- `trading_signal`: Trading signal with entry/exit levels
- `position_opened`: Position opened notification
- `position_closed`: Position closed notification
- `position_update`: Position update notification

#### System Templates

- `system_alert`: System alert notification
- `system_startup`: System startup notification
- `system_shutdown`: System shutdown notification
- `connection_lost`: Connection lost notification
- `connection_restored`: Connection restored notification

#### Error Templates

- `error_alert`: Error alert notification
- `strategy_error`: Strategy-specific error notification

#### Performance Templates

- `performance_alert`: Performance metric alert
- `drawdown_alert`: Drawdown alert notification

#### Market Templates

- `market_alert`: Market event notification
- `news_alert`: News event notification

#### Risk Templates

- `risk_alert`: Risk management alert
- `margin_alert`: Margin level alert

### Custom Templates

You can create custom templates using the template system:

```python
# Add custom template
notifier.add_template(
    "my_custom_alert",
    "Custom Alert: {alert_type}",
    """
🚨 Custom Alert

Type: {alert_type}
Message: {message}
Time: {timestamp}

Additional Info: {additional_info}
    """.strip()
)

# Use custom template
message = notifier.template.render(
    "my_custom_alert",
    alert_type="Volatility Alert",
    message="High volatility detected",
    additional_info="Consider reducing position sizes"
)
```

### Template Variables

Templates use Python string formatting with variables enclosed in curly braces:

```python
# Template with variables
title = "Alert: {symbol} {action}"
body = """
Symbol: {symbol}
Action: {action}
Price: {price}
Time: {timestamp}
"""

# Render template
message = notifier.template.render(
    "my_template",
    symbol="EURUSD",
    action="BUY",
    price=1.0850
)
```

## Best Practices

### Configuration

1. **Use Environment Variables**: Store sensitive information in environment variables
2. **Validate Configuration**: Always validate configuration before using
3. **Test Services**: Test all services before going live
4. **Use Rate Limits**: Configure appropriate rate limits for each service

### Message Content

1. **Keep Messages Concise**: SMS has 160 character limit
2. **Use Appropriate Levels**: Use correct notification levels
3. **Include Context**: Provide relevant context in messages
4. **Use Templates**: Use templates for consistent formatting

### Error Handling

1. **Handle Failures**: Always check notification results
2. **Log Errors**: Log failed notifications for debugging
3. **Retry Logic**: Use built-in retry logic for transient failures
4. **Fallback**: Consider multiple notification channels

### Performance

1. **Rate Limiting**: Respect rate limits to avoid service blocks
2. **Async Processing**: Consider async processing for high-volume notifications
3. **Batching**: Batch notifications when possible
4. **Monitoring**: Monitor notification delivery rates

## Troubleshooting

### Common Issues

#### Email Notifications

**Issue**: Authentication failed
- **Solution**: Check username/password and enable app passwords for Gmail

**Issue**: Connection timeout
- **Solution**: Check SMTP server and port settings

**Issue**: Messages not received
- **Solution**: Check spam folder and recipient email addresses

#### Telegram Notifications

**Issue**: Bot token invalid
- **Solution**: Verify bot token with @BotFather

**Issue**: Chat ID not found
- **Solution**: Send a message to your bot and get the correct chat ID

**Issue**: Messages not delivered
- **Solution**: Check if bot is blocked or chat is private

#### SMS Notifications

**Issue**: Twilio credentials invalid
- **Solution**: Verify Account SID and Auth Token

**Issue**: Phone number not verified
- **Solution**: Verify phone numbers in Twilio console

**Issue**: SMS not delivered
- **Solution**: Check phone number format and Twilio account status

### Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Individual Services

```python
# Test email service
email_notifier = notifier.get_notifier('email')
if email_notifier:
    result = email_notifier.test_connection()
    print(f"Email test: {'PASS' if result else 'FAIL'}")

# Test Telegram service
telegram_notifier = notifier.get_notifier('telegram')
if telegram_notifier:
    result = telegram_notifier.test_connection()
    print(f"Telegram test: {'PASS' if result else 'FAIL'}")
```

#### Check Service Status

```python
# Get detailed service status
status = notifier.get_service_status()
for service, info in status.items():
    print(f"Service: {service}")
    print(f"  Enabled: {info['enabled']}")
    print(f"  Rate Limit: {info['rate_limit']}")
```

#### Monitor Statistics

```python
# Get notification statistics
stats = notifier.get_statistics()
print(f"Total sent: {stats['total_sent']}")
print(f"Total failed: {stats['total_failed']}")
print(f"By service: {stats['by_service']}")
print(f"By level: {stats['by_level']}")
```

### Getting Help

1. **Check Logs**: Review application logs for error messages
2. **Test Configuration**: Use the example script to test configuration
3. **Verify Credentials**: Double-check all service credentials
4. **Check Documentation**: Review this documentation for configuration details

## Example Script

Run the example script to test your configuration:

```bash
# Test with INI file (default)
python scripts/notification_example.py --test-all

# Test with environment variables
python scripts/notification_example.py --env --test-all

# Test with configuration file
python scripts/notification_example.py --config config/notifications.json

# Test with basic configuration
python scripts/notification_example.py --basic
```

The example script demonstrates all notification features and provides a good starting point for integration. 