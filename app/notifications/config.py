"""
Configuration management for notification services.

This module provides configuration management for all notification services,
including loading from environment variables, configuration files, and validation.
"""

import os
import json
import configparser
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .email import EmailConfig, EmailProviders
from .telegram import TelegramConfig
from .sms import SMSConfig
from .base import NotificationLevel

from app.util import get_logger

logger = get_logger(__name__)


@dataclass
class NotificationConfig:
    """Main configuration class for notification services."""
    
    # Email configuration
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_use_tls: bool = True
    email_use_ssl: bool = False
    email_from_email: str = ""
    email_from_name: str = ""
    email_default_recipients: List[str] = None
    
    # Telegram configuration
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_ids: List[str] = None
    telegram_parse_mode: str = "HTML"
    telegram_disable_web_page_preview: bool = True
    telegram_disable_notification: bool = False
    telegram_protect_content: bool = False
    
    # SMS configuration
    sms_enabled: bool = False
    sms_account_sid: str = ""
    sms_auth_token: str = ""
    sms_from_number: str = ""
    sms_default_recipients: List[str] = None
    sms_webhook_url: str = ""
    sms_status_callback: str = ""
    
    # General settings
    default_levels: List[str] = None
    enable_all: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.email_default_recipients is None:
            self.email_default_recipients = []
        if self.telegram_chat_ids is None:
            self.telegram_chat_ids = []
        if self.sms_default_recipients is None:
            self.sms_default_recipients = []
        if self.default_levels is None:
            self.default_levels = ["WARNING", "ERROR", "CRITICAL"]
    
    @classmethod
    def from_ini(cls, config_file: str = 'config.ini') -> 'NotificationConfig':
        """Create configuration from INI file."""
        config = cls()
        
        try:
            parser = configparser.ConfigParser()
            parser.read(config_file)
            
            # Email configuration
            if parser.has_section('EMAIL'):
                config.email_enabled = parser.getboolean('EMAIL', 'enabled', fallback=False)
                config.email_smtp_server = parser.get('EMAIL', 'smtp_server', fallback='')
                config.email_smtp_port = parser.getint('EMAIL', 'smtp_port', fallback=587)
                config.email_username = parser.get('EMAIL', 'username', fallback='')
                config.email_password = parser.get('EMAIL', 'password', fallback='')
                config.email_use_tls = parser.getboolean('EMAIL', 'use_tls', fallback=True)
                config.email_use_ssl = parser.getboolean('EMAIL', 'use_ssl', fallback=False)
                config.email_from_email = parser.get('EMAIL', 'from_email', fallback='')
                config.email_from_name = parser.get('EMAIL', 'from_name', fallback='')
                
                # Parse email recipients
                email_recipients = parser.get('EMAIL', 'recipients', fallback='')
                if email_recipients:
                    config.email_default_recipients = [r.strip() for r in email_recipients.split(',') if r.strip()]
            
            # Telegram configuration
            if parser.has_section('TELEGRAM'):
                config.telegram_enabled = parser.getboolean('TELEGRAM', 'enabled', fallback=False)
                config.telegram_bot_token = parser.get('TELEGRAM', 'token', fallback='')
                config.telegram_parse_mode = parser.get('TELEGRAM', 'parse_mode', fallback='HTML')
                config.telegram_disable_web_page_preview = parser.getboolean('TELEGRAM', 'disable_web_page_preview', fallback=True)
                config.telegram_disable_notification = parser.getboolean('TELEGRAM', 'disable_notification', fallback=False)
                config.telegram_protect_content = parser.getboolean('TELEGRAM', 'protect_content', fallback=False)
                
                # Parse Telegram chat IDs
                telegram_chat_ids = parser.get('TELEGRAM', 'chat_ids', fallback='')
                if telegram_chat_ids:
                    config.telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids.split(',') if cid.strip()]
            
            # SMS configuration
            if parser.has_section('SMS'):
                config.sms_enabled = parser.getboolean('SMS', 'enabled', fallback=False)
                config.sms_account_sid = parser.get('SMS', 'account_sid', fallback='')
                config.sms_auth_token = parser.get('SMS', 'auth_token', fallback='')
                config.sms_from_number = parser.get('SMS', 'from_number', fallback='')
                config.sms_webhook_url = parser.get('SMS', 'webhook_url', fallback='')
                config.sms_status_callback = parser.get('SMS', 'status_callback', fallback='')
                
                # Parse SMS recipients
                sms_recipients = parser.get('SMS', 'recipients', fallback='')
                if sms_recipients:
                    config.sms_default_recipients = [r.strip() for r in sms_recipients.split(',') if r.strip()]
            
            # General settings
            if parser.has_section('NOTIFICATIONS'):
                config.enable_all = parser.getboolean('NOTIFICATIONS', 'enable_all', fallback=True)
                
                # Parse default levels
                default_levels = parser.get('NOTIFICATIONS', 'default_levels', fallback='WARNING,ERROR,CRITICAL')
                if default_levels:
                    config.default_levels = [level.strip() for level in default_levels.split(',') if level.strip()]
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
            # Return default configuration if file not found or error
            pass
        
        return config
    
    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Create configuration from environment variables (fallback method)."""
        config = cls()
        
        # Email configuration
        config.email_enabled = os.getenv('NOTIFICATION_EMAIL_ENABLED', 'false').lower() == 'true'
        config.email_smtp_server = os.getenv('NOTIFICATION_EMAIL_SMTP_SERVER', '')
        config.email_smtp_port = int(os.getenv('NOTIFICATION_EMAIL_SMTP_PORT', '587'))
        config.email_username = os.getenv('NOTIFICATION_EMAIL_USERNAME', '')
        config.email_password = os.getenv('NOTIFICATION_EMAIL_PASSWORD', '')
        config.email_use_tls = os.getenv('NOTIFICATION_EMAIL_USE_TLS', 'true').lower() == 'true'
        config.email_use_ssl = os.getenv('NOTIFICATION_EMAIL_USE_SSL', 'false').lower() == 'true'
        config.email_from_email = os.getenv('NOTIFICATION_EMAIL_FROM_EMAIL', '')
        config.email_from_name = os.getenv('NOTIFICATION_EMAIL_FROM_NAME', '')
        
        # Parse email recipients
        email_recipients = os.getenv('NOTIFICATION_EMAIL_RECIPIENTS', '')
        if email_recipients:
            config.email_default_recipients = [r.strip() for r in email_recipients.split(',') if r.strip()]
        
        # Telegram configuration
        config.telegram_enabled = os.getenv('NOTIFICATION_TELEGRAM_ENABLED', 'false').lower() == 'true'
        config.telegram_bot_token = os.getenv('NOTIFICATION_TELEGRAM_BOT_TOKEN', '')
        config.telegram_parse_mode = os.getenv('NOTIFICATION_TELEGRAM_PARSE_MODE', 'HTML')
        config.telegram_disable_web_page_preview = os.getenv('NOTIFICATION_TELEGRAM_DISABLE_WEB_PAGE_PREVIEW', 'true').lower() == 'true'
        config.telegram_disable_notification = os.getenv('NOTIFICATION_TELEGRAM_DISABLE_NOTIFICATION', 'false').lower() == 'true'
        config.telegram_protect_content = os.getenv('NOTIFICATION_TELEGRAM_PROTECT_CONTENT', 'false').lower() == 'true'
        
        # Parse Telegram chat IDs
        telegram_chat_ids = os.getenv('NOTIFICATION_TELEGRAM_CHAT_IDS', '')
        if telegram_chat_ids:
            config.telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids.split(',') if cid.strip()]
        
        # SMS configuration
        config.sms_enabled = os.getenv('NOTIFICATION_SMS_ENABLED', 'false').lower() == 'true'
        config.sms_account_sid = os.getenv('NOTIFICATION_SMS_ACCOUNT_SID', '')
        config.sms_auth_token = os.getenv('NOTIFICATION_SMS_AUTH_TOKEN', '')
        config.sms_from_number = os.getenv('NOTIFICATION_SMS_FROM_NUMBER', '')
        config.sms_webhook_url = os.getenv('NOTIFICATION_SMS_WEBHOOK_URL', '')
        config.sms_status_callback = os.getenv('NOTIFICATION_SMS_STATUS_CALLBACK', '')
        
        # Parse SMS recipients
        sms_recipients = os.getenv('NOTIFICATION_SMS_RECIPIENTS', '')
        if sms_recipients:
            config.sms_default_recipients = [r.strip() for r in sms_recipients.split(',') if r.strip()]
        
        # General settings
        config.enable_all = os.getenv('NOTIFICATION_ENABLE_ALL', 'true').lower() == 'true'
        
        # Parse default levels
        default_levels = os.getenv('NOTIFICATION_DEFAULT_LEVELS', 'WARNING,ERROR,CRITICAL')
        if default_levels:
            config.default_levels = [level.strip() for level in default_levels.split(',') if level.strip()]
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'NotificationConfig':
        """Create configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Update configuration with file data
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {file_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Configuration file {file_path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            return cls()
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Email validation
        if self.email_enabled:
            if not self.email_smtp_server:
                errors.append("Email SMTP server is required when email is enabled")
            if not self.email_username:
                errors.append("Email username is required when email is enabled")
            if not self.email_password:
                errors.append("Email password is required when email is enabled")
            if not self.email_from_email:
                errors.append("Email from address is required when email is enabled")
            if not self.email_default_recipients:
                errors.append("Email recipients are required when email is enabled")
        
        # Telegram validation
        if self.telegram_enabled:
            if not self.telegram_bot_token:
                errors.append("Telegram bot token is required when Telegram is enabled")
            if not self.telegram_chat_ids:
                errors.append("Telegram chat IDs are required when Telegram is enabled")
            if self.telegram_parse_mode not in ["HTML", "Markdown", "MarkdownV2"]:
                errors.append("Telegram parse mode must be HTML, Markdown, or MarkdownV2")
        
        # SMS validation
        if self.sms_enabled:
            if not self.sms_account_sid:
                errors.append("SMS account SID is required when SMS is enabled")
            if not self.sms_auth_token:
                errors.append("SMS auth token is required when SMS is enabled")
            if not self.sms_from_number:
                errors.append("SMS from number is required when SMS is enabled")
            if not self.sms_default_recipients:
                errors.append("SMS recipients are required when SMS is enabled")
        
        # General validation
        for level in self.default_levels:
            try:
                NotificationLevel(level)
            except ValueError:
                errors.append(f"Invalid notification level: {level}")
        
        return errors
    
    def get_email_config(self) -> Optional[EmailConfig]:
        """Get email configuration if enabled."""
        if not self.email_enabled:
            return None
        
        return EmailConfig(
            smtp_server=self.email_smtp_server,
            smtp_port=self.email_smtp_port,
            username=self.email_username,
            password=self.email_password,
            use_tls=self.email_use_tls,
            use_ssl=self.email_use_ssl,
            from_email=self.email_from_email,
            from_name=self.email_from_name,
            default_recipients=self.email_default_recipients
        )
    
    def get_telegram_config(self) -> Optional[TelegramConfig]:
        """Get Telegram configuration if enabled."""
        if not self.telegram_enabled:
            return None
        
        return TelegramConfig(
            bot_token=self.telegram_bot_token,
            chat_ids=self.telegram_chat_ids,
            parse_mode=self.telegram_parse_mode,
            disable_web_page_preview=self.telegram_disable_web_page_preview,
            disable_notification=self.telegram_disable_notification,
            protect_content=self.telegram_protect_content
        )
    
    def get_sms_config(self) -> Optional[SMSConfig]:
        """Get SMS configuration if enabled."""
        if not self.sms_enabled:
            return None
        
        return SMSConfig(
            account_sid=self.sms_account_sid,
            auth_token=self.sms_auth_token,
            from_number=self.sms_from_number,
            default_recipients=self.sms_default_recipients,
            webhook_url=self.sms_webhook_url if self.sms_webhook_url else None,
            status_callback=self.sms_status_callback if self.sms_status_callback else None
        )
    
    def get_default_levels(self) -> List[NotificationLevel]:
        """Get default notification levels."""
        return [NotificationLevel(level) for level in self.default_levels]
    
    def is_any_service_enabled(self) -> bool:
        """Check if any notification service is enabled."""
        return self.email_enabled or self.telegram_enabled or self.sms_enabled
    
    def get_enabled_services(self) -> List[str]:
        """Get list of enabled service names."""
        services = []
        if self.email_enabled:
            services.append('email')
        if self.telegram_enabled:
            services.append('telegram')
        if self.sms_enabled:
            services.append('sms')
        return services
    
    def print_configuration(self, show_passwords: bool = False):
        """Print current configuration (for debugging)."""
        print("Notification Configuration:")
        print("=" * 50)
        
        print(f"Enable All: {self.enable_all}")
        print(f"Default Levels: {', '.join(self.default_levels)}")
        print()
        
        # Email configuration
        print("Email Configuration:")
        print(f"  Enabled: {self.email_enabled}")
        if self.email_enabled:
            print(f"  SMTP Server: {self.email_smtp_server}")
            print(f"  SMTP Port: {self.email_smtp_port}")
            print(f"  Username: {self.email_username}")
            print(f"  Password: {'*' * len(self.email_password) if not show_passwords else self.email_password}")
            print(f"  Use TLS: {self.email_use_tls}")
            print(f"  Use SSL: {self.email_use_ssl}")
            print(f"  From Email: {self.email_from_email}")
            print(f"  From Name: {self.email_from_name}")
            print(f"  Recipients: {', '.join(self.email_default_recipients)}")
        print()
        
        # Telegram configuration
        print("Telegram Configuration:")
        print(f"  Enabled: {self.telegram_enabled}")
        if self.telegram_enabled:
            print(f"  Bot Token: {'*' * len(self.telegram_bot_token) if not show_passwords else self.telegram_bot_token}")
            print(f"  Chat IDs: {', '.join(self.telegram_chat_ids)}")
            print(f"  Parse Mode: {self.telegram_parse_mode}")
            print(f"  Disable Web Page Preview: {self.telegram_disable_web_page_preview}")
            print(f"  Disable Notification: {self.telegram_disable_notification}")
            print(f"  Protect Content: {self.telegram_protect_content}")
        print()
        
        # SMS configuration
        print("SMS Configuration:")
        print(f"  Enabled: {self.sms_enabled}")
        if self.sms_enabled:
            print(f"  Account SID: {self.sms_account_sid}")
            print(f"  Auth Token: {'*' * len(self.sms_auth_token) if not show_passwords else self.sms_auth_token}")
            print(f"  From Number: {self.sms_from_number}")
            print(f"  Recipients: {', '.join(self.sms_default_recipients)}")
            if self.sms_webhook_url:
                print(f"  Webhook URL: {self.sms_webhook_url}")
            if self.sms_status_callback:
                print(f"  Status Callback: {self.sms_status_callback}")
        print()


# Predefined configurations for common setups
class NotificationPresets:
    """Predefined notification configurations for common setups."""
    
    @staticmethod
    def development() -> NotificationConfig:
        """Development configuration with minimal notifications."""
        config = NotificationConfig()
        config.enable_all = False  # Disable all notifications in development
        config.default_levels = ["ERROR", "CRITICAL"]  # Only critical errors
        return config
    
    @staticmethod
    def production() -> NotificationConfig:
        """Production configuration with all services enabled."""
        config = NotificationConfig()
        config.enable_all = True
        config.default_levels = ["WARNING", "ERROR", "CRITICAL"]
        return config
    
    @staticmethod
    def gmail_setup(email: str, password: str, recipients: List[str]) -> NotificationConfig:
        """Configuration with Gmail email setup."""
        config = NotificationConfig()
        config.email_enabled = True
        config.email_smtp_server = "smtp.gmail.com"
        config.email_smtp_port = 587
        config.email_username = email
        config.email_password = password
        config.email_use_tls = True
        config.email_from_email = email
        config.email_default_recipients = recipients
        return config
    
    @staticmethod
    def telegram_setup(bot_token: str, chat_ids: List[str]) -> NotificationConfig:
        """Configuration with Telegram setup."""
        config = NotificationConfig()
        config.telegram_enabled = True
        config.telegram_bot_token = bot_token
        config.telegram_chat_ids = chat_ids
        config.telegram_parse_mode = "HTML"
        return config
    
    @staticmethod
    def twilio_setup(account_sid: str, auth_token: str, from_number: str, recipients: List[str]) -> NotificationConfig:
        """Configuration with Twilio SMS setup."""
        config = NotificationConfig()
        config.sms_enabled = True
        config.sms_account_sid = account_sid
        config.sms_auth_token = auth_token
        config.sms_from_number = from_number
        config.sms_default_recipients = recipients
        return config 