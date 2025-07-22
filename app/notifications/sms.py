"""
SMS notification service using Twilio.

This module provides SMS notification capabilities using the Twilio API.
Supports sending SMS messages to phone numbers worldwide.
"""

import requests
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlencode

from .base import BaseNotifier, NotificationMessage, NotificationResult, NotificationError

@dataclass
class SMSConfig:
    """Configuration for SMS notification service."""
    account_sid: str
    auth_token: str
    from_number: str
    default_recipients: List[str] = None
    webhook_url: Optional[str] = None
    status_callback: Optional[str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.default_recipients is None:
            self.default_recipients = []


class SMSNotifier(BaseNotifier):
    """SMS notification service using Twilio."""
    
    def __init__(self, config: SMSConfig, rate_limit=None):
        """
        Initialize SMS notifier.
        
        Args:
            config: SMS configuration
            rate_limit: Optional rate limiter
        """
        super().__init__("SMSNotifier", rate_limit)
        self.config = config
        self.api_base_url = f"https://api.twilio.com/2010-04-01/Accounts/{config.account_sid}"
        
        # Validate configuration
        if not all([config.account_sid, config.auth_token, config.from_number]):
            raise NotificationError("Account SID, auth token, and from number are required")
        
        # Test credentials
        if not self._test_credentials():
            raise NotificationError("Invalid Twilio credentials")
    
    def send(self, message: NotificationMessage) -> NotificationResult:
        """
        Send SMS notification.
        
        Args:
            message: The notification message to send
            
        Returns:
            NotificationResult with success status and details
        """
        try:
            # Prepare message text
            text = self._format_message(message)
            
            # Get recipients
            recipients = message.recipients or self.config.default_recipients
            if not recipients:
                raise NotificationError("No recipients specified")
            
            # Send to all recipients
            success_count = 0
            failed_count = 0
            message_ids = []
            
            for phone_number in recipients:
                try:
                    result = self._send_sms(phone_number, text, message)
                    if result.get('status') == 'queued':
                        success_count += 1
                        message_ids.append(result.get('sid', ''))
                    else:
                        failed_count += 1
                        self.logger.error(f"Failed to send to {phone_number}: {result.get('error_message', 'Unknown error')}")
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Exception sending to {phone_number}: {str(e)}")
            
            if success_count > 0:
                self.logger.info(f"SMS sent successfully to {success_count}/{len(recipients)} recipients")
                return NotificationResult(
                    success=True,
                    message_id=f"sms_{','.join(message_ids)}",
                    delivery_time_ms=0  # Will be set by parent method
                )
            else:
                return NotificationResult(
                    success=False,
                    error_message=f"Failed to send to all {len(recipients)} recipients"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {str(e)}", exc_info=True)
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def _format_message(self, message: NotificationMessage) -> str:
        """Format notification message for SMS."""
        # Level emoji mapping
        level_emojis = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        emoji = level_emojis.get(message.level.value, '📢')
        
        # SMS format (limited to 160 characters for single SMS)
        text = f"{emoji} {message.title}\n"
        text += f"Time: {message.timestamp.strftime('%H:%M UTC')}\n"
        text += f"Level: {message.level.value}\n\n"
        
        # Truncate body if too long
        body = message.body
        max_body_length = 160 - len(text) - 20  # Reserve space for metadata
        
        if len(body) > max_body_length:
            body = body[:max_body_length-3] + "..."
        
        text += body
        
        # Add critical metadata if space allows
        if message.metadata and len(text) < 140:
            text += "\n\n"
            for key, value in list(message.metadata.items())[:2]:  # Limit to 2 metadata items
                if len(text) + len(f"{key}:{value}\n") < 160:
                    text += f"{key}:{value}\n"
                else:
                    break
        
        return text
    
    def _send_sms(self, to_number: str, text: str, message: NotificationMessage) -> Dict[str, Any]:
        """Send SMS to specific phone number."""
        url = f"{self.api_base_url}/Messages.json"
        
        data = {
            'To': to_number,
            'From': self.config.from_number,
            'Body': text
        }
        
        # Add optional parameters
        if self.config.webhook_url:
            data['WebhookUrl'] = self.config.webhook_url
        
        if self.config.status_callback:
            data['StatusCallback'] = self.config.status_callback
        
        # Make request with basic auth
        response = requests.post(
            url, 
            data=data,
            auth=(self.config.account_sid, self.config.auth_token),
            timeout=30
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            error_data = response.json()
            return {
                'status': 'failed',
                'error_message': error_data.get('message', f'HTTP {response.status_code}')
            }
    
    def _test_credentials(self) -> bool:
        """Test if Twilio credentials are valid."""
        try:
            url = f"{self.api_base_url}.json"
            response = requests.get(
                url,
                auth=(self.config.account_sid, self.config.auth_token),
                timeout=10
            )
            
            if response.status_code == 200:
                account_info = response.json()
                self.logger.info(f"Twilio account connected: {account_info.get('friendly_name', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Twilio credentials test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Twilio credentials test exception: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test Twilio connection."""
        return self._test_credentials()
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get Twilio account information."""
        try:
            url = f"{self.api_base_url}.json"
            response = requests.get(
                url,
                auth=(self.config.account_sid, self.config.auth_token),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get account info: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting account info: {str(e)}")
            return None
    
    def send_test_sms(self, phone_number: str) -> NotificationResult:
        """Send a test SMS to verify configuration."""
        test_message = NotificationMessage(
            title="HaruPyQuant Test",
            body="This is a test SMS from HaruPyQuant notification service.",
            level="INFO",
            recipients=[phone_number]
        )
        
        return self.send_message(test_message)
    
    def get_message_status(self, message_sid: str) -> Optional[Dict[str, Any]]:
        """Get the status of a sent message."""
        try:
            url = f"{self.api_base_url}/Messages/{message_sid}.json"
            response = requests.get(
                url,
                auth=(self.config.account_sid, self.config.auth_token),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get message status: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting message status: {str(e)}")
            return None
    
    def get_message_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent message history."""
        try:
            url = f"{self.api_base_url}/Messages.json"
            params = {'PageSize': limit}
            
            response = requests.get(
                url,
                params=params,
                auth=(self.config.account_sid, self.config.auth_token),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('messages', [])
            else:
                self.logger.error(f"Failed to get message history: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting message history: {str(e)}")
            return []
    
    def validate_phone_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Validate a phone number using Twilio's Lookup API."""
        try:
            url = "https://lookups.twilio.com/v2/PhoneNumbers"
            params = {'PhoneNumber': phone_number}
            
            response = requests.get(
                url,
                params=params,
                auth=(self.config.account_sid, self.config.auth_token),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to validate phone number: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception validating phone number: {str(e)}")
            return None 