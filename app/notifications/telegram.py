"""
Telegram bot notification service.

This module provides Telegram bot notification capabilities using the Telegram Bot API.
Supports sending messages, photos, and documents to individual users or groups.
"""

import requests
import json
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from urllib.parse import urlencode

from .base import BaseNotifier, NotificationMessage, NotificationResult, NotificationError

@dataclass
class TelegramConfig:
    """Configuration for Telegram notification service."""
    bot_token: str
    chat_ids: List[str] = None
    parse_mode: str = "HTML"  # HTML or Markdown
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    protect_content: bool = False
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.chat_ids is None:
            self.chat_ids = []
        
        if self.parse_mode not in ["HTML", "Markdown", "MarkdownV2"]:
            self.parse_mode = "HTML"


class TelegramNotifier(BaseNotifier):
    """Telegram bot notification service."""
    
    def __init__(self, config: TelegramConfig, rate_limit=None):
        """
        Initialize Telegram notifier.
        
        Args:
            config: Telegram configuration
            rate_limit: Optional rate limiter
        """
        super().__init__("TelegramNotifier", rate_limit)
        self.config = config
        self.api_base_url = f"https://api.telegram.org/bot{config.bot_token}"
        
        # Validate configuration
        if not config.bot_token:
            raise NotificationError("Bot token is required")
        
        # Test bot token
        if not self._test_bot_token():
            raise NotificationError("Invalid bot token")
    
    def send(self, message: NotificationMessage) -> NotificationResult:
        """
        Send Telegram notification.
        
        Args:
            message: The notification message to send
            
        Returns:
            NotificationResult with success status and details
        """
        try:
            # Prepare message text
            text = self._format_message(message)
            
            # Get recipients
            recipients = message.recipients or self.config.chat_ids
            if not recipients:
                raise NotificationError("No recipients specified")
            
            # Send to all recipients
            success_count = 0
            failed_count = 0
            message_ids = []
            
            for chat_id in recipients:
                try:
                    result = self._send_message(chat_id, text, message)
                    if result.get('ok'):
                        success_count += 1
                        message_ids.append(str(result['result']['message_id']))
                    else:
                        failed_count += 1
                        self.logger.error(f"Failed to send to {chat_id}: {result.get('description', 'Unknown error')}")
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Exception sending to {chat_id}: {str(e)}")
            
            if success_count > 0:
                self.logger.info(f"Telegram message sent successfully to {success_count}/{len(recipients)} recipients")
                return NotificationResult(
                    success=True,
                    message_id=f"telegram_{','.join(message_ids)}",
                    delivery_time_ms=0  # Will be set by parent method
                )
            else:
                return NotificationResult(
                    success=False,
                    error_message=f"Failed to send to all {len(recipients)} recipients"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {str(e)}", exc_info=True)
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def _format_message(self, message: NotificationMessage) -> str:
        """Format notification message for Telegram."""
        # Level emoji mapping
        level_emojis = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        emoji = level_emojis.get(message.level.value, '📢')
        
        if self.config.parse_mode == "HTML":
            # HTML formatting
            text = f"{emoji} <b>{self._escape_html(message.title)}</b>\n\n"
            text += f"<i>Time:</i> {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            text += f"<i>Level:</i> {message.level.value}\n\n"
            
            # Format body with HTML
            body_lines = message.body.split('\n')
            formatted_body = []
            for line in body_lines:
                line = line.strip()
                if line:
                    # Detect and format key-value pairs
                    if ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key, value = parts
                            formatted_body.append(f"<b>{self._escape_html(key.strip())}:</b> {self._escape_html(value.strip())}")
                        else:
                            formatted_body.append(self._escape_html(line))
                    else:
                        formatted_body.append(self._escape_html(line))
            
            text += '\n'.join(formatted_body)
            
            # Add metadata if available
            if message.metadata:
                text += "\n\n<b>Additional Information:</b>\n"
                for key, value in message.metadata.items():
                    text += f"• <b>{self._escape_html(str(key))}:</b> {self._escape_html(str(value))}\n"
        
        else:
            # Plain text formatting
            text = f"{emoji} {message.title}\n\n"
            text += f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            text += f"Level: {message.level.value}\n\n"
            text += message.body
            
            # Add metadata if available
            if message.metadata:
                text += "\n\nAdditional Information:\n"
                for key, value in message.metadata.items():
                    text += f"• {key}: {value}\n"
        
        return text
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for Telegram."""
        if not text:
            return text
        
        # Replace HTML special characters
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _send_message(self, chat_id: str, text: str, message: NotificationMessage) -> Dict[str, Any]:
        """Send message to specific chat ID."""
        url = f"{self.api_base_url}/sendMessage"
        
        data = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': self.config.parse_mode,
            'disable_web_page_preview': self.config.disable_web_page_preview,
            'disable_notification': self.config.disable_notification,
            'protect_content': self.config.protect_content
        }
        
        response = requests.post(url, data=data, timeout=30)
        return response.json()
    
    def _test_bot_token(self) -> bool:
        """Test if bot token is valid."""
        try:
            url = f"{self.api_base_url}/getMe"
            response = requests.get(url, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                bot_info = result['result']
                self.logger.info(f"Bot connected: @{bot_info['username']} ({bot_info['first_name']})")
                return True
            else:
                self.logger.error(f"Bot token test failed: {result.get('description', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Bot token test exception: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        return self._test_bot_token()
    
    def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat."""
        try:
            url = f"{self.api_base_url}/getChat"
            data = {'chat_id': chat_id}
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                return result['result']
            else:
                self.logger.error(f"Failed to get chat info: {result.get('description', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting chat info: {str(e)}")
            return None
    
    def send_test_message(self, chat_id: str) -> NotificationResult:
        """Send a test message to verify configuration."""
        test_message = NotificationMessage(
            title="HaruPyQuant - Test Message",
            body="This is a test message from HaruPyQuant notification service.",
            level="INFO",
            recipients=[chat_id]
        )
        
        return self.send_message(test_message)
    
    def send_photo(self, chat_id: str, photo_path: str, caption: str = "") -> NotificationResult:
        """Send a photo with optional caption."""
        try:
            url = f"{self.api_base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': self.config.parse_mode
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                result = response.json()
                
                if result.get('ok'):
                    self.logger.info(f"Photo sent successfully to {chat_id}")
                    return NotificationResult(
                        success=True,
                        message_id=f"photo_{result['result']['message_id']}"
                    )
                else:
                    return NotificationResult(
                        success=False,
                        error_message=result.get('description', 'Unknown error')
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to send photo: {str(e)}", exc_info=True)
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def send_document(self, chat_id: str, document_path: str, caption: str = "") -> NotificationResult:
        """Send a document with optional caption."""
        try:
            url = f"{self.api_base_url}/sendDocument"
            
            with open(document_path, 'rb') as document:
                files = {'document': document}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': self.config.parse_mode
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                result = response.json()
                
                if result.get('ok'):
                    self.logger.info(f"Document sent successfully to {chat_id}")
                    return NotificationResult(
                        success=True,
                        message_id=f"doc_{result['result']['message_id']}"
                    )
                else:
                    return NotificationResult(
                        success=False,
                        error_message=result.get('description', 'Unknown error')
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to send document: {str(e)}", exc_info=True)
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def get_updates(self, offset: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get updates from Telegram (useful for getting chat IDs)."""
        try:
            url = f"{self.api_base_url}/getUpdates"
            params = {'limit': limit}
            if offset:
                params['offset'] = offset
            
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                return result['result']
            else:
                self.logger.error(f"Failed to get updates: {result.get('description', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting updates: {str(e)}")
            return [] 