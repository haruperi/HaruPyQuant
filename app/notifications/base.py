"""
Base classes and interfaces for notification services.

This module defines the core abstractions and base classes used by all notification services.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from app.util import get_logger

logger = get_logger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationError(Exception):
    """Base exception for notification-related errors."""
    pass


@dataclass
class NotificationMessage:
    """Represents a notification message with metadata."""
    title: str
    body: str
    level: NotificationLevel = NotificationLevel.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    recipients: List[str] = field(default_factory=list)
    template_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate message after initialization."""
        if not self.title or not self.body:
            raise NotificationError("Title and body are required")
        
        if not isinstance(self.level, NotificationLevel):
            self.level = NotificationLevel(self.level)


@dataclass
class NotificationResult:
    """Result of a notification attempt."""
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    delivery_time_ms: Optional[int] = None


class RateLimiter:
    """Rate limiter for notification services to prevent spam."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[datetime] = []
        self.lock = threading.Lock()
    
    def can_send(self) -> bool:
        """Check if a notification can be sent based on rate limits."""
        with self.lock:
            now = datetime.now()
            # Remove old requests outside the time window
            self.requests = [req for req in self.requests 
                           if now - req < timedelta(seconds=self.time_window)]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def get_wait_time(self) -> int:
        """Get the number of seconds to wait before next request."""
        with self.lock:
            if not self.requests:
                return 0
            
            oldest_request = min(self.requests)
            wait_time = self.time_window - (datetime.now() - oldest_request).seconds
            return max(0, wait_time)


class BaseNotifier(ABC):
    """Abstract base class for all notification services."""
    
    def __init__(self, name: str, rate_limit: Optional[RateLimiter] = None):
        """
        Initialize base notifier.
        
        Args:
            name: Name of the notifier service
            rate_limit: Optional rate limiter instance
        """
        self.name = name
        self.rate_limit = rate_limit or RateLimiter()
        self.logger = get_logger(f"{__name__}.{name}")
        self.enabled = True
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
    
    @abstractmethod
    def send(self, message: NotificationMessage) -> NotificationResult:
        """
        Send a notification message.
        
        Args:
            message: The notification message to send
            
        Returns:
            NotificationResult with success status and details
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the notification service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending."""
        if not self.rate_limit.can_send():
            wait_time = self.rate_limit.get_wait_time()
            self.logger.warning(f"Rate limit exceeded. Wait {wait_time} seconds")
            return False
        return True
    
    def _send_with_retry(self, message: NotificationMessage) -> NotificationResult:
        """Send message with retry logic."""
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                if not self._check_rate_limit():
                    return NotificationResult(
                        success=False,
                        error_message="Rate limit exceeded"
                    )
                
                result = self.send(message)
                result.retry_count = attempt
                result.delivery_time_ms = int((time.time() - start_time) * 1000)
                
                if result.success:
                    self.logger.info(f"Notification sent successfully on attempt {attempt + 1}")
                    return result
                else:
                    self.logger.warning(f"Notification failed on attempt {attempt + 1}: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Exception on attempt {attempt + 1}: {str(e)}", exc_info=True)
                result = NotificationResult(
                    success=False,
                    error_message=str(e),
                    retry_count=attempt
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        # All attempts failed
        result.delivery_time_ms = int((time.time() - start_time) * 1000)
        return result
    
    def send_message(self, message: NotificationMessage) -> NotificationResult:
        """
        Public method to send a message with retry logic.
        
        Args:
            message: The notification message to send
            
        Returns:
            NotificationResult with success status and details
        """
        if not self.enabled:
            return NotificationResult(
                success=False,
                error_message="Notifier is disabled"
            )
        
        self.logger.info(f"Sending notification: {message.title}")
        return self._send_with_retry(message)
    
    def enable(self):
        """Enable the notifier."""
        self.enabled = True
        self.logger.info("Notifier enabled")
    
    def disable(self):
        """Disable the notifier."""
        self.enabled = False
        self.logger.info("Notifier disabled")
    
    def is_enabled(self) -> bool:
        """Check if notifier is enabled."""
        return self.enabled


class NotificationTemplate:
    """Template system for notification messages."""
    
    def __init__(self):
        self.templates = {
            "trading_alert": {
                "title": "Trading Alert: {symbol} {action}",
                "body": """
🔔 Trading Alert

Symbol: {symbol}
Action: {action}
Price: {price}
Reason: {reason}
Time: {timestamp}

Account: {account}
Strategy: {strategy}
Risk Level: {risk_level}
                """.strip()
            },
            "system_alert": {
                "title": "System Alert: {level} - {message}",
                "body": """
⚠️ System Alert

Level: {level}
Message: {message}
Details: {details}
Time: {timestamp}

Component: {component}
Status: {status}
                """.strip()
            },
            "position_update": {
                "title": "Position Update: {symbol}",
                "body": """
📊 Position Update

Symbol: {symbol}
Type: {position_type}
Size: {size}
Entry Price: {entry_price}
Current Price: {current_price}
P&L: {pnl}
P&L %: {pnl_percent}
Time: {timestamp}
                """.strip()
            },
            "error_alert": {
                "title": "Error Alert: {error_type}",
                "body": """
❌ Error Alert

Type: {error_type}
Message: {message}
Component: {component}
Time: {timestamp}
Stack Trace: {stack_trace}
                """.strip()
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, str]:
        """Get a template by name."""
        if template_name not in self.templates:
            raise NotificationError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def render(self, template_name: str, **kwargs) -> NotificationMessage:
        """Render a template with provided variables."""
        template = self.get_template(template_name)
        
        # Add timestamp if not provided
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            title = template['title'].format(**kwargs)
            body = template['body'].format(**kwargs)
        except KeyError as e:
            raise NotificationError(f"Missing required template variable: {e}")
        
        return NotificationMessage(
            title=title,
            body=body,
            template_name=template_name,
            metadata=kwargs
        )
    
    def add_template(self, name: str, title_template: str, body_template: str):
        """Add a new template."""
        self.templates[name] = {
            'title': title_template,
            'body': body_template
        }
        logger.info(f"Added notification template: {name}")
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys()) 