"""
Email notification service using SMTP.

This module provides email notification capabilities using SMTP protocol.
Supports various email providers including Gmail, Outlook, and custom SMTP servers.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseNotifier, NotificationMessage, NotificationResult, NotificationError

@dataclass
class EmailConfig:
    """Configuration for email notification service."""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    default_recipients: List[str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.from_email is None:
            self.from_email = self.username
        if self.default_recipients is None:
            self.default_recipients = []


class EmailNotifier(BaseNotifier):
    """Email notification service using SMTP."""
    
    def __init__(self, config: EmailConfig, rate_limit=None):
        """
        Initialize email notifier.
        
        Args:
            config: Email configuration
            rate_limit: Optional rate limiter
        """
        super().__init__("EmailNotifier", rate_limit)
        self.config = config
        self._connection = None
        
        # Validate configuration
        if not all([config.smtp_server, config.smtp_port, config.username, config.password]):
            raise NotificationError("SMTP server, port, username, and password are required")
    
    def send(self, message: NotificationMessage) -> NotificationResult:
        """
        Send email notification.
        
        Args:
            message: The notification message to send
            
        Returns:
            NotificationResult with success status and details
        """
        try:
            # Create email message
            email_msg = self._create_email_message(message)
            
            # Send email
            with self._get_connection() as server:
                recipients = message.recipients or self.config.default_recipients
                if not recipients:
                    raise NotificationError("No recipients specified")
                
                server.send_message(email_msg)
                
                self.logger.info(f"Email sent successfully to {len(recipients)} recipients")
                return NotificationResult(
                    success=True,
                    message_id=f"email_{message.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    delivery_time_ms=0  # Will be set by parent method
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}", exc_info=True)
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_email_message(self, message: NotificationMessage) -> MIMEMultipart:
        """Create email message from notification message."""
        # Create message container
        email_msg = MIMEMultipart('alternative')
        
        # Set headers
        email_msg['Subject'] = message.title
        email_msg['From'] = self._get_from_address()
        
        # Set recipients
        recipients = message.recipients or self.config.default_recipients
        email_msg['To'] = ', '.join(recipients)
        
        # Create HTML version
        html_body = self._create_html_body(message)
        html_part = MIMEText(html_body, 'html')
        email_msg.attach(html_part)
        
        # Create plain text version
        text_body = self._create_text_body(message)
        text_part = MIMEText(text_body, 'plain')
        email_msg.attach(text_part)
        
        return email_msg
    
    def _create_html_body(self, message: NotificationMessage) -> str:
        """Create HTML version of email body."""
        level_colors = {
            'DEBUG': '#6c757d',
            'INFO': '#17a2b8', 
            'WARNING': '#ffc107',
            'ERROR': '#dc3545',
            'CRITICAL': '#721c24'
        }
        
        color = level_colors.get(message.level.value, '#6c757d')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }}
                .metadata {{ background-color: #e9ecef; padding: 10px; border-radius: 3px; margin-top: 10px; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{message.title}</h2>
                <div class="timestamp">{message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            <div class="content">
                {message.body.replace(chr(10), '<br>')}
            </div>
        """
        
        # Add metadata if available
        if message.metadata:
            html += '<div class="metadata"><h4>Additional Information:</h4><ul>'
            for key, value in message.metadata.items():
                html += f'<li><strong>{key}:</strong> {value}</li>'
            html += '</ul></div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _create_text_body(self, message: NotificationMessage) -> str:
        """Create plain text version of email body."""
        text = f"{message.title}\n"
        text += "=" * len(message.title) + "\n\n"
        text += f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        text += f"Level: {message.level.value}\n\n"
        text += message.body + "\n\n"
        
        # Add metadata if available
        if message.metadata:
            text += "Additional Information:\n"
            text += "-" * 20 + "\n"
            for key, value in message.metadata.items():
                text += f"{key}: {value}\n"
        
        return text
    
    def _get_from_address(self) -> str:
        """Get the from address for emails."""
        if self.config.from_name:
            return f"{self.config.from_name} <{self.config.from_email}>"
        return self.config.from_email
    
    def _get_connection(self):
        """Get SMTP connection."""
        if self.config.use_ssl:
            context = ssl.create_default_context()
            return smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port, context=context)
        else:
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            if self.config.use_tls:
                server.starttls()
            return server
    
    def test_connection(self) -> bool:
        """Test SMTP connection."""
        try:
            with self._get_connection() as server:
                server.login(self.config.username, self.config.password)
                self.logger.info("Email connection test successful")
                return True
        except Exception as e:
            self.logger.error(f"Email connection test failed: {str(e)}")
            return False
    
    def send_test_email(self, recipient: str) -> NotificationResult:
        """Send a test email to verify configuration."""
        test_message = NotificationMessage(
            title="HaruPyQuant - Test Email",
            body="This is a test email from HaruPyQuant notification service.",
            level="INFO",
            recipients=[recipient]
        )
        
        return self.send_message(test_message)


# Predefined configurations for common email providers
class EmailProviders:
    """Predefined configurations for common email providers."""
    
    @staticmethod
    def gmail(username: str, password: str, app_password: Optional[str] = None) -> EmailConfig:
        """Gmail configuration."""
        return EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=username,
            password=app_password or password,
            use_tls=True,
            from_email=username
        )
    
    @staticmethod
    def outlook(username: str, password: str) -> EmailConfig:
        """Outlook/Hotmail configuration."""
        return EmailConfig(
            smtp_server="smtp-mail.outlook.com",
            smtp_port=587,
            username=username,
            password=password,
            use_tls=True,
            from_email=username
        )
    
    @staticmethod
    def yahoo(username: str, password: str, app_password: Optional[str] = None) -> EmailConfig:
        """Yahoo Mail configuration."""
        return EmailConfig(
            smtp_server="smtp.mail.yahoo.com",
            smtp_port=587,
            username=username,
            password=app_password or password,
            use_tls=True,
            from_email=username
        )
    
    @staticmethod
    def custom(smtp_server: str, smtp_port: int, username: str, password: str, 
               use_tls: bool = True, use_ssl: bool = False) -> EmailConfig:
        """Custom SMTP server configuration."""
        return EmailConfig(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            use_tls=use_tls,
            use_ssl=use_ssl,
            from_email=username
        ) 