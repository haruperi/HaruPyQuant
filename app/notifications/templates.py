"""
Notification templates for different types of alerts and messages.

This module provides pre-defined templates for common notification types
and allows for custom template creation and management.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import NotificationMessage, NotificationError

from app.util import get_logger

logger = get_logger(__name__)


class NotificationTemplate:
    """Template system for notification messages."""
    
    def __init__(self):
        """Initialize with default templates."""
        self.templates = {
            # Trading alerts
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
            
            "trading_signal": {
                "title": "Trading Signal: {symbol} {signal_type}",
                "body": """
                📊 New Trade Signal Detected!

                🕒 {timestamp}

                💡 {signal_type} {symbol} @ {entry_price}

                ❌ Stop Loss: {stop_loss}
                🛑 {stop_loss_pips} pips

                💵 Take Profit: {take_profit}
                🎯 {take_profit_pips} pips

                💼 Lots: {lots}
                🧾 Strategy: {strategy}
                💪 Signal Strength: {strength}

                📏 ADR: {adr}
                📐 Range: {range}%

                📉 Current VAR: ${current_var}
                🎯 Proposed VAR: ${proposed_var}
                ⚖️ VAR Difference: {var_difference}%
                                """.strip()
                            },
            
            "position_opened": {
                "title": "Position Opened: {symbol} {direction}",
                "body": """
✅ Position Opened

Symbol: {symbol}
Direction: {direction}
Size: {size}
Entry Price: {entry_price}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Time: {timestamp}

Account: {account}
Strategy: {strategy}
Risk: {risk_amount}
                """.strip()
            },
            
            "position_closed": {
                "title": "Position Closed: {symbol} {direction}",
                "body": """
🔚 Position Closed

Symbol: {symbol}
Direction: {direction}
Size: {size}
Entry Price: {entry_price}
Exit Price: {exit_price}
P&L: {pnl}
P&L %: {pnl_percent}
Duration: {duration}
Time: {timestamp}

Account: {account}
Strategy: {strategy}
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
            
            # System alerts
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
            
            "system_startup": {
                "title": "System Startup: HaruPyQuant",
                "body": """
🚀 System Startup

HaruPyQuant trading system has started successfully.

Startup Time: {timestamp}
Version: {version}
Environment: {environment}
Account: {account}

Services Status:
- MT5 Connection: {mt5_status}
- Data Feed: {data_feed_status}
- Strategy Engine: {strategy_status}
- Risk Manager: {risk_manager_status}
                """.strip()
            },
            
            "system_shutdown": {
                "title": "System Shutdown: HaruPyQuant",
                "body": """
🛑 System Shutdown

HaruPyQuant trading system is shutting down.

Shutdown Time: {timestamp}
Reason: {reason}
Duration: {duration}

Final Status:
- Open Positions: {open_positions}
- Account Balance: {account_balance}
- Daily P&L: {daily_pnl}
                """.strip()
            },
            
            "connection_lost": {
                "title": "Connection Lost: {service}",
                "body": """
❌ Connection Lost

Service: {service}
Lost At: {timestamp}
Error: {error_message}

Attempting to reconnect...
Retry Count: {retry_count}
Next Retry: {next_retry}
                """.strip()
            },
            
            "connection_restored": {
                "title": "Connection Restored: {service}",
                "body": """
✅ Connection Restored

Service: {service}
Restored At: {timestamp}
Downtime: {downtime}

Status: Active
                """.strip()
            },
            
            # Error alerts
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
            },
            
            "strategy_error": {
                "title": "Strategy Error: {strategy_name}",
                "body": """
⚠️ Strategy Error

Strategy: {strategy_name}
Error: {error_message}
Symbol: {symbol}
Time: {timestamp}

Action: {action}
Status: {status}
                """.strip()
            },
            
            # Performance alerts
            "performance_alert": {
                "title": "Performance Alert: {alert_type}",
                "body": """
📈 Performance Alert

Type: {alert_type}
Metric: {metric}
Value: {value}
Threshold: {threshold}
Time: {timestamp}

Period: {period}
Account: {account}
                """.strip()
            },
            
            "drawdown_alert": {
                "title": "Drawdown Alert: {drawdown_type}",
                "body": """
📉 Drawdown Alert

Type: {drawdown_type}
Current Drawdown: {current_drawdown}%
Peak Drawdown: {peak_drawdown}%
Duration: {duration}
Time: {timestamp}

Account: {account}
Balance: {balance}
Equity: {equity}
                """.strip()
            },
            
            # Market alerts
            "market_alert": {
                "title": "Market Alert: {symbol}",
                "body": """
🌍 Market Alert

Symbol: {symbol}
Event: {event}
Price: {price}
Time: {timestamp}

Impact: {impact}
Details: {details}
                """.strip()
            },
            
            "news_alert": {
                "title": "News Alert: {headline}",
                "body": """
📰 News Alert

Headline: {headline}
Source: {source}
Time: {timestamp}
Impact: {impact}

Summary: {summary}
Related Symbols: {symbols}
                """.strip()
            },
            
            # Risk alerts
            "risk_alert": {
                "title": "Risk Alert: {risk_type}",
                "body": """
⚠️ Risk Alert

Type: {risk_type}
Severity: {severity}
Message: {message}
Time: {timestamp}

Account: {account}
Current Risk: {current_risk}
Max Risk: {max_risk}
Action: {action}
                """.strip()
            },
            
            "margin_alert": {
                "title": "Margin Alert: {account}",
                "body": """
💰 Margin Alert

Account: {account}
Margin Level: {margin_level}%
Free Margin: {free_margin}
Used Margin: {used_margin}
Time: {timestamp}

Warning Level: {warning_level}%
Action Required: {action}
                """.strip()
            },
            
            # Custom templates
            "custom_message": {
                "title": "{title}",
                "body": "{body}"
            },
            
            "test_message": {
                "title": "Test Message: {service}",
                "body": """
🧪 Test Message

Service: {service}
Time: {timestamp}
Status: {status}

This is a test message to verify notification configuration.
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
    
    def update_template(self, name: str, title_template: Optional[str] = None, body_template: Optional[str] = None):
        """Update an existing template."""
        if name not in self.templates:
            raise NotificationError(f"Template '{name}' not found")
        
        if title_template:
            self.templates[name]['title'] = title_template
        if body_template:
            self.templates[name]['body'] = body_template
        
        logger.info(f"Updated notification template: {name}")
    
    def remove_template(self, name: str):
        """Remove a template."""
        if name not in self.templates:
            raise NotificationError(f"Template '{name}' not found")
        
        # Don't allow removal of essential templates
        essential_templates = ['trading_alert', 'system_alert', 'error_alert', 'custom_message']
        if name in essential_templates:
            raise NotificationError(f"Cannot remove essential template: {name}")
        
        del self.templates[name]
        logger.info(f"Removed notification template: {name}")
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """Get list of variables required by a template."""
        template = self.get_template(template_name)
        
        # Extract variables from both title and body
        import re
        variables = set()
        
        for text in [template['title'], template['body']]:
            matches = re.findall(r'\{(\w+)\}', text)
            variables.update(matches)
        
        return sorted(list(variables))
    
    def validate_template(self, template_name: str, **kwargs) -> List[str]:
        """Validate that all required variables are provided."""
        required_vars = self.get_template_variables(template_name)
        missing_vars = []
        
        for var in required_vars:
            if var not in kwargs:
                missing_vars.append(var)
        
        return missing_vars
    
    def preview_template(self, template_name: str, **kwargs) -> str:
        """Preview a template without creating a NotificationMessage."""
        template = self.get_template(template_name)
        
        # Add timestamp if not provided
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            title = template['title'].format(**kwargs)
            body = template['body'].format(**kwargs)
            
            return f"Title: {title}\n\nBody:\n{body}"
            
        except KeyError as e:
            raise NotificationError(f"Missing required template variable: {e}")
    
    def export_templates(self) -> Dict[str, Dict[str, str]]:
        """Export all templates for backup or sharing."""
        return self.templates.copy()
    
    def import_templates(self, templates: Dict[str, Dict[str, str]], overwrite: bool = False):
        """Import templates from a dictionary."""
        for name, template in templates.items():
            if name in self.templates and not overwrite:
                logger.warning(f"Template '{name}' already exists, skipping (use overwrite=True to replace)")
                continue
            
            if 'title' not in template or 'body' not in template:
                logger.warning(f"Template '{name}' missing required fields, skipping")
                continue
            
            self.templates[name] = template
            logger.info(f"Imported template: {name}")
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        if template_name not in self.templates:
            raise NotificationError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        variables = self.get_template_variables(template_name)
        
        return {
            'name': template_name,
            'title_template': template['title'],
            'body_template': template['body'],
            'required_variables': variables,
            'title_length': len(template['title']),
            'body_length': len(template['body']),
            'total_length': len(template['title']) + len(template['body'])
        } 