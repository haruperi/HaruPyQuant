"""
Live Trading Configuration

This module defines the configuration structure for the live trading system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import time
from enum import Enum

from app.util import get_logger

logger = get_logger(__name__)


class TradingMode(Enum):
    """Trading modes."""
    DEMO = "demo"
    LIVE = "live"
    PAPER = "paper"


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_positions: int = 3
    max_daily_trades: int = 10
    max_daily_loss: float = 100.0  # USD
    max_drawdown: float = 5.0  # Percentage


@dataclass
class ExecutionConfig:
    """Configuration for trade execution."""
    max_slippage: int = 3  # Points
    max_deviation: int = 5  # Points
    retry_attempts: int = 3
    retry_delay: float = 1.0  # Seconds
    execution_timeout: float = 30.0  # Seconds
    use_market_orders: bool = True
    confirm_trades: bool = False


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_risk_per_trade: float = 1.0  # Percentage
    max_portfolio_risk: float = 5.0  # Percentage
    max_correlation: float = 0.7
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    stop_loss_atr_multiplier: float = 2.0
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiplier: float = 1.5


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    health_check_interval: float = 60.0  # Seconds
    performance_update_interval: float = 300.0  # Seconds
    position_update_interval: float = 30.0  # Seconds
    connection_check_interval: float = 30.0  # Seconds
    max_cpu_usage: float = 80.0  # Percentage
    max_memory_usage: float = 80.0  # Percentage
    max_disk_usage: float = 90.0  # Percentage


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled: bool = True
    services: List[str] = field(default_factory=lambda: ["email", "telegram"])
    trade_notifications: bool = True
    error_notifications: bool = True
    performance_notifications: bool = True
    system_notifications: bool = True
    notification_levels: List[str] = field(default_factory=lambda: ["WARNING", "ERROR", "CRITICAL"])


@dataclass
class ScheduleConfig:
    """Configuration for trading schedule."""
    enabled: bool = True
    trading_hours: Dict[str, List[time]] = field(default_factory=dict)
    timezone: str = "UTC"
    weekend_trading: bool = False
    holiday_trading: bool = False
    market_holidays: List[str] = field(default_factory=list)


@dataclass
class LiveTradingConfig:
    """Main configuration for live trading system."""
    
    # Basic settings
    mode: TradingMode = TradingMode.DEMO
    account_name: str = "Demo Account"
    broker_name: str = "MT5"
    
    # Strategy configuration
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # Execution configuration
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Risk configuration
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Notification configuration
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Schedule configuration
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    # Advanced settings
    data_update_interval: float = 5.0  # Seconds
    strategy_update_interval: float = 60.0  # Seconds
    log_level: str = "INFO"
    enable_debug: bool = False
    save_trades: bool = True
    save_performance: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.risk.max_risk_per_trade > 10.0:
            logger.warning("Max risk per trade is very high (>10%)")
        
        if self.risk.max_portfolio_risk > 20.0:
            logger.warning("Max portfolio risk is very high (>20%)")
        
        if self.monitoring.health_check_interval < 10.0:
            logger.warning("Health check interval is very short (<10s)")
        
        if self.execution.retry_attempts > 10:
            logger.warning("Too many retry attempts (>10)")
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy."""
        return self.strategies.get(strategy_name)
    
    def add_strategy(self, strategy_config: StrategyConfig):
        """Add a strategy configuration."""
        self.strategies[strategy_config.name] = strategy_config
        logger.info(f"Added strategy configuration: {strategy_config.name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy configuration."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Removed strategy configuration: {strategy_name}")
            return True
        return False
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            logger.info(f"Enabled strategy: {strategy_name}")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            logger.info(f"Disabled strategy: {strategy_name}")
            return True
        return False
    
    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategies."""
        return [name for name, config in self.strategies.items() if config.enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode.value,
            'account_name': self.account_name,
            'broker_name': self.broker_name,
            'strategies': {name: {
                'enabled': config.enabled,
                'symbols': config.symbols,
                'parameters': config.parameters,
                'risk_level': config.risk_level.value,
                'max_positions': config.max_positions,
                'max_daily_trades': config.max_daily_trades,
                'max_daily_loss': config.max_daily_loss,
                'max_drawdown': config.max_drawdown
            } for name, config in self.strategies.items()},
            'execution': {
                'max_slippage': self.execution.max_slippage,
                'max_deviation': self.execution.max_deviation,
                'retry_attempts': self.execution.retry_attempts,
                'retry_delay': self.execution.retry_delay,
                'execution_timeout': self.execution.execution_timeout,
                'use_market_orders': self.execution.use_market_orders,
                'confirm_trades': self.execution.confirm_trades
            },
            'risk': {
                'max_risk_per_trade': self.risk.max_risk_per_trade,
                'max_portfolio_risk': self.risk.max_portfolio_risk,
                'max_correlation': self.risk.max_correlation,
                'max_positions_per_symbol': self.risk.max_positions_per_symbol,
                'max_total_positions': self.risk.max_total_positions,
                'stop_loss_atr_multiplier': self.risk.stop_loss_atr_multiplier,
                'trailing_stop_enabled': self.risk.trailing_stop_enabled,
                'trailing_stop_atr_multiplier': self.risk.trailing_stop_atr_multiplier
            },
            'monitoring': {
                'health_check_interval': self.monitoring.health_check_interval,
                'performance_update_interval': self.monitoring.performance_update_interval,
                'position_update_interval': self.monitoring.position_update_interval,
                'connection_check_interval': self.monitoring.connection_check_interval,
                'max_cpu_usage': self.monitoring.max_cpu_usage,
                'max_memory_usage': self.monitoring.max_memory_usage,
                'max_disk_usage': self.monitoring.max_disk_usage
            },
            'notifications': {
                'enabled': self.notifications.enabled,
                'services': self.notifications.services,
                'trade_notifications': self.notifications.trade_notifications,
                'error_notifications': self.notifications.error_notifications,
                'performance_notifications': self.notifications.performance_notifications,
                'system_notifications': self.notifications.system_notifications,
                'notification_levels': self.notifications.notification_levels
            },
            'schedule': {
                'enabled': self.schedule.enabled,
                'timezone': self.schedule.timezone,
                'weekend_trading': self.schedule.weekend_trading,
                'holiday_trading': self.schedule.holiday_trading,
                'market_holidays': self.schedule.market_holidays
            },
            'data_update_interval': self.data_update_interval,
            'strategy_update_interval': self.strategy_update_interval,
            'log_level': self.log_level,
            'enable_debug': self.enable_debug,
            'save_trades': self.save_trades,
            'save_performance': self.save_performance
        } 