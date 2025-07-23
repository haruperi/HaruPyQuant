"""
Risk Monitor

This module handles risk management and monitoring for the live trading system.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.trading.risk_manager import RiskManager
from app.trading.trader import Trader
from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """Risk alert event."""
    id: str
    level: RiskLevel
    message: str
    component: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class RiskMetrics:
    """Risk metrics."""
    total_risk: float
    portfolio_var: float
    max_drawdown: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    margin_utilization: float
    timestamp: datetime = field(default_factory=datetime.now)


class RiskMonitor:
    """Monitors and manages trading risk."""
    
    def __init__(self, trader: Trader, risk_manager: RiskManager, 
                 notification_manager: Optional[NotificationManager] = None):
        """
        Initialize risk monitor.
        
        Args:
            trader: Trader instance
            risk_manager: Risk manager instance
            notification_manager: Notification manager for alerts
        """
        self.trader = trader
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.risk_metrics: List[RiskMetrics] = []
        self.risk_thresholds: Dict[str, float] = {
            'max_portfolio_risk': 5.0,  # Percentage
            'max_drawdown': 10.0,  # Percentage
            'max_correlation': 0.7,
            'max_concentration': 20.0,  # Percentage
            'max_leverage': 3.0,
            'max_margin_utilization': 80.0  # Percentage
        }
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'high_risk_alerts': 0,
            'critical_alerts': 0,
            'risk_checks': 0
        }
        
        self.logger.info("Risk monitor initialized")
    
    def start(self):
        """Start the risk monitor."""
        if self._running:
            self.logger.warning("Risk monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="RiskMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Risk monitor started")
    
    def stop(self):
        """Stop the risk monitor."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        self.logger.info("Risk monitor stopped")
    
    def set_risk_threshold(self, threshold_name: str, value: float):
        """Set a risk threshold."""
        if threshold_name in self.risk_thresholds:
            self.risk_thresholds[threshold_name] = value
            self.logger.info(f"Risk threshold updated: {threshold_name} = {value}")
    
    def get_risk_threshold(self, threshold_name: str) -> Optional[float]:
        """Get a risk threshold."""
        return self.risk_thresholds.get(threshold_name)
    
    def check_risk(self) -> RiskMetrics:
        """Perform comprehensive risk check."""
        try:
            # Get current positions
            positions = self.trader.open_positions
            
            # Calculate risk metrics
            total_risk = self._calculate_total_risk(positions)
            portfolio_var = self._calculate_portfolio_var(positions)
            max_drawdown = self._calculate_max_drawdown(positions)
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            leverage_ratio = self._calculate_leverage_ratio(positions)
            margin_utilization = self._calculate_margin_utilization(positions)
            
            # Create risk metrics
            metrics = RiskMetrics(
                total_risk=total_risk,
                portfolio_var=portfolio_var,
                max_drawdown=max_drawdown,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                margin_utilization=margin_utilization
            )
            
            # Store metrics
            with self._lock:
                self.risk_metrics.append(metrics)
                self.stats['risk_checks'] += 1
                
                # Keep only recent metrics
                if len(self.risk_metrics) > 1000:
                    self.risk_metrics = self.risk_metrics[-1000:]
            
            # Check for risk alerts
            self._check_risk_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error checking risk: {e}", exc_info=True)
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_current_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        metrics = self.check_risk()
        
        if (metrics.total_risk > self.risk_thresholds['max_portfolio_risk'] or
            metrics.max_drawdown > self.risk_thresholds['max_drawdown'] or
            metrics.margin_utilization > self.risk_thresholds['max_margin_utilization']):
            return RiskLevel.CRITICAL
        elif (metrics.total_risk > self.risk_thresholds['max_portfolio_risk'] * 0.7 or
              metrics.max_drawdown > self.risk_thresholds['max_drawdown'] * 0.7):
            return RiskLevel.HIGH
        elif (metrics.total_risk > self.risk_thresholds['max_portfolio_risk'] * 0.5 or
              metrics.max_drawdown > self.risk_thresholds['max_drawdown'] * 0.5):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_risk_alerts(self, level: Optional[RiskLevel] = None, 
                       acknowledged: Optional[bool] = None) -> List[RiskAlert]:
        """Get risk alerts."""
        with self._lock:
            alerts = self.risk_alerts.copy()
        
        if level is not None:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if acknowledged is not None:
            alerts = [alert for alert in alerts if alert.acknowledged == acknowledged]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert."""
        with self._lock:
            for alert in self.risk_alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Risk alert acknowledged: {alert_id}")
                    return True
        return False
    
    def get_recent_metrics(self, limit: int = 100) -> List[RiskMetrics]:
        """Get recent risk metrics."""
        with self._lock:
            return self.risk_metrics[-limit:] if self.risk_metrics else []
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self.check_risk()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in risk monitor loop: {e}", exc_info=True)
                time.sleep(120)  # Wait longer on error
    
    def _calculate_total_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate total portfolio risk."""
        if not positions:
            return 0.0
        
        try:
            # Use risk manager to calculate total risk
            return self.risk_manager.run()
        except Exception as e:
            self.logger.error(f"Error calculating total risk: {e}")
            return 0.0
    
    def _calculate_portfolio_var(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio Value at Risk."""
        if not positions:
            return 0.0
        
        try:
            # Calculate VaR using risk manager
            return self.risk_manager.run()
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, positions: Dict[str, Any]) -> float:
        """Calculate maximum drawdown."""
        if not positions:
            return 0.0
        
        try:
            # Calculate drawdown from position history
            total_pnl = sum(pos.pnl for pos in positions.values())
            total_value = sum(pos.entry_price * pos.volume for pos in positions.values())
            
            if total_value > 0:
                return abs(total_pnl / total_value) * 100
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate correlation risk."""
        if len(positions) < 2:
            return 0.0
        
        try:
            # Calculate average correlation between positions
            symbols = list(positions.keys())
            correlations = []
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    # This is a simplified correlation calculation
                    # In practice, you'd use historical price data
                    correlations.append(0.5)  # Placeholder
            
            return sum(correlations) / len(correlations) if correlations else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate concentration risk."""
        if not positions:
            return 0.0
        
        try:
            # Calculate largest position as percentage of total
            total_value = sum(pos.entry_price * pos.volume for pos in positions.values())
            if total_value == 0:
                return 0.0
            
            max_position_value = max(pos.entry_price * pos.volume for pos in positions.values())
            return (max_position_value / total_value) * 100
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_leverage_ratio(self, positions: Dict[str, Any]) -> float:
        """Calculate leverage ratio."""
        try:
            # Get account balance
            account_info = self.trader.broker.get_account_info()
            balance = account_info.get('balance', 10000)  # Default fallback
            
            # Calculate total position value
            total_position_value = sum(pos.entry_price * pos.volume for pos in positions.values())
            
            if balance > 0:
                return total_position_value / balance
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating leverage ratio: {e}")
            return 0.0
    
    def _calculate_margin_utilization(self, positions: Dict[str, Any]) -> float:
        """Calculate margin utilization."""
        try:
            # Get account info
            account_info = self.trader.broker.get_account_info()
            balance = account_info.get('balance', 10000)
            equity = account_info.get('equity', balance)
            
            # Calculate margin used
            margin_used = sum(pos.entry_price * pos.volume * 0.01 for pos in positions.values())  # Simplified
            
            if equity > 0:
                return (margin_used / equity) * 100
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating margin utilization: {e}")
            return 0.0
    
    def _check_risk_alerts(self, metrics: RiskMetrics):
        """Check for risk alerts and create them if needed."""
        alerts = []
        
        # Check portfolio risk
        if metrics.total_risk > self.risk_thresholds['max_portfolio_risk']:
            alerts.append(RiskAlert(
                id=f"risk_{int(time.time())}",
                level=RiskLevel.CRITICAL,
                message=f"Portfolio risk ({metrics.total_risk:.2f}%) exceeds threshold ({self.risk_thresholds['max_portfolio_risk']}%)",
                component="Portfolio Risk",
                value=metrics.total_risk,
                threshold=self.risk_thresholds['max_portfolio_risk']
            ))
        
        # Check drawdown
        if metrics.max_drawdown > self.risk_thresholds['max_drawdown']:
            alerts.append(RiskAlert(
                id=f"drawdown_{int(time.time())}",
                level=RiskLevel.HIGH,
                message=f"Maximum drawdown ({metrics.max_drawdown:.2f}%) exceeds threshold ({self.risk_thresholds['max_drawdown']}%)",
                component="Drawdown",
                value=metrics.max_drawdown,
                threshold=self.risk_thresholds['max_drawdown']
            ))
        
        # Check correlation risk
        if metrics.correlation_risk > self.risk_thresholds['max_correlation']:
            alerts.append(RiskAlert(
                id=f"correlation_{int(time.time())}",
                level=RiskLevel.MEDIUM,
                message=f"Correlation risk ({metrics.correlation_risk:.2f}) exceeds threshold ({self.risk_thresholds['max_correlation']})",
                component="Correlation Risk",
                value=metrics.correlation_risk,
                threshold=self.risk_thresholds['max_correlation']
            ))
        
        # Check concentration risk
        if metrics.concentration_risk > self.risk_thresholds['max_concentration']:
            alerts.append(RiskAlert(
                id=f"concentration_{int(time.time())}",
                level=RiskLevel.MEDIUM,
                message=f"Concentration risk ({metrics.concentration_risk:.2f}%) exceeds threshold ({self.risk_thresholds['max_concentration']}%)",
                component="Concentration Risk",
                value=metrics.concentration_risk,
                threshold=self.risk_thresholds['max_concentration']
            ))
        
        # Check leverage ratio
        if metrics.leverage_ratio > self.risk_thresholds['max_leverage']:
            alerts.append(RiskAlert(
                id=f"leverage_{int(time.time())}",
                level=RiskLevel.HIGH,
                message=f"Leverage ratio ({metrics.leverage_ratio:.2f}) exceeds threshold ({self.risk_thresholds['max_leverage']})",
                component="Leverage",
                value=metrics.leverage_ratio,
                threshold=self.risk_thresholds['max_leverage']
            ))
        
        # Check margin utilization
        if metrics.margin_utilization > self.risk_thresholds['max_margin_utilization']:
            alerts.append(RiskAlert(
                id=f"margin_{int(time.time())}",
                level=RiskLevel.CRITICAL,
                message=f"Margin utilization ({metrics.margin_utilization:.2f}%) exceeds threshold ({self.risk_thresholds['max_margin_utilization']}%)",
                component="Margin Utilization",
                value=metrics.margin_utilization,
                threshold=self.risk_thresholds['max_margin_utilization']
            ))
        
        # Add alerts and send notifications
        for alert in alerts:
            self._add_risk_alert(alert)
            self._send_risk_notification(alert)
    
    def _add_risk_alert(self, alert: RiskAlert):
        """Add a risk alert."""
        with self._lock:
            self.risk_alerts.append(alert)
            self.stats['total_alerts'] += 1
            
            if alert.level == RiskLevel.HIGH:
                self.stats['high_risk_alerts'] += 1
            elif alert.level == RiskLevel.CRITICAL:
                self.stats['critical_alerts'] += 1
        
        self.logger.warning(f"Risk alert: {alert.message}")
    
    def _send_risk_notification(self, alert: RiskAlert):
        """Send risk notification."""
        if not self.notification_manager:
            return
        
        try:
            self.notification_manager.send_system_alert(
                level=alert.level.value.upper(),
                message=alert.message,
                details=f"Value: {alert.value:.2f}, Threshold: {alert.threshold:.2f}",
                component=alert.component,
                status="Active"
            )
        except Exception as e:
            self.logger.error(f"Failed to send risk notification: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk monitor statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'current_risk_level': self.get_current_risk_level().value,
                'active_alerts': len([a for a in self.risk_alerts if not a.acknowledged]),
                'total_alerts': len(self.risk_alerts),
                'recent_metrics': len(self.risk_metrics)
            })
            return stats 