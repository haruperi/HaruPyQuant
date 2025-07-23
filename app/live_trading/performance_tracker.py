"""
Performance Tracker

This module tracks trading performance metrics and generates reports.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.trading.trader import Trader
from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class PerformancePeriod(Enum):
    """Performance periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    period: PerformancePeriod
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    total_volume: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRecord:
    """Trade record."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    pnl_percent: float
    entry_time: datetime
    exit_time: datetime
    duration: timedelta
    strategy: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PerformanceTracker:
    """Tracks trading performance and generates reports."""
    
    def __init__(self, trader: Trader, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize performance tracker.
        
        Args:
            trader: Trader instance
            notification_manager: Notification manager for alerts
        """
        self.trader = trader
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Performance data
        self.trades: List[TradeRecord] = []
        self.performance_metrics: Dict[PerformancePeriod, List[PerformanceMetrics]] = {
            period: [] for period in PerformancePeriod
        }
        
        # Threading
        self._lock = threading.Lock()
        self._update_thread = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'current_streak': 0
        }
        
        self.logger.info("Performance tracker initialized")
    
    def start(self):
        """Start the performance tracker."""
        if self._running:
            self.logger.warning("Performance tracker already running")
            return
        
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="PerformanceTracker"
        )
        self._update_thread.start()
        self.logger.info("Performance tracker started")
    
    def stop(self):
        """Stop the performance tracker."""
        if not self._running:
            return
        
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=10)
        
        self.logger.info("Performance tracker stopped")
    
    def add_trade(self, trade: TradeRecord):
        """Add a completed trade."""
        with self._lock:
            self.trades.append(trade)
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += trade.pnl
            
            # Update best/worst trade
            if trade.pnl > self.stats['best_trade']:
                self.stats['best_trade'] = trade.pnl
            if trade.pnl < self.stats['worst_trade']:
                self.stats['worst_trade'] = trade.pnl
        
        self.logger.info(f"Trade recorded: {trade.symbol} {trade.direction} PnL: {trade.pnl:.2f}")
    
    def get_performance_metrics(self, period: PerformancePeriod, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Get performance metrics for a period."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter trades for the period
        period_trades = [
            trade for trade in self.trades
            if start_date <= trade.exit_time <= end_date
        ]
        
        if not period_trades:
            return PerformanceMetrics(
                period=period,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                total_volume=0.0
            )
        
        # Calculate metrics
        total_trades = len(period_trades)
        winning_trades = [t for t in period_trades if t.pnl > 0]
        losing_trades = [t for t in period_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t.pnl for t in period_trades)
        
        # Calculate average win/loss
        average_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        average_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Calculate max drawdown
        max_drawdown, max_drawdown_percent = self._calculate_max_drawdown(period_trades)
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio(period_trades)
        
        # Calculate total volume
        total_volume = sum(t.volume for t in period_trades)
        
        # Calculate total PnL percentage
        total_volume_value = sum(t.entry_price * t.volume for t in period_trades)
        total_pnl_percent = (total_pnl / total_volume_value) * 100 if total_volume_value > 0 else 0.0
        
        return PerformanceMetrics(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            total_volume=total_volume
        )
    
    def get_recent_trades(self, limit: int = 100) -> List[TradeRecord]:
        """Get recent trades."""
        with self._lock:
            return sorted(self.trades, key=lambda x: x.exit_time, reverse=True)[:limit]
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get trades for a specific symbol."""
        with self._lock:
            return [trade for trade in self.trades if trade.symbol == symbol]
    
    def get_trades_by_strategy(self, strategy: str) -> List[TradeRecord]:
        """Get trades for a specific strategy."""
        with self._lock:
            return [trade for trade in self.trades if trade.strategy == strategy]
    
    def get_equity_curve(self, period: PerformancePeriod = PerformancePeriod.DAILY) -> List[Dict[str, Any]]:
        """Get equity curve data."""
        if not self.trades:
            return []
        
        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda x: x.exit_time)
        
        equity_curve = []
        cumulative_pnl = 0.0
        
        for trade in sorted_trades:
            cumulative_pnl += trade.pnl
            equity_curve.append({
                'date': trade.exit_time,
                'pnl': trade.pnl,
                'cumulative_pnl': cumulative_pnl,
                'trade_id': trade.trade_id,
                'symbol': trade.symbol
            })
        
        return equity_curve
    
    def generate_performance_report(self, period: PerformancePeriod = PerformancePeriod.MONTHLY) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        metrics = self.get_performance_metrics(period)
        
        report = {
            'period': period.value,
            'start_date': metrics.start_date.isoformat(),
            'end_date': metrics.end_date.isoformat(),
            'summary': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': f"{metrics.win_rate:.2%}",
                'total_pnl': f"${metrics.total_pnl:.2f}",
                'total_pnl_percent': f"{metrics.total_pnl_percent:.2f}%"
            },
            'risk_metrics': {
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"${metrics.max_drawdown:.2f}",
                'max_drawdown_percent': f"{metrics.max_drawdown_percent:.2f}%"
            },
            'trade_analysis': {
                'average_win': f"${metrics.average_win:.2f}",
                'average_loss': f"${metrics.average_loss:.2f}",
                'total_volume': f"{metrics.total_volume:.2f}"
            }
        }
        
        return report
    
    def _update_loop(self):
        """Main update loop."""
        while self._running:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in performance tracker loop: {e}", exc_info=True)
                time.sleep(600)  # Wait longer on error
    
    def _update_performance_metrics(self):
        """Update performance metrics for all periods."""
        try:
            # Calculate metrics for different periods
            now = datetime.now()
            
            # Daily metrics
            daily_start = now - timedelta(days=1)
            daily_metrics = self.get_performance_metrics(PerformancePeriod.DAILY, daily_start, now)
            
            # Weekly metrics
            weekly_start = now - timedelta(weeks=1)
            weekly_metrics = self.get_performance_metrics(PerformancePeriod.WEEKLY, weekly_start, now)
            
            # Monthly metrics
            monthly_start = now - timedelta(days=30)
            monthly_metrics = self.get_performance_metrics(PerformancePeriod.MONTHLY, monthly_start, now)
            
            # Store metrics
            with self._lock:
                self.performance_metrics[PerformancePeriod.DAILY].append(daily_metrics)
                self.performance_metrics[PerformancePeriod.WEEKLY].append(weekly_metrics)
                self.performance_metrics[PerformancePeriod.MONTHLY].append(monthly_metrics)
                
                # Keep only recent metrics
                for period in PerformancePeriod:
                    if len(self.performance_metrics[period]) > 100:
                        self.performance_metrics[period] = self.performance_metrics[period][-100:]
            
            # Send performance notifications if significant
            self._check_performance_alerts(monthly_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> tuple[float, float]:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0, 0.0
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x.exit_time)
        
        peak = 0.0
        max_drawdown = 0.0
        cumulative_pnl = 0.0
        
        for trade in sorted_trades:
            cumulative_pnl += trade.pnl
            
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate percentage
        max_drawdown_percent = (max_drawdown / peak) * 100 if peak > 0 else 0.0
        
        return max_drawdown, max_drawdown_percent
    
    def _calculate_sharpe_ratio(self, trades: List[TradeRecord]) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if not trades:
            return 0.0
        
        # Calculate returns
        returns = [trade.pnl_percent for trade in trades]
        
        if not returns:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = sum(returns) / len(returns)
        
        if len(returns) < 2:
            return 0.0
        
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0.0
        
        return sharpe_ratio
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        if not self.notification_manager:
            return
        
        try:
            # Check for significant losses
            if metrics.total_pnl < -1000:  # $1000 loss threshold
                self.notification_manager.send_system_alert(
                    level="WARNING",
                    message=f"Significant trading loss detected: ${metrics.total_pnl:.2f}",
                    details=f"Period: {metrics.period.value}, Win Rate: {metrics.win_rate:.2%}",
                    component="Performance Tracker",
                    status="Active"
                )
            
            # Check for poor win rate
            if metrics.win_rate < 0.3 and metrics.total_trades > 10:
                self.notification_manager.send_system_alert(
                    level="WARNING",
                    message=f"Low win rate detected: {metrics.win_rate:.2%}",
                    details=f"Total trades: {metrics.total_trades}, PnL: ${metrics.total_pnl:.2f}",
                    component="Performance Tracker",
                    status="Active"
                )
            
            # Check for high drawdown
            if metrics.max_drawdown_percent > 10:
                self.notification_manager.send_system_alert(
                    level="WARNING",
                    message=f"High drawdown detected: {metrics.max_drawdown_percent:.2f}%",
                    details=f"Max drawdown: ${metrics.max_drawdown:.2f}",
                    component="Performance Tracker",
                    status="Active"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send performance alert: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance tracker statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'total_trades': len(self.trades),
                'recent_trades': len(self.trades[-100:]) if self.trades else 0,
                'performance_metrics': {
                    period.value: len(metrics) for period, metrics in self.performance_metrics.items()
                }
            })
            return stats 