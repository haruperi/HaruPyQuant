"""
Strategy Runner

This module manages strategy execution, signal generation, and strategy lifecycle.
"""

import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

from app.strategy.base import BaseStrategy
from app.strategy.swing_trend_momentum import SwingTrendMomentumStrategy
from app.data.mt5_client import MT5Client
from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class StrategyStatus(Enum):
    """Strategy status."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StrategySignal:
    """Strategy signal."""
    id: str
    strategy_name: str
    symbol: str
    signal: int  # 1 for buy, -1 for sell, 0 for no signal
    strength: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class StrategyResult:
    """Strategy execution result."""
    strategy_name: str
    symbol: str
    signals_generated: int = 0
    execution_time: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    status: StrategyStatus = StrategyStatus.STOPPED


class StrategyRunner:
    """Manages strategy execution and signal generation."""
    
    def __init__(self, mt5_client: MT5Client, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize strategy runner.
        
        Args:
            mt5_client: MT5 client for data access
            notification_manager: Notification manager for alerts
        """
        self.mt5_client = mt5_client
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        self.strategy_results: Dict[str, StrategyResult] = {}
        
        # Signal management
        self.signals: List[StrategySignal] = []
        self.signal_callbacks: List[Callable[[StrategySignal], None]] = []
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._strategy_threads: Dict[str, threading.Thread] = {}
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'total_executions': 0,
            'total_errors': 0
        }
        
        self.logger.info("Strategy runner initialized")
    
    def add_strategy(self, strategy_name: str, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        Add a strategy to the runner.
        
        Args:
            strategy_name: Name of the strategy
            strategy: Strategy instance
            config: Strategy configuration
        """
        with self._lock:
            self.strategies[strategy_name] = strategy
            self.strategy_configs[strategy_name] = config
            self.strategy_results[strategy_name] = StrategyResult(
                strategy_name=strategy_name,
                symbol=config.get('symbol', 'Unknown'),
                status=StrategyStatus.STOPPED
            )
        
        self.logger.info(f"Strategy added: {strategy_name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from the runner.
        
        Args:
            strategy_name: Name of the strategy to remove
            
        Returns:
            True if removed successfully
        """
        with self._lock:
            if strategy_name in self.strategies:
                # Stop the strategy if running
                if strategy_name in self._strategy_threads:
                    self._stop_strategy_thread(strategy_name)
                
                del self.strategies[strategy_name]
                del self.strategy_configs[strategy_name]
                if strategy_name in self.strategy_results:
                    del self.strategy_results[strategy_name]
                
                self.logger.info(f"Strategy removed: {strategy_name}")
                return True
        
        return False
    
    def start_strategy(self, strategy_name: str) -> bool:
        """
        Start a strategy.
        
        Args:
            strategy_name: Name of the strategy to start
            
        Returns:
            True if started successfully
        """
        with self._lock:
            if strategy_name not in self.strategies:
                self.logger.error(f"Strategy not found: {strategy_name}")
                return False
            
            if strategy_name in self._strategy_threads:
                self.logger.warning(f"Strategy already running: {strategy_name}")
                return False
            
            # Start strategy thread
            thread = threading.Thread(
                target=self._strategy_loop,
                args=(strategy_name,),
                daemon=True,
                name=f"Strategy-{strategy_name}"
            )
            thread.start()
            self._strategy_threads[strategy_name] = thread
            
            # Update status
            self.strategy_results[strategy_name].status = StrategyStatus.RUNNING
        
        self.logger.info(f"Strategy started: {strategy_name}")
        return True
    
    def stop_strategy(self, strategy_name: str) -> bool:
        """
        Stop a strategy.
        
        Args:
            strategy_name: Name of the strategy to stop
            
        Returns:
            True if stopped successfully
        """
        with self._lock:
            if strategy_name not in self._strategy_threads:
                return False
            
            self._stop_strategy_thread(strategy_name)
            self.strategy_results[strategy_name].status = StrategyStatus.STOPPED
        
        self.logger.info(f"Strategy stopped: {strategy_name}")
        return True
    
    def start_all_strategies(self):
        """Start all strategies."""
        with self._lock:
            strategy_names = list(self.strategies.keys())
        
        for strategy_name in strategy_names:
            self.start_strategy(strategy_name)
    
    def stop_all_strategies(self):
        """Stop all strategies."""
        with self._lock:
            strategy_names = list(self._strategy_threads.keys())
        
        for strategy_name in strategy_names:
            self.stop_strategy(strategy_name)
    
    def pause_strategy(self, strategy_name: str) -> bool:
        """
        Pause a strategy.
        
        Args:
            strategy_name: Name of the strategy to pause
            
        Returns:
            True if paused successfully
        """
        with self._lock:
            if strategy_name not in self.strategy_results:
                return False
            
            self.strategy_results[strategy_name].status = StrategyStatus.PAUSED
        
        self.logger.info(f"Strategy paused: {strategy_name}")
        return True
    
    def resume_strategy(self, strategy_name: str) -> bool:
        """
        Resume a paused strategy.
        
        Args:
            strategy_name: Name of the strategy to resume
            
        Returns:
            True if resumed successfully
        """
        with self._lock:
            if strategy_name not in self.strategy_results:
                return False
            
            if self.strategy_results[strategy_name].status == StrategyStatus.PAUSED:
                self.strategy_results[strategy_name].status = StrategyStatus.RUNNING
        
        self.logger.info(f"Strategy resumed: {strategy_name}")
        return True
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]):
        """Add a callback for signal notifications."""
        self.signal_callbacks.append(callback)
    
    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """Get status of a strategy."""
        with self._lock:
            if strategy_name in self.strategy_results:
                return self.strategy_results[strategy_name].status
        return None
    
    def get_all_strategy_status(self) -> Dict[str, StrategyStatus]:
        """Get status of all strategies."""
        with self._lock:
            return {
                name: result.status 
                for name, result in self.strategy_results.items()
            }
    
    def get_recent_signals(self, limit: int = 100) -> List[StrategySignal]:
        """Get recent signals."""
        with self._lock:
            return sorted(self.signals, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_strategy_signals(self, strategy_name: str, limit: int = 50) -> List[StrategySignal]:
        """Get recent signals for a specific strategy."""
        with self._lock:
            strategy_signals = [
                signal for signal in self.signals 
                if signal.strategy_name == strategy_name
            ]
            return sorted(strategy_signals, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def _strategy_loop(self, strategy_name: str):
        """Main loop for strategy execution."""
        self.logger.info(f"Starting strategy loop: {strategy_name}")
        
        while self._is_strategy_running(strategy_name):
            try:
                start_time = time.time()
                
                # Execute strategy
                signals = self._execute_strategy(strategy_name)
                
                # Process signals
                for signal in signals:
                    self._process_signal(signal)
                
                # Update statistics
                execution_time = time.time() - start_time
                self._update_strategy_stats(strategy_name, len(signals), execution_time, 0)
                
                # Wait for next execution
                config = self.strategy_configs.get(strategy_name, {})
                interval = config.get('update_interval', 60.0)
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in strategy loop {strategy_name}: {e}", exc_info=True)
                self._update_strategy_stats(strategy_name, 0, 0.0, 1)
                self._handle_strategy_error(strategy_name, str(e))
                time.sleep(10)  # Wait before retry
        
        self.logger.info(f"Strategy loop ended: {strategy_name}")
    
    def _is_strategy_running(self, strategy_name: str) -> bool:
        """Check if strategy is running."""
        with self._lock:
            if strategy_name not in self.strategy_results:
                return False
            return self.strategy_results[strategy_name].status == StrategyStatus.RUNNING
    
    def _execute_strategy(self, strategy_name: str) -> List[StrategySignal]:
        """Execute a strategy and return signals."""
        strategy = self.strategies.get(strategy_name)
        config = self.strategy_configs.get(strategy_name, {})
        
        if not strategy or not config:
            return []
        
        signals = []
        symbols = config.get('symbols', [])
        
        for symbol in symbols:
            try:
                # Get market data
                data = self._get_strategy_data(strategy_name, symbol)
                if data is None:
                    continue
                
                # Execute strategy
                if isinstance(strategy, SwingTrendMomentumStrategy):
                    signal = self._execute_swing_trend_strategy(strategy, symbol, data, config)
                else:
                    signal = self._execute_generic_strategy(strategy, symbol, data, config)
                
                if signal:
                    signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error executing strategy {strategy_name} for {symbol}: {e}")
        
        return signals
    
    def _get_strategy_data(self, strategy_name: str, symbol: str) -> Optional[pd.DataFrame]:
        """Get data for strategy execution."""
        try:
            config = self.strategy_configs.get(strategy_name, {})
            timeframe = config.get('timeframe', 'M5')
            bars = config.get('bars', 300)
            
            data = self.mt5_client.fetch_data(symbol, timeframe, start_pos=0, end_pos=bars)
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {strategy_name} {symbol}: {e}")
            return None
    
    def _execute_swing_trend_strategy(self, strategy: SwingTrendMomentumStrategy, 
                                    symbol: str, data: pd.DataFrame, 
                                    config: Dict[str, Any]) -> Optional[StrategySignal]:
        """Execute Swing Trend Momentum strategy."""
        try:
            # Get H1 data for higher timeframe analysis
            h1_data = self.mt5_client.fetch_data(symbol, "H1", start_pos=0, end_pos=150)
            if h1_data is None:
                return None
            
            # Check for signal trigger
            trigger, data_with_trigger = strategy.signal_trigger(data, h1_data)
            if not trigger:
                return None
            
            # Get base and quote currency data
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            df_base = self.mt5_client.fetch_data(f"{base_currency}X", "H1", start_pos=0, end_pos=150)
            df_quote = self.mt5_client.fetch_data(f"{quote_currency}X", "H1", start_pos=0, end_pos=150)
            
            if df_base is None or df_quote is None:
                return None
            
            # Get signals
            signal_trigger, data_with_signals = strategy.get_signals(data_with_trigger, df_base, df_quote)
            if not signal_trigger:
                return None
            
            # Get the signal
            action = data_with_signals["Signal"].iloc[-2]
            if action == 0:
                return None
            
            # Create signal
            signal = StrategySignal(
                id=str(uuid.uuid4()),
                strategy_name="SwingTrendMomentum",
                symbol=symbol,
                signal=action,
                strength=abs(data_with_signals["Trend_Strength"].iloc[-2]),
                confidence=0.8 if abs(action) == 1 else 0.6,
                metadata={
                    'trend_strength': data_with_signals["Trend_Strength"].iloc[-2],
                    'trigger': data_with_trigger["Trigger"].iloc[-2]
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error executing Swing Trend strategy for {symbol}: {e}")
            return None
    
    def _execute_generic_strategy(self, strategy: BaseStrategy, symbol: str, 
                                data: pd.DataFrame, config: Dict[str, Any]) -> Optional[StrategySignal]:
        """Execute generic strategy."""
        try:
            # Get signals from strategy
            signals = strategy.get_signals(data)
            if signals is None or len(signals) == 0:
                return None
            
            # Get latest signal
            latest_signal = signals.iloc[-1]
            if latest_signal == 0:
                return None
            
            # Create signal
            signal = StrategySignal(
                id=str(uuid.uuid4()),
                strategy_name=strategy.__class__.__name__,
                symbol=symbol,
                signal=int(latest_signal),
                strength=1.0,
                confidence=0.7
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error executing generic strategy for {symbol}: {e}")
            return None
    
    def _process_signal(self, signal: StrategySignal):
        """Process a strategy signal."""
        with self._lock:
            self.signals.append(signal)
            self.stats['total_signals'] += 1
            
            if signal.signal > 0:
                self.stats['buy_signals'] += 1
            elif signal.signal < 0:
                self.stats['sell_signals'] += 1
        
        # Notify callbacks
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"Error in signal callback: {e}")
        
        # Send notification
        self._send_signal_notification(signal)
        
        self.logger.info(f"Signal generated: {signal.strategy_name} {signal.symbol} {signal.signal}")
    
    def _send_signal_notification(self, signal: StrategySignal):
        """Send signal notification."""
        if not self.notification_manager:
            return
        
        try:
            action = "BUY" if signal.signal > 0 else "SELL"
            self.notification_manager.send_trading_alert(
                symbol=signal.symbol,
                action=action,
                price=0.0,  # Will be filled by execution engine
                reason=f"Strategy: {signal.strategy_name}",
                account="Demo Account",
                strategy=signal.strategy_name,
                risk_level="Medium"
            )
        except Exception as e:
            self.logger.error(f"Failed to send signal notification: {e}")
    
    def _update_strategy_stats(self, strategy_name: str, signals: int, 
                             execution_time: float, errors: int):
        """Update strategy statistics."""
        with self._lock:
            if strategy_name in self.strategy_results:
                result = self.strategy_results[strategy_name]
                result.signals_generated += signals
                result.execution_time = execution_time
                result.error_count += errors
                result.last_execution = datetime.now()
            
            self.stats['total_executions'] += 1
            self.stats['total_errors'] += errors
    
    def _handle_strategy_error(self, strategy_name: str, error_message: str):
        """Handle strategy error."""
        with self._lock:
            if strategy_name in self.strategy_results:
                self.strategy_results[strategy_name].status = StrategyStatus.ERROR
        
        # Send notification
        if self.notification_manager:
            try:
                self.notification_manager.send_error_alert(
                    error_type="Strategy Error",
                    message=f"Strategy {strategy_name} encountered an error",
                    component="Strategy Runner",
                    stack_trace=error_message
                )
            except Exception as e:
                self.logger.error(f"Failed to send error notification: {e}")
    
    def _stop_strategy_thread(self, strategy_name: str):
        """Stop a strategy thread."""
        if strategy_name in self._strategy_threads:
            thread = self._strategy_threads[strategy_name]
            del self._strategy_threads[strategy_name]
            # Thread will stop naturally when _is_strategy_running returns False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy runner statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'active_strategies': len([s for s in self.strategy_results.values() 
                                        if s.status == StrategyStatus.RUNNING]),
                'total_strategies': len(self.strategies),
                'recent_signals': len(self.signals)
            })
            return stats
    
    def get_strategy_results(self) -> Dict[str, StrategyResult]:
        """Get all strategy results."""
        with self._lock:
            return self.strategy_results.copy() 