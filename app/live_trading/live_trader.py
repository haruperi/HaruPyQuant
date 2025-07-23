"""
Live Trader

This is the main orchestrator for the live trading system that coordinates all components.
"""

import time
import threading
import signal
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.data.mt5_client import MT5Client
from app.trading.trader import Trader
from app.trading.broker import Broker
from app.trading.risk_manager import RiskManager
from app.notifications.manager import NotificationManager
from app.strategy.swing_trend_momentum import SwingTrendMomentumStrategy

from .config import LiveTradingConfig, StrategyConfig
from .execution_engine import ExecutionEngine, ExecutionRequest
from .strategy_runner import StrategyRunner, StrategySignal
from .position_manager import PositionManager
from .risk_monitor import RiskMonitor
from .system_monitor import SystemMonitor
from .performance_tracker import PerformanceTracker
from .trading_scheduler import TradingScheduler, MarketStatus
from app.notifications.manager import NotificationManagerConfig

from app.util import get_logger

logger = get_logger(__name__)


class LiveTraderStatus(Enum):
    """Live trader status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class LiveTraderStats:
    """Live trader statistics."""
    status: LiveTraderStatus
    uptime: timedelta
    total_signals: int
    total_trades: int
    total_pnl: float
    active_positions: int
    system_health: str
    market_status: str
    timestamp: datetime = field(default_factory=datetime.now)


class LiveTrader:
    """Main orchestrator for live trading system."""
    
    def __init__(self, config: LiveTradingConfig):
        """
        Initialize live trader.
        
        Args:
            config: Live trading configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Status tracking
        self.status = LiveTraderStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.error_count = 0
        
        # Core components
        self.mt5_client: Optional[MT5Client] = None
        self.trader: Optional[Trader] = None
        self.risk_manager: Optional[RiskManager] = None
        self.notification_manager: Optional[NotificationManager] = None
        
        # Live trading components
        self.execution_engine: Optional[ExecutionEngine] = None
        self.strategy_runner: Optional[StrategyRunner] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_monitor: Optional[RiskMonitor] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.trading_scheduler: Optional[TradingScheduler] = None
        
        # Threading
        self._lock = threading.Lock()
        self._main_thread = None
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.stats = LiveTraderStats(
            status=LiveTraderStatus.STOPPED,
            uptime=timedelta(0),
            total_signals=0,
            total_trades=0,
            total_pnl=0.0,
            active_positions=0,
            system_health="unknown",
            market_status="unknown"
        )
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Live trader initialized")
    
    def start(self) -> bool:
        """Start the live trading system."""
        if self.status != LiveTraderStatus.STOPPED:
            self.logger.warning(f"Cannot start live trader. Current status: {self.status}")
            return False
        
        try:
            self.status = LiveTraderStatus.STARTING
            self.logger.info("Starting live trading system...")
            
            # Initialize components
            if not self._initialize_components():
                self.status = LiveTraderStatus.ERROR
                return False
            
            # Start all components
            if not self._start_components():
                self.status = LiveTraderStatus.ERROR
                return False
            
            # Start main trading loop
            self._running = True
            self._shutdown_event.clear()
            self.start_time = datetime.now()
            self.status = LiveTraderStatus.RUNNING
            
            self._main_thread = threading.Thread(
                target=self._main_loop,
                daemon=False,
                name="LiveTrader"
            )
            self._main_thread.start()
            
            self.logger.info("Live trading system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start live trading system: {e}", exc_info=True)
            self.status = LiveTraderStatus.ERROR
            return False
    
    def stop(self):
        """Stop the live trading system."""
        if self.status == LiveTraderStatus.STOPPED:
            return
        
        self.logger.info("Stopping live trading system...")
        self.status = LiveTraderStatus.STOPPING
        self._running = False
        self._shutdown_event.set()
        
        # Stop all components
        self._stop_components()
        
        # Wait for main thread
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=30)
        
        self.status = LiveTraderStatus.STOPPED
        self.logger.info("Live trading system stopped")
    
    def pause(self):
        """Pause the live trading system."""
        if self.status == LiveTraderStatus.RUNNING:
            self.status = LiveTraderStatus.PAUSED
            self.logger.info("Live trading system paused")
    
    def resume(self):
        """Resume the live trading system."""
        if self.status == LiveTraderStatus.PAUSED:
            self.status = LiveTraderStatus.RUNNING
            self.logger.info("Live trading system resumed")
    
    def get_status(self) -> LiveTraderStatus:
        """Get current status."""
        return self.status
    
    def get_statistics(self) -> LiveTraderStats:
        """Get live trader statistics."""
        with self._lock:
            # Create a new stats instance instead of copying
            stats = LiveTraderStats(
                status=self.status,
                uptime=datetime.now() - self.start_time if self.start_time else timedelta(0),
                total_signals=self.stats.total_signals,
                total_trades=self.stats.total_trades,
                total_pnl=self.stats.total_pnl,
                active_positions=self.stats.active_positions,
                system_health=self.stats.system_health,
                market_status=self.stats.market_status,
                timestamp=datetime.now()
            )
            
            # Update component statistics if available
            try:
                if self.strategy_runner:
                    strategy_stats = self.strategy_runner.get_statistics()
                    stats.total_signals = strategy_stats.get('total_signals', 0)
                
                if self.position_manager:
                    position_stats = self.position_manager.get_statistics()
                    stats.total_trades = position_stats.get('total_positions', 0)
                    stats.total_pnl = position_stats.get('total_pnl', 0.0)
                    stats.active_positions = position_stats.get('position_count', 0)
                
                if self.system_monitor:
                    system_stats = self.system_monitor.get_statistics()
                    stats.system_health = system_stats.get('current_status', 'unknown')
                
                if self.trading_scheduler:
                    scheduler_stats = self.trading_scheduler.get_statistics()
                    stats.market_status = scheduler_stats.get('current_status', 'unknown')
            except Exception as e:
                self.logger.warning(f"Error updating component statistics: {e}")
            
            return stats
    
    def _initialize_components(self) -> bool:
        """Initialize all trading components."""
        try:
            # Initialize MT5 client
            self.logger.info("Initializing MT5 client...")
            from app.config.setup import DEFAULT_CONFIG_PATH, ALL_SYMBOLS, BROKER
            self.mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
            if not self.mt5_client.is_connected():
                self.logger.error("Failed to connect to MT5")
                return False
            
            # Initialize broker and trader
            self.logger.info("Initializing broker and trader...")
            from app.trading.mt5_broker import MT5Broker
            broker = MT5Broker(self.mt5_client)
            
            account_balance = broker.get_account_balance()
            self.risk_manager = RiskManager(self.mt5_client, account_balance, self.config.risk.max_risk_per_trade)
            self.trader = Trader(broker, self.risk_manager)
            
            # Initialize notification manager
            if self.config.notifications.enabled:
                self.logger.info("Initializing notification manager...")
                notification_config = self._load_notification_config()
                if notification_config:
                    self.notification_manager = NotificationManager(notification_config)
                else:
                    self.logger.warning("Notification configuration not found or disabled. Notifications will be disabled.")
            
            # Initialize live trading components
            self.logger.info("Initializing live trading components...")
            
            self.execution_engine = ExecutionEngine(self.trader, self.notification_manager)
            self.strategy_runner = StrategyRunner(self.mt5_client, self.notification_manager)
            self.position_manager = PositionManager(self.trader, self.notification_manager)
            self.risk_monitor = RiskMonitor(self.trader, self.risk_manager, self.notification_manager)
            self.system_monitor = SystemMonitor(self.notification_manager)
            self.performance_tracker = PerformanceTracker(self.trader, self.notification_manager)
            self.trading_scheduler = TradingScheduler(self.notification_manager)
            
            # Setup strategy
            self._setup_strategies()
            
            # Setup callbacks
            self._setup_callbacks()
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", exc_info=True)
            return False
    
    def _load_notification_config(self) -> Optional[NotificationManagerConfig]:
        """Load notification configuration from config.ini file."""
        try:
            from app.config.setup import DEFAULT_CONFIG_PATH
            from configparser import ConfigParser
            from app.notifications.email import EmailConfig
            from app.notifications.telegram import TelegramConfig
            from app.notifications.sms import SMSConfig
            
            config = ConfigParser()
            config.read(DEFAULT_CONFIG_PATH)
            
            # Check if notifications are enabled globally
            if config.has_section('NOTIFICATIONS'):
                enable_all = config.getboolean('NOTIFICATIONS', 'enable_all', fallback=True)
                if not enable_all:
                    return None
            
            notification_config = NotificationManagerConfig()
            
            # Load email configuration
            if config.has_section('EMAIL') and config.getboolean('EMAIL', 'enabled', fallback=False):
                email_config = EmailConfig(
                    smtp_server=config.get('EMAIL', 'smtp_server', fallback='smtp.gmail.com'),
                    smtp_port=config.getint('EMAIL', 'smtp_port', fallback=587),
                    username=config.get('EMAIL', 'username', fallback=''),
                    password=config.get('EMAIL', 'password', fallback=''),
                    use_tls=config.getboolean('EMAIL', 'use_tls', fallback=True),
                    use_ssl=config.getboolean('EMAIL', 'use_ssl', fallback=False),
                    from_email=config.get('EMAIL', 'from_email', fallback=''),
                    from_name=config.get('EMAIL', 'from_name', fallback='HaruPyQuant'),
                    recipients=config.get('EMAIL', 'recipients', fallback='').split(',')
                )
                notification_config.email_config = email_config
                self.logger.info("Email notification configuration loaded")
            
            # Load telegram configuration
            if config.has_section('TELEGRAM') and config.getboolean('TELEGRAM', 'enabled', fallback=False):
                telegram_config = TelegramConfig(
                    bot_token=config.get('TELEGRAM', 'token', fallback=''),
                    chat_ids=config.get('TELEGRAM', 'chat_ids', fallback='').split(','),
                    parse_mode=config.get('TELEGRAM', 'parse_mode', fallback='HTML'),
                    disable_web_page_preview=config.getboolean('TELEGRAM', 'disable_web_page_preview', fallback=True),
                    disable_notification=config.getboolean('TELEGRAM', 'disable_notification', fallback=False),
                    protect_content=config.getboolean('TELEGRAM', 'protect_content', fallback=False)
                )
                notification_config.telegram_config = telegram_config
                self.logger.info("Telegram notification configuration loaded")
            
            # Load SMS configuration
            if config.has_section('SMS') and config.getboolean('SMS', 'enabled', fallback=False):
                sms_config = SMSConfig(
                    account_sid=config.get('SMS', 'account_sid', fallback=''),
                    auth_token=config.get('SMS', 'auth_token', fallback=''),
                    from_number=config.get('SMS', 'from_number', fallback=''),
                    recipients=config.get('SMS', 'recipients', fallback='').split(','),
                    webhook_url=config.get('SMS', 'webhook_url', fallback=''),
                    status_callback=config.get('SMS', 'status_callback', fallback='')
                )
                notification_config.sms_config = sms_config
                self.logger.info("SMS notification configuration loaded")
            
            return notification_config
            
        except Exception as e:
            self.logger.error(f"Failed to load notification configuration: {e}")
            return None

    def _setup_strategies(self):
        """Setup trading strategies."""
        try:
            # Setup Swing Trend Momentum strategy
            swing_config = self.config.get_strategy_config("SwingTrendMomentum")
            if swing_config and swing_config.enabled:
                self.logger.info("Setting up Swing Trend Momentum strategy...")
                
                for symbol in swing_config.symbols:
                    symbol_info = self.mt5_client.get_symbol_info(symbol)
                    if symbol_info:
                        strategy = SwingTrendMomentumStrategy(self.mt5_client, symbol_info)
                        self.strategy_runner.add_strategy(
                            f"SwingTrendMomentum_{symbol}",
                            strategy,
                            {
                                'symbol': symbol,
                                'timeframe': 'M5',
                                'update_interval': 60.0,
                                'bars': 300
                            }
                        )
            
            self.logger.info("Strategies setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup strategies: {e}", exc_info=True)
    
    def _setup_callbacks(self):
        """Setup component callbacks."""
        try:
            # Strategy runner callbacks
            if self.strategy_runner:
                self.strategy_runner.add_signal_callback(self._on_strategy_signal)
            
            # Execution engine callbacks
            if self.execution_engine:
                self.execution_engine.on_execution_complete = self._on_execution_complete
                self.execution_engine.on_execution_failed = self._on_execution_failed
            
            # Trading scheduler callbacks
            if self.trading_scheduler:
                self.trading_scheduler.on_market_open = self._on_market_open
                self.trading_scheduler.on_market_close = self._on_market_close
            
            self.logger.info("Callbacks setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup callbacks: {e}", exc_info=True)
    
    def _start_components(self) -> bool:
        """Start all components."""
        try:
            self.logger.info("Starting components...")
            
            # Start monitors
            if self.system_monitor:
                self.system_monitor.start()
            
            if self.risk_monitor:
                self.risk_monitor.start()
            
            if self.performance_tracker:
                self.performance_tracker.start()
            
            if self.trading_scheduler:
                self.trading_scheduler.start()
            
            # Start execution engine
            if self.execution_engine:
                self.execution_engine.start()
            
            # Start position manager
            if self.position_manager:
                self.position_manager.start()
            
            # Start strategy runner
            if self.strategy_runner:
                self.strategy_runner.start_all_strategies()
            
            self.logger.info("All components started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start components: {e}", exc_info=True)
            return False
    
    def _stop_components(self):
        """Stop all components."""
        try:
            self.logger.info("Stopping components...")
            
            # Stop strategy runner
            if self.strategy_runner:
                self.strategy_runner.stop_all_strategies()
            
            # Stop position manager
            if self.position_manager:
                self.position_manager.stop()
            
            # Stop execution engine
            if self.execution_engine:
                self.execution_engine.stop()
            
            # Stop monitors
            if self.trading_scheduler:
                self.trading_scheduler.stop()
            
            if self.performance_tracker:
                self.performance_tracker.stop()
            
            if self.risk_monitor:
                self.risk_monitor.stop()
            
            if self.system_monitor:
                self.system_monitor.stop()
            
            # Disconnect MT5
            if self.mt5_client:
                self.mt5_client.disconnect()
            
            self.logger.info("All components stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}", exc_info=True)
    
    def _main_loop(self):
        """Main trading loop."""
        self.logger.info("Main trading loop started")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # Check if system should be paused
                if self.status == LiveTraderStatus.PAUSED:
                    time.sleep(1)
                    continue
                
                # Check market status
                if self.trading_scheduler and not self.trading_scheduler.is_market_open():
                    time.sleep(30)  # Wait longer when market is closed
                    continue
                
                # Update trader state
                if self.trader:
                    self.trader.update()
                
                # Update position manager
                if self.position_manager:
                    self.position_manager.update_positions()
                
                # Check risk
                if self.risk_monitor:
                    self.risk_monitor.check_risk()
                
                # Update statistics
                self._update_statistics()
                
                # Small delay to prevent excessive CPU usage, but check status frequently
                for _ in range(int(self.config.data_update_interval)):
                    if not self._running or self.status == LiveTraderStatus.STOPPING or self._shutdown_event.is_set():
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                
                # Send error notification
                if self.notification_manager:
                    try:
                        self.notification_manager.send_error_alert(
                            error_type="Main Loop Error",
                            message=f"Error in live trading main loop: {str(e)}",
                            component="Live Trader",
                            stack_trace=str(e)
                        )
                    except Exception as notify_error:
                        self.logger.error(f"Failed to send error notification: {notify_error}")
                
                # Wait longer on error
                time.sleep(60)
        
        self.logger.info("Main trading loop ended")
    
    def _on_strategy_signal(self, signal: StrategySignal):
        """Handle strategy signal."""
        try:
            self.logger.info(f"Strategy signal received: {signal.strategy_name} {signal.symbol} {signal.signal}")
            
            # Check if we should execute the signal
            if not self._should_execute_signal(signal):
                return
            
            # Create execution request
            request = ExecutionRequest(
                id=f"signal_{signal.id}",
                symbol=signal.symbol,
                direction=self._signal_to_direction(signal.signal),
                volume=self._calculate_position_size(signal),
                strategy_name=signal.strategy_name,
                comment=f"Signal strength: {signal.strength}, Confidence: {signal.confidence}"
            )
            
            # Submit for execution
            if self.execution_engine:
                self.execution_engine.submit_order(request)
            
        except Exception as e:
            self.logger.error(f"Error handling strategy signal: {e}", exc_info=True)
    
    def _on_execution_complete(self, result):
        """Handle execution completion."""
        try:
            self.logger.info(f"Execution completed: {result.order_id}")
            
            # Update statistics
            with self._lock:
                self.stats.total_trades += 1
            
        except Exception as e:
            self.logger.error(f"Error handling execution completion: {e}", exc_info=True)
    
    def _on_execution_failed(self, result):
        """Handle execution failure."""
        try:
            self.logger.error(f"Execution failed: {result.error_message}")
            
            # Send notification
            if self.notification_manager:
                self.notification_manager.send_error_alert(
                    error_type="Execution Failure",
                    message=f"Trade execution failed: {result.error_message}",
                    component="Execution Engine",
                    stack_trace=result.error_message
                )
            
        except Exception as e:
            self.logger.error(f"Error handling execution failure: {e}", exc_info=True)
    
    def _on_market_open(self):
        """Handle market open."""
        self.logger.info("Market opened - resuming trading operations")
        
        # Resume strategy execution
        if self.strategy_runner:
            self.strategy_runner.start_all_strategies()
    
    def _on_market_close(self):
        """Handle market close."""
        self.logger.info("Market closed - pausing trading operations")
        
        # Pause strategy execution
        if self.strategy_runner:
            self.strategy_runner.stop_all_strategies()
    
    def _should_execute_signal(self, signal: StrategySignal) -> bool:
        """Check if signal should be executed."""
        # Check if system is running
        if self.status != LiveTraderStatus.RUNNING:
            return False
        
        # Check if market is open
        if self.trading_scheduler and not self.trading_scheduler.is_market_open():
            return False
        
        # Check signal strength and confidence
        if signal.strength < 0.5 or signal.confidence < 0.6:
            return False
        
        # Check risk limits
        if self.risk_monitor:
            risk_level = self.risk_monitor.get_current_risk_level()
            if risk_level.value in ['high', 'critical']:
                self.logger.warning(f"Risk level too high ({risk_level.value}), skipping signal")
                return False
        
        return True
    
    def _signal_to_direction(self, signal: int):
        """Convert signal to order direction."""
        from app.trading.order import OrderDirection
        return OrderDirection.BUY if signal > 0 else OrderDirection.SELL
    
    def _calculate_position_size(self, signal: StrategySignal) -> float:
        """Calculate position size for signal."""
        # This is a simplified calculation
        # In practice, you'd use the risk manager to calculate proper position size
        base_size = 0.01  # Minimum lot size
        
        # Adjust based on signal strength and confidence
        adjusted_size = base_size * signal.strength * signal.confidence
        
        # Ensure minimum size
        return max(adjusted_size, base_size)
    
    def _update_statistics(self):
        """Update live trader statistics."""
        try:
            with self._lock:
                # Update uptime
                if self.start_time:
                    self.stats.uptime = datetime.now() - self.start_time
                
                # Update component statistics
                if self.strategy_runner:
                    strategy_stats = self.strategy_runner.get_statistics()
                    self.stats.total_signals = strategy_stats.get('total_signals', 0)
                
                if self.position_manager:
                    position_stats = self.position_manager.get_statistics()
                    self.stats.total_trades = position_stats.get('total_positions', 0)
                    self.stats.total_pnl = position_stats.get('total_pnl', 0.0)
                    self.stats.active_positions = position_stats.get('position_count', 0)
                
                if self.system_monitor:
                    system_stats = self.system_monitor.get_statistics()
                    self.stats.system_health = system_stats.get('current_status', 'unknown')
                
                if self.trading_scheduler:
                    scheduler_stats = self.trading_scheduler.get_statistics()
                    self.stats.market_status = scheduler_stats.get('current_status', 'unknown')
                
                self.stats.status = self.status
                self.stats.timestamp = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self._shutdown_event.set()
            self._running = False
            self.status = LiveTraderStatus.STOPPING
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        status = {
            'live_trader': {
                'status': self.status.value,
                'uptime': str(self.stats.uptime) if self.start_time else '0:00:00',
                'error_count': self.error_count
            }
        }
        
        # Add component statuses
        if self.execution_engine:
            status['execution_engine'] = self.execution_engine.get_statistics()
        
        if self.strategy_runner:
            status['strategy_runner'] = self.strategy_runner.get_statistics()
        
        if self.position_manager:
            status['position_manager'] = self.position_manager.get_statistics()
        
        if self.risk_monitor:
            status['risk_monitor'] = self.risk_monitor.get_statistics()
        
        if self.system_monitor:
            status['system_monitor'] = self.system_monitor.get_statistics()
        
        if self.performance_tracker:
            status['performance_tracker'] = self.performance_tracker.get_statistics()
        
        if self.trading_scheduler:
            status['trading_scheduler'] = self.trading_scheduler.get_statistics()
        
        return status 