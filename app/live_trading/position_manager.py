"""
Position Manager

This module handles position tracking, modification, and management.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.trading.trader import Trader
from app.trading.position import Position, PositionDirection, PositionStatus
from app.trading.order import Order, OrderDirection
from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class PositionAction(Enum):
    """Position actions."""
    OPEN = "open"
    CLOSE = "close"
    MODIFY = "modify"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class PositionUpdate:
    """Position update event."""
    position_id: str
    symbol: str
    action: PositionAction
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionMetrics:
    """Position performance metrics."""
    position_id: str
    symbol: str
    entry_price: float
    current_price: float
    volume: float
    direction: PositionDirection
    unrealized_pnl: float
    unrealized_pnl_percent: float
    duration: timedelta
    max_profit: float
    max_loss: float
    drawdown: float
    timestamp: datetime = field(default_factory=datetime.now)


class PositionManager:
    """Manages position tracking and modification."""
    
    def __init__(self, trader: Trader, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize position manager.
        
        Args:
            trader: Trader instance for position operations
            notification_manager: Notification manager for alerts
        """
        self.trader = trader
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.position_updates: List[PositionUpdate] = []
        
        # Performance tracking
        self.position_metrics: Dict[str, PositionMetrics] = {}
        self.max_profits: Dict[str, float] = {}
        self.max_losses: Dict[str, float] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._update_thread = None
        self._running = False
        
        # Statistics
        self.stats = {
            'total_positions': 0,
            'open_positions': 0,
            'closed_positions': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0
        }
        
        self.logger.info("Position manager initialized")
    
    def start(self):
        """Start the position manager."""
        if self._running:
            self.logger.warning("Position manager already running")
            return
        
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="PositionManager"
        )
        self._update_thread.start()
        self.logger.info("Position manager started")
    
    def stop(self):
        """Stop the position manager."""
        if not self._running:
            return
        
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=10)
        
        self.logger.info("Position manager stopped")
    
    def update_positions(self):
        """Update position information from broker."""
        try:
            # Get current positions from broker
            broker_positions = self.trader.broker.get_open_positions()
            
            with self._lock:
                # Track closed positions
                closed_positions = []
                for pos_id, position in self.positions.items():
                    if pos_id not in [p.order_id for p in broker_positions]:
                        closed_positions.append(position)
                        self._handle_position_closed(position)
                
                # Update existing positions
                for broker_pos in broker_positions:
                    if broker_pos.order_id in self.positions:
                        old_position = self.positions[broker_pos.order_id]
                        self._update_position(old_position, broker_pos)
                    else:
                        self._handle_position_opened(broker_pos)
                
                # Update statistics
                self.stats['open_positions'] = len(self.positions)
                self.stats['closed_positions'] = len(self.position_history)
            
            # Send notifications for closed positions
            for position in closed_positions:
                self._send_position_notification(position, PositionAction.CLOSE)
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}", exc_info=True)
    
    def close_position(self, position_id: str, volume: Optional[float] = None) -> bool:
        """
        Close a position.
        
        Args:
            position_id: Position ID to close
            volume: Volume to close (None for full position)
            
        Returns:
            True if close request sent successfully
        """
        with self._lock:
            if position_id not in self.positions:
                self.logger.error(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
            close_volume = volume or position.volume
        
        try:
            success = self.trader.close_position(position_id, close_volume)
            if success:
                self.logger.info(f"Close request sent for position {position_id}")
                return True
            else:
                self.logger.error(f"Failed to close position {position_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    def modify_position(self, position_id: str, stop_loss: Optional[float] = None, 
                       take_profit: Optional[float] = None) -> bool:
        """
        Modify position stop loss or take profit.
        
        Args:
            position_id: Position ID to modify
            stop_loss: New stop loss level
            take_profit: New take profit level
            
        Returns:
            True if modification successful
        """
        with self._lock:
            if position_id not in self.positions:
                self.logger.error(f"Position not found: {position_id}")
                return False
        
        try:
            success = self.trader.modify_position(position_id, stop_loss, take_profit)
            if success:
                # Update local position
                with self._lock:
                    position = self.positions[position_id]
                    if stop_loss is not None:
                        position.stop_loss = stop_loss
                    if take_profit is not None:
                        position.take_profit = take_profit
                
                self.logger.info(f"Position {position_id} modified successfully")
                return True
            else:
                self.logger.error(f"Failed to modify position {position_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying position {position_id}: {e}")
            return False
    
    def scale_in_position(self, position_id: str, additional_volume: float) -> bool:
        """
        Scale into an existing position.
        
        Args:
            position_id: Position ID to scale into
            additional_volume: Additional volume to add
            
        Returns:
            True if scale-in successful
        """
        with self._lock:
            if position_id not in self.positions:
                self.logger.error(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
        
        try:
            # Create a new market order in the same direction
            direction = OrderDirection.BUY if position.direction == PositionDirection.LONG else OrderDirection.SELL
            
            order = self.trader.create_market_order(
                symbol=position.symbol,
                direction=direction,
                volume=additional_volume,
                comment=f"Scale-in to {position_id}"
            )
            
            if order:
                self.logger.info(f"Scale-in order created for position {position_id}")
                return True
            else:
                self.logger.error(f"Failed to create scale-in order for position {position_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error scaling into position {position_id}: {e}")
            return False
    
    def scale_out_position(self, position_id: str, volume_to_close: float) -> bool:
        """
        Scale out of an existing position.
        
        Args:
            position_id: Position ID to scale out of
            volume_to_close: Volume to close
            
        Returns:
            True if scale-out successful
        """
        return self.close_position(position_id, volume_to_close)
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position."""
        with self._lock:
            return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol."""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        with self._lock:
            return list(self.positions.values())
    
    def get_position_metrics(self, position_id: str) -> Optional[PositionMetrics]:
        """Get metrics for a specific position."""
        with self._lock:
            return self.position_metrics.get(position_id)
    
    def get_total_pnl(self) -> float:
        """Get total unrealized PnL."""
        with self._lock:
            return sum(pos.pnl for pos in self.positions.values())
    
    def get_total_volume(self) -> float:
        """Get total position volume."""
        with self._lock:
            return sum(pos.volume for pos in self.positions.values())
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        with self._lock:
            return len(self.positions)
    
    def get_recent_updates(self, limit: int = 100) -> List[PositionUpdate]:
        """Get recent position updates."""
        with self._lock:
            return sorted(self.position_updates, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def _update_loop(self):
        """Main update loop."""
        while self._running:
            try:
                self.update_positions()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in position update loop: {e}", exc_info=True)
                time.sleep(60)  # Wait longer on error
    
    def _handle_position_opened(self, position: Position):
        """Handle newly opened position."""
        self.positions[position.order_id] = position
        self.stats['total_positions'] += 1
        
        # Initialize metrics
        self.position_metrics[position.order_id] = PositionMetrics(
            position_id=position.order_id,
            symbol=position.symbol,
            entry_price=position.entry_price,
            current_price=position.entry_price,
            volume=position.volume,
            direction=position.direction,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
            duration=timedelta(0),
            max_profit=0.0,
            max_loss=0.0,
            drawdown=0.0
        )
        
        # Initialize max profit/loss tracking
        self.max_profits[position.order_id] = 0.0
        self.max_losses[position.order_id] = 0.0
        
        # Record update
        self.position_updates.append(PositionUpdate(
            position_id=position.order_id,
            symbol=position.symbol,
            action=PositionAction.OPEN,
            new_value=position.volume
        ))
        
        self.logger.info(f"Position opened: {position.order_id} - {position.symbol} {position.direction.name} {position.volume}")
    
    def _handle_position_closed(self, position: Position):
        """Handle closed position."""
        # Move to history
        self.position_history.append(position)
        
        # Remove from active positions
        del self.positions[position.order_id]
        
        # Remove metrics
        if position.order_id in self.position_metrics:
            del self.position_metrics[position.order_id]
        
        # Remove max profit/loss tracking
        if position.order_id in self.max_profits:
            del self.max_profits[position.order_id]
        if position.order_id in self.max_losses:
            del self.max_losses[position.order_id]
        
        # Record update
        self.position_updates.append(PositionUpdate(
            position_id=position.order_id,
            symbol=position.symbol,
            action=PositionAction.CLOSE,
            old_value=position.volume
        ))
        
        self.logger.info(f"Position closed: {position.order_id} - {position.symbol}")
    
    def _update_position(self, old_position: Position, new_position: Position):
        """Update existing position."""
        # Update position data
        old_position.volume = new_position.volume
        old_position.pnl = new_position.pnl
        old_position.stop_loss = new_position.stop_loss
        old_position.take_profit = new_position.take_profit
        
        # Update metrics
        if old_position.order_id in self.position_metrics:
            metrics = self.position_metrics[old_position.order_id]
            metrics.current_price = new_position.entry_price  # Use entry price as current for now
            metrics.unrealized_pnl = new_position.pnl
            metrics.duration = datetime.now() - metrics.timestamp
            
            # Update max profit/loss
            if new_position.pnl > self.max_profits[old_position.order_id]:
                self.max_profits[old_position.order_id] = new_position.pnl
                metrics.max_profit = new_position.pnl
            
            if new_position.pnl < self.max_losses[old_position.order_id]:
                self.max_losses[old_position.order_id] = new_position.pnl
                metrics.max_loss = new_position.pnl
            
            # Calculate drawdown
            if self.max_profits[old_position.order_id] > 0:
                metrics.drawdown = (self.max_profits[old_position.order_id] - new_position.pnl) / self.max_profits[old_position.order_id] * 100
    
    def _send_position_notification(self, position: Position, action: PositionAction):
        """Send position notification."""
        if not self.notification_manager:
            return
        
        try:
            if action == PositionAction.CLOSE:
                # Calculate final PnL
                pnl = position.pnl
                pnl_percent = (pnl / (position.entry_price * position.volume)) * 100 if position.entry_price * position.volume > 0 else 0
                
                self.notification_manager.send_position_update(
                    symbol=position.symbol,
                    position_type=position.direction.name,
                    size=position.volume,
                    entry_price=position.entry_price,
                    current_price=position.entry_price,  # Use entry price as current
                    pnl=pnl,
                    pnl_percent=pnl_percent
                )
        except Exception as e:
            self.logger.error(f"Failed to send position notification: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get position manager statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'total_pnl': self.get_total_pnl(),
                'total_volume': self.get_total_volume(),
                'position_count': len(self.positions),
                'recent_updates': len(self.position_updates)
            })
            return stats 