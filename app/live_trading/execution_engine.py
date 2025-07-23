"""
Execution Engine

This module handles trade execution with retry logic, error handling, and execution monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.trading.trader import Trader
from app.trading.order import Order, OrderType, OrderDirection, OrderStatus
from app.trading.position import Position, PositionDirection, PositionStatus
from app.trading.broker import Broker
from app.trading.risk_manager import RiskManager
from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ExecutionRequest:
    """Trade execution request."""
    id: str
    symbol: str
    direction: OrderDirection
    volume: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: ExecutionStatus = ExecutionStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    error_message: str = ""
    execution_time: Optional[float] = None
    filled_price: Optional[float] = None
    filled_volume: Optional[float] = None
    slippage: Optional[float] = None


@dataclass
class ExecutionResult:
    """Execution result."""
    success: bool
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_volume: Optional[float] = None
    slippage: Optional[float] = None
    execution_time: Optional[float] = None
    error_message: str = ""
    retry_count: int = 0


class ExecutionEngine:
    """Handles trade execution with retry logic and monitoring."""
    
    def __init__(self, trader: Trader, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize execution engine.
        
        Args:
            trader: Trader instance for order execution
            notification_manager: Notification manager for alerts
        """
        self.trader = trader
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Execution tracking
        self.pending_requests: Dict[str, ExecutionRequest] = {}
        self.completed_requests: Dict[str, ExecutionRequest] = {}
        self.failed_requests: Dict[str, ExecutionRequest] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_slippage': 0.0,
            'retry_count': 0
        }
        
        # Threading
        self._lock = threading.Lock()
        self._execution_thread = None
        self._running = False
        
        # Callbacks
        self.on_execution_complete: Optional[Callable[[ExecutionResult], None]] = None
        self.on_execution_failed: Optional[Callable[[ExecutionResult], None]] = None
        
        self.logger.info("Execution engine initialized")
    
    def start(self):
        """Start the execution engine."""
        if self._running:
            self.logger.warning("Execution engine already running")
            return
        
        self._running = True
        self._execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True,
            name="ExecutionEngine"
        )
        self._execution_thread.start()
        self.logger.info("Execution engine started")
    
    def stop(self):
        """Stop the execution engine."""
        if not self._running:
            return
        
        self._running = False
        if self._execution_thread:
            self._execution_thread.join(timeout=10)
        
        self.logger.info("Execution engine stopped")
    
    def submit_order(self, request: ExecutionRequest) -> str:
        """
        Submit an order for execution.
        
        Args:
            request: Execution request
            
        Returns:
            Request ID
        """
        with self._lock:
            self.pending_requests[request.id] = request
            self.stats['total_requests'] += 1
        
        self.logger.info(f"Order submitted: {request.id} - {request.symbol} {request.direction.name} {request.volume}")
        return request.id
    
    def cancel_order(self, request_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            request_id: Request ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        with self._lock:
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                request.status = ExecutionStatus.CANCELLED
                self.completed_requests[request_id] = request
                del self.pending_requests[request_id]
                
                self.logger.info(f"Order cancelled: {request_id}")
                return True
        
        return False
    
    def get_request_status(self, request_id: str) -> Optional[ExecutionStatus]:
        """Get status of an execution request."""
        with self._lock:
            if request_id in self.pending_requests:
                return self.pending_requests[request_id].status
            elif request_id in self.completed_requests:
                return self.completed_requests[request_id].status
            elif request_id in self.failed_requests:
                return self.failed_requests[request_id].status
        
        return None
    
    def _execution_loop(self):
        """Main execution loop."""
        while self._running:
            try:
                # Process pending requests
                self._process_pending_requests()
                
                # Clean up old completed requests
                self._cleanup_old_requests()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _process_pending_requests(self):
        """Process all pending execution requests."""
        with self._lock:
            requests_to_process = list(self.pending_requests.values())
        
        for request in requests_to_process:
            if not self._running:
                break
            
            try:
                self._execute_request(request)
            except Exception as e:
                self.logger.error(f"Error executing request {request.id}: {e}", exc_info=True)
                self._handle_execution_error(request, str(e))
    
    def _execute_request(self, request: ExecutionRequest):
        """Execute a single request."""
        if request.attempts >= request.max_attempts:
            self._handle_execution_failure(request, "Max attempts exceeded")
            return
        
        # Update request status
        request.status = ExecutionStatus.EXECUTING
        request.attempts += 1
        
        start_time = time.time()
        
        try:
            # Execute the order
            if request.order_type == OrderType.MARKET:
                order = self._execute_market_order(request)
            else:
                order = self._execute_pending_order(request)
            
            if order:
                # Calculate execution metrics
                execution_time = time.time() - start_time
                slippage = self._calculate_slippage(request, order)
                
                # Create result
                result = ExecutionResult(
                    success=True,
                    order_id=order.order_id,
                    filled_price=order.filled_price if hasattr(order, 'filled_price') else None,
                    filled_volume=order.volume,
                    slippage=slippage,
                    execution_time=execution_time,
                    retry_count=request.attempts - 1
                )
                
                # Update request
                request.status = ExecutionStatus.COMPLETED
                request.execution_time = execution_time
                request.filled_price = result.filled_price
                request.filled_volume = result.filled_volume
                request.slippage = slippage
                
                # Move to completed
                with self._lock:
                    self.completed_requests[request.id] = request
                    del self.pending_requests[request.id]
                    self.stats['successful_executions'] += 1
                
                # Update statistics
                self._update_execution_stats(result)
                
                # Send notification
                self._send_execution_notification(request, result)
                
                # Call callback
                if self.on_execution_complete:
                    self.on_execution_complete(result)
                
                self.logger.info(f"Order executed successfully: {request.id} - {result.order_id}")
                
            else:
                # Retry logic
                if request.attempts < request.max_attempts:
                    self.logger.warning(f"Execution failed, retrying: {request.id} (attempt {request.attempts})")
                    request.status = ExecutionStatus.PENDING
                    time.sleep(1)  # Wait before retry
                else:
                    self._handle_execution_failure(request, "Order creation failed")
        
        except Exception as e:
            self._handle_execution_error(request, str(e))
    
    def _execute_market_order(self, request: ExecutionRequest) -> Optional[Order]:
        """Execute a market order."""
        return self.trader.create_market_order(
            symbol=request.symbol,
            direction=request.direction,
            volume=request.volume,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=f"{request.strategy_name}: {request.comment}"
        )
    
    def _execute_pending_order(self, request: ExecutionRequest) -> Optional[Order]:
        """Execute a pending order."""
        if not request.price:
            raise ValueError("Price required for pending orders")
        
        return self.trader.create_pending_order(
            symbol=request.symbol,
            order_type=request.order_type,
            direction=request.direction,
            volume=request.volume,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=f"{request.strategy_name}: {request.comment}"
        )
    
    def _calculate_slippage(self, request: ExecutionRequest, order: Order) -> Optional[float]:
        """Calculate slippage for executed order."""
        if request.price and hasattr(order, 'filled_price') and order.filled_price:
            if request.direction == OrderDirection.BUY:
                return order.filled_price - request.price
            else:
                return request.price - order.filled_price
        return None
    
    def _handle_execution_error(self, request: ExecutionRequest, error_message: str):
        """Handle execution error."""
        request.error_message = error_message
        self.stats['retry_count'] += 1
        
        if request.attempts >= request.max_attempts:
            self._handle_execution_failure(request, error_message)
    
    def _handle_execution_failure(self, request: ExecutionRequest, error_message: str):
        """Handle execution failure after max attempts."""
        request.status = ExecutionStatus.FAILED
        request.error_message = error_message
        
        # Create result
        result = ExecutionResult(
            success=False,
            error_message=error_message,
            retry_count=request.attempts
        )
        
        # Move to failed
        with self._lock:
            self.failed_requests[request.id] = request
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
            self.stats['failed_executions'] += 1
        
        # Send notification
        self._send_execution_notification(request, result)
        
        # Call callback
        if self.on_execution_failed:
            self.on_execution_failed(result)
        
        self.logger.error(f"Order execution failed: {request.id} - {error_message}")
    
    def _send_execution_notification(self, request: ExecutionRequest, result: ExecutionResult):
        """Send execution notification."""
        if not self.notification_manager:
            return
        
        try:
            if result.success:
                self.notification_manager.send_trading_alert(
                    symbol=request.symbol,
                    action=request.direction.name,
                    price=result.filled_price or 0.0,
                    reason=f"Strategy: {request.strategy_name}",
                    account=self.trader.broker.get_account_info().get('name', 'Unknown'),
                    strategy=request.strategy_name,
                    risk_level="Medium"
                )
            else:
                self.notification_manager.send_error_alert(
                    error_type="Execution Error",
                    message=f"Failed to execute {request.symbol} order",
                    component="Execution Engine",
                    stack_trace=result.error_message
                )
        except Exception as e:
            self.logger.error(f"Failed to send execution notification: {e}")
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        with self._lock:
            if result.execution_time:
                # Update average execution time
                total_time = self.stats['average_execution_time'] * (self.stats['successful_executions'] - 1)
                total_time += result.execution_time
                self.stats['average_execution_time'] = total_time / self.stats['successful_executions']
            
            if result.slippage:
                self.stats['total_slippage'] += abs(result.slippage)
    
    def _cleanup_old_requests(self):
        """Clean up old completed and failed requests."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            # Clean completed requests
            old_completed = [
                req_id for req_id, req in self.completed_requests.items()
                if req.timestamp < cutoff_time
            ]
            for req_id in old_completed:
                del self.completed_requests[req_id]
            
            # Clean failed requests
            old_failed = [
                req_id for req_id, req in self.failed_requests.items()
                if req.timestamp < cutoff_time
            ]
            for req_id in old_failed:
                del self.failed_requests[req_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'pending_requests': len(self.pending_requests),
                'completed_requests': len(self.completed_requests),
                'failed_requests': len(self.failed_requests),
                'success_rate': (
                    stats['successful_executions'] / max(stats['total_requests'], 1) * 100
                )
            })
            return stats
    
    def get_pending_requests(self) -> List[ExecutionRequest]:
        """Get all pending requests."""
        with self._lock:
            return list(self.pending_requests.values())
    
    def get_recent_executions(self, limit: int = 50) -> List[ExecutionRequest]:
        """Get recent executions."""
        with self._lock:
            all_requests = list(self.completed_requests.values()) + list(self.failed_requests.values())
            sorted_requests = sorted(all_requests, key=lambda x: x.timestamp, reverse=True)
            return sorted_requests[:limit] 