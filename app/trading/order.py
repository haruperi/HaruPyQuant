# app/trading/order.py

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC
from typing import Optional

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"

class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    """
    Represents a trading order.
    """
    symbol: str
    order_type: OrderType
    direction: OrderDirection
    volume: float
    order_id: str = field(default_factory=lambda: f"order_{int(datetime.now(UTC).timestamp() * 1000)}")
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    comment: str = ""

    def __post_init__(self):
        if self.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError("Price must be specified for LIMIT, STOP, and STOP_LIMIT orders.")
        if self.volume <= 0:
            raise ValueError("Order volume must be positive.")

    def fill(self, filled_price: float, filled_at: Optional[datetime] = None):
        """Marks the order as filled."""
        self.status = OrderStatus.FILLED
        self.filled_price = filled_price
        self.filled_at = filled_at or datetime.now(UTC)

    def cancel(self):
        """Marks the order as cancelled."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.REJECTED]:
            self.status = OrderStatus.CANCELLED
        else:
            raise ValueError(f"Cannot cancel order with status {self.status.name}")

    def reject(self):
        """Marks the order as rejected."""
        self.status = OrderStatus.REJECTED 