from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC
from typing import Optional

from app.trading.order import OrderDirection, Order, OrderStatus

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

class PositionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Position:
    """
    Represents an open trading position.
    """
    symbol: str
    direction: PositionDirection
    volume: float
    entry_price: float
    status: PositionStatus = PositionStatus.OPEN
    entry_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    exit_price: Optional[float] = None
    exit_at: Optional[datetime] = None
    pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: str = ""
    comment: str = ""

    def __post_init__(self):
        if self.volume <= 0:
            raise ValueError("Position volume must be positive.")

    @classmethod
    def from_order(cls, order: Order) -> "Position":
        """
        Creates a Position from a filled Order.
        """
        if order.status != OrderStatus.FILLED:
            raise ValueError("Cannot create position from an unfilled order.")
        
        if order.direction == OrderDirection.BUY:
            direction = PositionDirection.LONG
        else:
            direction = PositionDirection.SHORT
            
        return cls(
            symbol=order.symbol,
            direction=direction,
            volume=order.volume,
            entry_price=order.filled_price,
            entry_at=order.filled_at,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            order_id=order.order_id,
            comment=order.comment
        )

    def update_pnl(self, current_price: float):
        """
        Updates the profit and loss of the position based on the current price.
        """
        if self.status == PositionStatus.CLOSED:
            return

        price_diff = current_price - self.entry_price
        if self.direction == PositionDirection.SHORT:
            price_diff = -price_diff

        # This is a simplified PnL calculation.
        # A more accurate calculation would consider pip value, contract size, etc.
        self.pnl = price_diff * self.volume

    def close(self, exit_price: float, exit_at: Optional[datetime] = None):
        """
        Closes the position.
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Position is already closed.")
            
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_at = exit_at or datetime.now(UTC)
        self.update_pnl(exit_price) 