from abc import ABC, abstractmethod
from typing import List, Optional
from app.trading.order import Order
from app.trading.position import Position

class Broker(ABC):
    """
    Abstract base class for a broker connection.
    """

    @abstractmethod
    def get_account_balance(self) -> float:
        """Returns the current account balance."""
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Returns a list of open positions."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Returns a list of open orders."""
        pass

    @abstractmethod
    def send_order(self, order: Order) -> Optional[str]:
        """
        Sends an order to the broker.
        Returns the broker-specific order ID if successful, otherwise None.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an open order."""
        pass

    @abstractmethod
    def close_position(self, position_id: str, amount: float) -> bool:
        """Closes an open position."""
        pass

    @abstractmethod
    def modify_position(self, position_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """Modifies an open position's stop loss or take profit."""
        pass

    @abstractmethod
    def modify_order(self, order_id: str, price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """Modifies a pending order's price, stop loss, or take profit."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str):
        """Gets information about a symbol."""
        pass
        
    @abstractmethod
    def get_current_price(self, symbol: str, price_type: str = 'ask') -> float:
        """Gets the current ask or bid price for a symbol."""
        pass 