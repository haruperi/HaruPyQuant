from typing import List, Dict, Optional
from unittest.mock import MagicMock
from app.trading.broker import Broker
from app.trading.risk_manager import RiskManager
from app.trading.order import Order, OrderType, OrderDirection, OrderStatus
from app.trading.position import Position, PositionDirection, PositionStatus
from app.util.logger import get_logger

logger = get_logger(__name__)

class Trader:
    """
    The Trader class orchestrates trading activities, including order creation,
    risk management, and position tracking.
    """
    def __init__(self, broker: Broker, risk_manager: RiskManager):
        self.broker = broker
        self.risk_manager = risk_manager
        self.open_orders: Dict[str, Order] = {}
        self.open_positions: Dict[str, Position] = {}
        self.trade_history: List[Position] = []
        self._initialize_state()

    def _initialize_state(self):
        """
        Initializes the trader's state from the broker.
        """
        logger.info("Initializing trader state from broker...")
        self.update()

    def update(self):
        """
        Updates the trader's state by fetching the latest info from the broker.
        This should be called periodically.
        """
        logger.info("Updating trader state...")
        self.open_orders = {o.order_id: o for o in self.broker.get_open_orders()}
        open_positions_from_broker = {p.order_id: p for p in self.broker.get_open_positions()}

        # Update PnL for open positions and move closed ones to history
        current_open_positions = {}
        for order_id, pos in self.open_positions.items():
            if order_id not in open_positions_from_broker:
                # Position was closed
                self.trade_history.append(pos)
            else:
                # Position is still open
                current_open_positions[order_id] = pos
                price = self.broker.get_current_price(pos.symbol)
                pos.update_pnl(price)
        
        self.open_positions = current_open_positions

        # Add any new positions that were not tracked before
        for order_id, pos in open_positions_from_broker.items():
            if order_id not in self.open_positions:
                self.open_positions[order_id] = pos

        logger.info(f"Trader state updated. Open orders: {len(self.open_orders)}, Open positions: {len(self.open_positions)}")

    def create_market_order(
        self,
        symbol: str,
        direction: OrderDirection,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = ""
    ) -> Optional[Order]:
        """
        Creates a market order.
        """
        logger.info(f"Creating market order for {symbol} {direction.name} with volume {volume}")
        return self._create_order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            direction=direction,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )

    def create_pending_order(
        self,
        symbol: str,
        order_type: OrderType,
        direction: OrderDirection,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = ""
    ) -> Optional[Order]:
        """
        Creates a pending order (LIMIT, STOP, etc.).
        """
        logger.info(f"Creating {order_type.name} order for {symbol} at {price}")
        if order_type == OrderType.MARKET:
            logger.error("Use create_market_order for market orders.")
            return None

        return self._create_order(
            symbol=symbol,
            order_type=order_type,
            direction=direction,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment
        )

    def _create_order(self, **kwargs) -> Optional[Order]:
        """
        Internal method to create and send an order.
        """
        order = Order(**kwargs)
        broker_order_id = self.broker.send_order(order)
        if broker_order_id:
            order.order_id = broker_order_id
            self.open_orders[order.order_id] = order
            logger.info(f"Order sent successfully. Broker ID: {broker_order_id}")
            return order
        else:
            logger.error(f"Failed to send order for {kwargs.get('symbol')}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open pending order.
        """
        if order_id not in self.open_orders:
            logger.error(f"Cannot cancel order. Order ID {order_id} not found.")
            return False

        logger.info(f"Attempting to cancel order {order_id}")
        success = self.broker.cancel_order(order_id)
        if success:
            if order_id in self.open_orders:
                self.open_orders.pop(order_id).cancel()
            logger.info(f"Order {order_id} cancelled successfully.")
        else:
            logger.error(f"Failed to cancel order {order_id}.")
        return success

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modifies a pending order.
        """
        if order_id not in self.open_orders:
            logger.error(f"Cannot modify order. Order ID {order_id} not found.")
            return False

        logger.info(f"Attempting to modify order {order_id}")
        success = self.broker.modify_order(order_id, price=price, stop_loss=stop_loss, take_profit=take_profit)
        if success:
            # Update local order state
            order = self.open_orders[order_id]
            if price is not None:
                order.price = price
            if stop_loss is not None:
                order.stop_loss = stop_loss
            if take_profit is not None:
                order.take_profit = take_profit
            logger.info(f"Order {order_id} modified successfully.")
        else:
            logger.error(f"Failed to modify order {order_id}.")
        return success

    def modify_position(self, order_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Modifies the SL/TP of an open position.
        """
        if order_id not in self.open_positions:
            logger.error(f"Cannot modify position. Order ID {order_id} not found.")
            return False

        logger.info(f"Attempting to modify position {order_id} (SL: {stop_loss}, TP: {take_profit})")
        position = self.open_positions[order_id]
        success = self.broker.modify_position(position.order_id, stop_loss=stop_loss, take_profit=take_profit)

        if success:
            if stop_loss is not None:
                position.stop_loss = stop_loss
            if take_profit is not None:
                position.take_profit = take_profit
            logger.info(f"Position {order_id} modified successfully.")
        else:
            logger.error(f"Failed to modify position {order_id}.")
        return success

    def close_position(self, order_id: str, volume: float) -> bool:
        """
        Closes a portion or all of an open position.
        """
        if order_id not in self.open_positions:
            logger.error(f"Cannot close position. Order ID {order_id} not found in open positions.")
            return False
            
        position = self.open_positions[order_id]
        if volume > position.volume:
            logger.error(f"Cannot close {volume} lots. Position size is {position.volume}.")
            return False

        logger.info(f"Attempting to close {volume} lots of position {order_id} ({position.symbol})")
        
        # The broker is responsible for creating a closing order.
        success = self.broker.close_position(position.order_id, volume)
        
        if success:
            logger.info(f"Close request for position {order_id} sent successfully.")
        else:
            logger.error(f"Failed to send close request for position {order_id}.")
            
        return success 

if __name__ == '__main__':
    # This block provides usage examples for the Trader class.
    # It uses mock objects to simulate a broker and risk manager.

    print("--- Trader Usage Examples ---")

    # 1. Setup mock components
    mock_broker = MagicMock(spec=Broker)
    mock_risk_manager = MagicMock(spec=RiskManager)

    # Configure mock broker behavior
    mock_broker.get_open_orders.return_value = []
    mock_broker.get_open_positions.return_value = []
    mock_broker.send_order.side_effect = lambda order: f"broker_{order.order_id}"
    mock_broker.cancel_order.return_value = True
    mock_broker.modify_order.return_value = True
    mock_broker.modify_position.return_value = True
    mock_broker.close_position.return_value = True

    # 2. Initialize the Trader
    trader = Trader(broker=mock_broker, risk_manager=mock_risk_manager)
    print("\nTrader initialized with mock components.")

    # 3. Example: Entering a trade (Market Order)
    print("\n--- Example: Entering a Trade (Market Order) ---")
    market_order = trader.create_market_order(
        symbol="EURUSD",
        direction=OrderDirection.BUY,
        volume=0.1,
        stop_loss=1.0700,
        take_profit=1.0800
    )
    if market_order:
        print(f"Market order created: {market_order.order_id}")
        # In a real scenario, this order would be filled and a position created.
        # We'll simulate this for later examples.
        market_order.fill(filled_price=1.0750)
        position = Position.from_order(market_order)
        trader.open_positions[position.order_id] = position
        print(f"Position opened: {position.order_id}")

    # 4. Example: Adding a Pending Trade (Limit Order)
    print("\n--- Example: Adding a Pending Trade (Limit Order) ---")
    limit_order = trader.create_pending_order(
        symbol="EURUSD",
        order_type=OrderType.LIMIT,
        direction=OrderDirection.SELL,
        volume=0.1,
        price=1.0850,
        stop_loss=1.0900,
        take_profit=1.0800
    )
    if limit_order:
        print(f"Pending limit order created: {limit_order.order_id}")

    # 5. Example: Modifying a Pending Trade
    print("\n--- Example: Modifying a Pending Trade ---")
    if limit_order:
        success = trader.modify_order(limit_order.order_id, price=1.0860, stop_loss=1.0910)
        if success:
            print(f"Pending order {limit_order.order_id} modified. New price: {trader.open_orders[limit_order.order_id].price}")

    # 6. Example: Deleting a Pending Trade (Cancelling an Order)
    print("\n--- Example: Deleting a Pending Trade ---")
    if limit_order:
        success = trader.cancel_order(limit_order.order_id)
        if success:
            print(f"Pending order {limit_order.order_id} cancelled.")
            assert limit_order.order_id not in trader.open_orders

    # 7. Example: Modifying an Open Position
    print("\n--- Example: Modifying an Open Position ---")
    if market_order:
        success = trader.modify_position(market_order.order_id, stop_loss=1.0710)
        if success:
            print(f"Position {market_order.order_id} modified. New SL: {trader.open_positions[market_order.order_id].stop_loss}")

    # 8. Example: Closing a Position
    print("\n--- Example: Closing a Position ---")
    if market_order:
        success = trader.close_position(market_order.order_id, volume=0.1)
        if success:
            print(f"Close order sent for position {market_order.order_id}.")
            # In a real system, the update() loop would remove this from open_positions.
            trader.open_positions.pop(market_order.order_id)
            print("Position closed and removed from trader's active positions.")

    print("\n--- End of Examples ---") 