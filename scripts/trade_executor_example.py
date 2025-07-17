"""
This script provides a live demonstration of the Trader class, using a real
connection to the MetaTrader 5 terminal.

**WARNING: This script will execute real trades if connected to a live
or demo MT5 account. Use with a demo account for testing purposes.**
"""
import sys
import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import MetaTrader5 as mt5
from app.data.mt5_client import MT5Client
from app.trading.broker import Broker
from app.trading.order import Order, OrderDirection, OrderType, OrderStatus
from app.trading.position import Position, PositionDirection
from app.trading.risk_manager import RiskManager
from app.trading.trader import Trader
from app.util.logger import get_logger

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class MT5Broker(Broker):
    """
    Broker implementation for MetaTrader 5. Translates between the application's
    data models and the MT5 API.
    """

    def __init__(self, mt5_client: MT5Client):
        self.client = mt5_client
        self._symbol_info_cache: Dict[str, Any] = {}

    def _get_cached_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol not in self._symbol_info_cache:
            try:
                self._symbol_info_cache[symbol] = self.client.get_symbol_info(symbol)
            except RuntimeError as e:
                logger.error(f"Failed to get symbol info for {symbol}: {e}")
                return None
        return self._symbol_info_cache[symbol]

    def _map_mt5_order_type_to_order_direction(self, mt5_order_type: int) -> OrderDirection:
        if mt5_order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT]:
            return OrderDirection.BUY
        return OrderDirection.SELL

    def _map_mt5_order_type_to_order_type(self, mt5_order_type: int) -> OrderType:
        return {
            mt5.ORDER_TYPE_BUY: OrderType.MARKET,
            mt5.ORDER_TYPE_SELL: OrderType.MARKET,
            mt5.ORDER_TYPE_BUY_LIMIT: OrderType.LIMIT,
            mt5.ORDER_TYPE_SELL_LIMIT: OrderType.LIMIT,
            mt5.ORDER_TYPE_BUY_STOP: OrderType.STOP,
            mt5.ORDER_TYPE_SELL_STOP: OrderType.STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: OrderType.STOP_LIMIT,
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: OrderType.STOP_LIMIT,
        }.get(mt5_order_type, OrderType.MARKET)

    def _map_order_to_mt5_order_type(self, order: Order) -> Optional[int]:
        mapping = {
            (OrderDirection.BUY, OrderType.MARKET): mt5.ORDER_TYPE_BUY,
            (OrderDirection.SELL, OrderType.MARKET): mt5.ORDER_TYPE_SELL,
            (OrderDirection.BUY, OrderType.LIMIT): mt5.ORDER_TYPE_BUY_LIMIT,
            (OrderDirection.SELL, OrderType.LIMIT): mt5.ORDER_TYPE_SELL_LIMIT,
            (OrderDirection.BUY, OrderType.STOP): mt5.ORDER_TYPE_BUY_STOP,
            (OrderDirection.SELL, OrderType.STOP): mt5.ORDER_TYPE_SELL_STOP,
        }
        return mapping.get((order.direction, order.order_type))

    def get_account_balance(self) -> float:
        info = self.client.get_account_info()
        return info['balance'] if info else 0.0

    def get_open_positions(self) -> List[Position]:
        positions_raw = self.client.get_positions()
        return [
            Position(
                order_id=str(p['ticket']),
                symbol=p['symbol'],
                direction=PositionDirection.LONG if p['type'] == mt5.POSITION_TYPE_BUY else PositionDirection.SHORT,
                volume=p['volume'],
                entry_price=p['price_open'],
                stop_loss=p['sl'],
                take_profit=p['tp'],
                entry_at=datetime.fromtimestamp(p['time'], tz=UTC),
                comment=p['comment'],
            )
            for p in positions_raw
        ]

    def get_open_orders(self) -> List[Order]:
        orders_raw = self.client.get_orders()
        return [
            Order(
                order_id=str(o['ticket']),
                symbol=o['symbol'],
                order_type=self._map_mt5_order_type_to_order_type(o['type']),
                direction=self._map_mt5_order_type_to_order_direction(o['type']),
                volume=o['volume_current'],
                price=o['price_open'],
                stop_loss=o['sl'],
                take_profit=o['tp'],
                status=OrderStatus.PENDING,
                comment=o['comment'],
            )
            for o in orders_raw
        ]

    def send_order(self, order: Order) -> Optional[str]:
        mt5_order_type = self._map_order_to_mt5_order_type(order)
        if mt5_order_type is None:
            logger.error(f"Unsupported order type: {order.order_type}")
            return None

        symbol_info = self._get_cached_symbol_info(order.symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {order.symbol}, cannot send order.")
            return None

        # Determine the correct filling type by checking the bitmask
        allowed_filling_modes = symbol_info['filling_modes']
        filling_type = mt5.ORDER_FILLING_FOK  # Default

        # Prefer IOC, then FOK, then RETURN
        if allowed_filling_modes & (1 << mt5.ORDER_FILLING_IOC):
            filling_type = mt5.ORDER_FILLING_IOC
        elif allowed_filling_modes & (1 << mt5.ORDER_FILLING_FOK):
            filling_type = mt5.ORDER_FILLING_FOK
        elif allowed_filling_modes & (1 << mt5.ORDER_FILLING_RETURN):
            filling_type = mt5.ORDER_FILLING_RETURN
        else:
            logger.warning(f"Could not determine a supported filling mode from flags: {allowed_filling_modes}. Using default FOK.")

        price = order.price
        if order.order_type == OrderType.MARKET:
            price = self.get_current_price(order.symbol, 'ask' if order.direction == OrderDirection.BUY else 'bid')

        request = {
            "action": mt5.TRADE_ACTION_DEAL if order.order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5_order_type,
            "price": price,
            "sl": order.stop_loss or 0.0,
            "tp": order.take_profit or 0.0,
            "comment": order.comment or "",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = self.client.order_send(request)
        if result and result.get('order'):
            return str(result['order'])
        return None

    def cancel_order(self, order_id: str) -> bool:
        request = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(order_id)}
        return self.client.order_send(request) is not None

    def close_position(self, position_id: str, amount: float) -> bool:
        # MT5 requires creating an opposing order to close a position.
        positions = mt5.positions_get(ticket=int(position_id))  # type: ignore
        if not positions:
            return False
        
        pos = positions[0]
        price = self.get_current_price(pos.symbol, 'bid' if pos.type == mt5.POSITION_TYPE_BUY else 'ask')

        symbol_info = self._get_cached_symbol_info(pos.symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {pos.symbol}, cannot close position.")
            return False

        # Determine the correct filling type by checking the bitmask
        allowed_filling_modes = symbol_info['filling_modes']
        filling_type = mt5.ORDER_FILLING_FOK  # Default

        if allowed_filling_modes & (1 << mt5.ORDER_FILLING_IOC):
            filling_type = mt5.ORDER_FILLING_IOC
        elif allowed_filling_modes & (1 << mt5.ORDER_FILLING_FOK):
            filling_type = mt5.ORDER_FILLING_FOK
        elif allowed_filling_modes & (1 << mt5.ORDER_FILLING_RETURN):
            filling_type = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": amount,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": price,
            "comment": f"Close position {pos.ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        return self.client.order_send(request) is not None

    def modify_position(self, position_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        positions = mt5.positions_get(ticket=int(position_id))  # type: ignore
        if not positions:
            return False
        
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": stop_loss if stop_loss is not None else pos.sl,
            "tp": take_profit if take_profit is not None else pos.tp,
        }
        return self.client.order_send(request) is not None

    def modify_order(self, order_id: str, price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        orders = mt5.orders_get(ticket=int(order_id))  # type: ignore
        if not orders:
            return False
            
        order = orders[0]
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": order.ticket,
            "price": price if price is not None else order.price_open,
            "sl": stop_loss if stop_loss is not None else order.sl,
            "tp": take_profit if take_profit is not None else order.tp,
        }
        return self.client.order_send(request) is not None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get_cached_symbol_info(symbol)
        
    def get_current_price(self, symbol: str, price_type: str = 'ask') -> float:
        tick = self.client.get_tick(symbol)
        if tick and price_type in tick:
            return tick[price_type]
        
        # Fallback if tick fails
        info = self._get_cached_symbol_info(symbol)
        if info and price_type in info:
            return info[price_type]
            
        raise RuntimeError(f"Could not retrieve current price for {symbol}")

def main():
    """Main execution block."""
    logger.info("--- Live Trader Execution Example ---")

    try:
        # 1. Initialize MT5 Client
        mt5_client = MT5Client()
        if not mt5_client.is_connected():
            logger.error("Failed to connect to MT5. Exiting.")
            return
        
        

        # 2. Setup components
        broker = MT5Broker(mt5_client)
        account_balance = broker.get_account_balance()
        risk_manager = RiskManager(mt5_client=mt5_client, account_balance=account_balance, risk_percentage=1.0)
        trader = Trader(broker=broker, risk_manager=risk_manager)
        
        logger.info(f"Trader initialized. Account Balance: ${account_balance:.2f}")

        

        # --- Live Trading Examples ---
        # Note: These actions are REAL. Use a demo account.
        symbol = "EURUSD"
        
        # A. Clean up existing test orders/positions first
        logger.info("\n--- Cleaning up previous test trades ---")
        for p in broker.get_open_positions():
            if "test trade" in p.comment:
                logger.info(f"Closing position {p.order_id}...")
                trader.close_position(p.order_id, p.volume)
        for o in broker.get_open_orders():
             if "test trade" in o.comment:
                logger.info(f"Cancelling order {o.order_id}...")
                trader.cancel_order(o.order_id)
        
        time.sleep(2) # Allow time for cleanup
        trader.update() # Refresh state


        # 1. Add a Pending Trade (Limit Order)
        logger.info("\n--- 1. Adding a Pending Trade ---")
        current_price = broker.get_current_price(symbol, "ask")
        limit_price = round(current_price * 0.999, 5) # 0.1% below current price
        
        limit_order = trader.create_pending_order(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            direction=OrderDirection.BUY,
            volume=0.01,
            price=limit_price,
            comment="test trade pending"
        )
        if limit_order:
            logger.info(f"Pending limit order created: {limit_order.order_id}")
            time.sleep(1)

        

        # 2. Modify a Pending Trade
        logger.info("\n--- 2. Modifying a Pending Trade ---")
        if limit_order:
            new_price = round(limit_price * 1.0005, 5)
            success = trader.modify_order(limit_order.order_id, price=new_price)
            if success:
                logger.info(f"Pending order {limit_order.order_id} modified. New price: {new_price}")
            time.sleep(1)

        

        # 3. Delete a Pending Trade
        logger.info("\n--- 3. Deleting a Pending Trade ---")
        if limit_order:
            success = trader.cancel_order(limit_order.order_id)
            if success:
                logger.info(f"Pending order {limit_order.order_id} cancelled.")
            time.sleep(1)


        # 4. Enter a Market Trade
        logger.info("\n--- 4. Entering a Market Trade ---")
        market_order = trader.create_market_order(
            symbol=symbol,
            direction=OrderDirection.BUY,
            volume=0.01,
            comment="test trade market"
        )
        if market_order:
            logger.info(f"Market order sent. Broker Order ID: {market_order.order_id}")
            time.sleep(2) # Wait for order to become a position
            trader.update()
            
            
            position_id = market_order.order_id # In MT5, position ticket can be the same as order ticket
            
            if position_id in trader.open_positions:
                logger.info(f"Position {position_id} opened successfully.")
                
                # 5. Modify the Open Position
                logger.info("\n--- 5. Modifying an Open Position ---")
                
                position_to_modify = trader.open_positions[position_id]
                entry_price = position_to_modify.entry_price
                
                # Correctly calculate SL based on position direction
                if position_to_modify.direction == PositionDirection.LONG:
                    pos_sl = round(entry_price * 0.999, 5) # 0.1% below entry
                    logger.info(f"Calculating SL for LONG position: {entry_price} -> {pos_sl}")
                else:  # SHORT position
                    pos_sl = round(entry_price * 1.001, 5) # 0.1% above entry
                    logger.info(f"Calculating SL for SHORT position: {entry_price} -> {pos_sl}")
                
                success = trader.modify_position(position_id, stop_loss=pos_sl)
                if success:
                    logger.info(f"Position {position_id} modified. New SL: {pos_sl}")
                time.sleep(1)
                
                # 6. Close the Position
                logger.info("\n--- 6. Closing the Position ---")
                success = trader.close_position(position_id, volume=0.01)
                if success:
                    logger.info(f"Close order sent for position {position_id}.")

    except Exception as e:
        logger.exception(f"An error occurred during the trade execution example: {e}")
    finally:
        if 'mt5_client' in locals() and mt5_client.is_connected():
            mt5_client.shutdown()
        logger.info("--- Script Finished ---")


if __name__ == "__main__":
    main() 