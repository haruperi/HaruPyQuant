"""
MT5 Broker Implementation

This module provides a concrete implementation of the Broker interface for MetaTrader 5.
"""

import MetaTrader5 as mt5
from typing import List, Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from app.trading.broker import Broker
from app.trading.order import Order, OrderType, OrderDirection, OrderStatus
from app.trading.position import Position, PositionDirection
from app.data.mt5_client import MT5Client
from app.util import get_logger

logger = get_logger(__name__)

# Timezone for MT5 timestamps
UTC = ZoneInfo("UTC")


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

    def get_account_info(self) -> Dict[str, Any]:
        """Get comprehensive account information."""
        info = self.client.get_account_info()
        if info:
            return {
                'balance': info.get('balance', 0.0),
                'equity': info.get('equity', 0.0),
                'margin': info.get('margin', 0.0),
                'free_margin': info.get('free_margin', 0.0),
                'profit': info.get('profit', 0.0),
                'currency': info.get('currency', 'USD')
            }
        return {
            'balance': 0.0,
            'equity': 0.0,
            'margin': 0.0,
            'free_margin': 0.0,
            'profit': 0.0,
            'currency': 'USD'
        }

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
        filling_type = mt5.ORDER_FILLING_FOK  # Default to Fill or Kill
        
        if allowed_filling_modes & mt5.ORDER_FILLING_IOC:
            filling_type = mt5.ORDER_FILLING_IOC
        elif allowed_filling_modes & mt5.ORDER_FILLING_FOK:
            filling_type = mt5.ORDER_FILLING_FOK

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5_order_type,
            "price": order.price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        logger.info(f"Order sent successfully: {result.order}")
        return str(result.order)

    def cancel_order(self, order_id: str) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def close_position(self, position_id: str, amount: float) -> bool:
        # MT5 requires creating an opposing order to close a position.
        positions = self.get_open_positions()
        position = next((p for p in positions if p.order_id == position_id), None)
        
        if not position:
            logger.error(f"Position {position_id} not found")
            return False

        # Create opposing order
        opposite_direction = OrderDirection.SELL if position.direction == PositionDirection.LONG else OrderDirection.BUY
        
        order = Order(
            order_id="",  # Will be assigned by broker
            symbol=position.symbol,
            order_type=OrderType.MARKET,
            direction=opposite_direction,
            volume=amount,
            price=0.0,  # Market order
            stop_loss=None,
            take_profit=None,
            status=OrderStatus.PENDING,
            comment=f"Close position {position_id}"
        )

        result_id = self.send_order(order)
        return result_id is not None

    def modify_position(self, position_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(position_id),
        }
        
        if stop_loss is not None:
            request["sl"] = stop_loss
        if take_profit is not None:
            request["tp"] = take_profit

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def modify_order(self, order_id: str, price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": int(order_id),
        }
        
        if price is not None:
            request["price"] = price
        if stop_loss is not None:
            request["sl"] = stop_loss
        if take_profit is not None:
            request["tp"] = take_profit

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get_cached_symbol_info(symbol)

    def get_current_price(self, symbol: str, price_type: str = 'ask') -> float:
        tick = self.client.get_tick(symbol)
        if tick:
            return tick['ask'] if price_type == 'ask' else tick['bid']
        return 0.0 