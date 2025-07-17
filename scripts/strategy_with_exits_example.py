"""
Example script demonstrating how to use strategies with exit strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.strategy.strategy_with_exits import (
    create_strategy_with_fixed_exits,
    create_strategy_with_atr_exits,
    create_strategy_with_trailing_stops
)
from app.data.mt5_client import MT5Client
from app.trading.trader import Trader
from app.trading.broker import Broker
from app.trading.risk_manager import RiskManager
from app.trading.order import OrderDirection
from app.util.logger import get_logger

logger = get_logger(__name__)

def example_fixed_pip_strategy():
    """Example using strategy with fixed pip exits."""
    logger.info("=== Fixed Pip Exit Strategy Example ===")
    
    # Initialize components
    mt5_client = MT5Client()
    broker = Broker(mt5_client)
    risk_manager = RiskManager(mt5_client)
    trader = Trader(broker, risk_manager)
    
    # Create strategy with fixed pip exits
    strategy = create_strategy_with_fixed_exits()
    
    # Get market data
    symbol = "EURUSD"
    data = mt5_client.fetch_data(symbol, timeframe="H1", start_pos=100, end_pos=0)
    
    if data is None or len(data) == 0:
        logger.error("No data available")
        return
    
    # Generate signals
    signals = strategy.get_signals(data)
    latest_signal = signals['signal'].iloc[-1]
    
    if latest_signal != 0:
        # Get current price
        current_price = data['Close'].iloc[-1]
        symbol_info = mt5_client.get_symbol_info(symbol)
        
        # Get exit levels from strategy
        direction = "BUY" if latest_signal == 1 else "SELL"
        stop_loss, take_profit = strategy.get_exit_levels(
            data, current_price, direction, symbol_info
        )
        
        logger.info(f"Signal: {direction} at {current_price}")
        logger.info(f"Stop Loss: {stop_loss}")
        logger.info(f"Take Profit: {take_profit}")
        
        # Calculate position size
        if stop_loss is not None:
            stop_loss_pips = abs(current_price - stop_loss) / symbol_info.get('point', 0.00001) / 10
            position_size = risk_manager.calculate_position_size(stop_loss_pips, symbol_info)
            
            if position_size > 0:
                # Create order
                order_direction = OrderDirection.BUY if latest_signal == 1 else OrderDirection.SELL
                order = trader.create_market_order(
                    symbol=symbol,
                    direction=order_direction,
                    volume=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment="Fixed pip exit strategy"
                )
                
                if order:
                    logger.info(f"Order created: {order}")
                else:
                    logger.error("Failed to create order")
            else:
                logger.warning("Position size too small or risk too high")

def example_atr_strategy():
    """Example using strategy with ATR-based exits."""
    logger.info("=== ATR Exit Strategy Example ===")
    
    # Initialize components
    mt5_client = MT5Client()
    broker = Broker(mt5_client)
    risk_manager = RiskManager(mt5_client)
    trader = Trader(broker, risk_manager)
    
    # Create strategy with ATR exits
    strategy = create_strategy_with_atr_exits()
    
    # Get market data
    symbol = "GBPUSD"
    data = mt5_client.fetch_data(symbol, timeframe="H4", start_pos=100, end_pos=0)
    
    if data is None or len(data) == 0:
        logger.error("No data available")
        return
    
    # Generate signals
    signals = strategy.get_signals(data)
    latest_signal = signals['signal'].iloc[-1]
    
    if latest_signal != 0:
        current_price = data['Close'].iloc[-1]
        symbol_info = mt5_client.get_symbol_info(symbol)
        
        direction = "BUY" if latest_signal == 1 else "SELL"
        stop_loss, take_profit = strategy.get_exit_levels(
            data, current_price, direction, symbol_info
        )
        
        logger.info(f"ATR-based exit levels for {direction} at {current_price}")
        logger.info(f"Stop Loss: {stop_loss}")
        logger.info(f"Take Profit: {take_profit}")
        
        # The ATR strategy provides dynamic levels based on market volatility
        if stop_loss is not None:
            stop_loss_pips = abs(current_price - stop_loss) / symbol_info.get('point', 0.00001) / 10
            position_size = risk_manager.calculate_position_size(stop_loss_pips, symbol_info)
            
            logger.info(f"ATR-based position size: {position_size} lots")

def example_trailing_stop_strategy():
    """Example using strategy with trailing stops."""
    logger.info("=== Trailing Stop Strategy Example ===")
    
    # Initialize components
    mt5_client = MT5Client()
    broker = Broker(mt5_client)
    risk_manager = RiskManager(mt5_client)
    trader = Trader(broker, risk_manager)
    
    # Create strategy with trailing stops
    strategy = create_strategy_with_trailing_stops()
    
    # Get market data
    symbol = "USDJPY"
    data = mt5_client.fetch_data(symbol, timeframe="H1", start_pos=100, end_pos=0)
    
    if data is None or len(data) == 0:
        logger.error("No data available")
        return
    
    # Generate signals
    signals = strategy.get_signals(data)
    latest_signal = signals['signal'].iloc[-1]
    
    if latest_signal != 0:
        current_price = data['Close'].iloc[-1]
        symbol_info = mt5_client.get_symbol_info(symbol)
        
        direction = "BUY" if latest_signal == 1 else "SELL"
        initial_stop_loss, take_profit = strategy.get_exit_levels(
            data, current_price, direction, symbol_info
        )
        
        logger.info(f"Trailing stop strategy for {direction} at {current_price}")
        logger.info(f"Initial Stop Loss: {initial_stop_loss}")
        logger.info(f"Take Profit: {take_profit}")
        
        # Simulate trailing stop updates
        if initial_stop_loss is not None:
            # Simulate price movement
            if direction == "BUY":
                new_price = current_price + 0.0050  # Price moved up
            else:
                new_price = current_price - 0.0050  # Price moved down
            
            # Update trailing stop
            updated_stop = strategy.exit_strategy.update_trailing_stop(
                new_price, initial_stop_loss, direction, symbol_info
            )
            
            logger.info(f"Price moved to: {new_price}")
            logger.info(f"Updated trailing stop: {updated_stop}")

def main():
    """Run all examples."""
    try:
        # Test fixed pip strategy
        example_fixed_pip_strategy()
        print()
        
        # Test ATR strategy
        example_atr_strategy()
        print()
        
        # Test trailing stop strategy
        example_trailing_stop_strategy()
        
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)

if __name__ == "__main__":
    main() 