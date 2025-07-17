from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from app.util.logger import get_logger

logger = get_logger(__name__)

class ExitStrategy(ABC):
    """
    Abstract base class for exit strategies (Stop Loss and Take Profit).
    
    This class handles the logic for determining when and how to exit positions,
    separate from the entry signal generation in BaseStrategy.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the exit strategy with parameters.
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters including SL/TP settings
        """
        self.parameters = parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validates that required parameters are present and valid."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Required parameter '{param}' not found in exit strategy parameters")
    
    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """
        Returns a list of required parameter names for this exit strategy.
        
        Returns:
            list[str]: List of required parameter names
        """
        pass
    
    @abstractmethod
    def calculate_exit_levels(
        self, 
        data: pd.DataFrame, 
        entry_price: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates stop loss and take profit levels.
        
        Args:
            data (pd.DataFrame): Market data (OHLCV)
            entry_price (float): Entry price for the position
            direction (str): 'BUY' or 'SELL'
            symbol_info (Dict[str, Any]): Symbol information from broker
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (stop_loss, take_profit) levels
        """
        pass
    
    def should_exit(
        self, 
        data: pd.DataFrame, 
        position_info: Dict[str, Any]
    ) -> bool:
        """
        Determines if a position should be exited based on current market conditions.
        
        Args:
            data (pd.DataFrame): Current market data
            position_info (Dict[str, Any]): Current position information
            
        Returns:
            bool: True if position should be exited, False otherwise
        """
        return False


class FixedPipExitStrategy(ExitStrategy):
    """
    Simple exit strategy using fixed pip distances for SL/TP.
    """
    
    def get_required_parameters(self) -> list[str]:
        return ['stop_loss_pips', 'take_profit_pips']
    
    def calculate_exit_levels(
        self, 
        data: pd.DataFrame, 
        entry_price: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates fixed pip-based SL/TP levels.
        """
        stop_loss_pips = self.parameters['stop_loss_pips']
        take_profit_pips = self.parameters['take_profit_pips']
        
        # Get pip value from symbol info
        point = symbol_info.get('point', 0.00001)  # Default for most forex pairs
        pip_value = point * 10  # 1 pip = 10 points for most forex pairs
        
        if direction.upper() == 'BUY':
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
        else:  # SELL
            stop_loss = entry_price + (stop_loss_pips * pip_value)
            take_profit = entry_price - (take_profit_pips * pip_value)
        
        logger.info(f"Calculated exit levels for {direction} at {entry_price}: SL={stop_loss}, TP={take_profit}")
        return stop_loss, take_profit


class ATRExitStrategy(ExitStrategy):
    """
    Exit strategy using Average True Range (ATR) for dynamic SL/TP levels.
    """
    
    def get_required_parameters(self) -> list[str]:
        return ['atr_period', 'sl_atr_multiplier', 'tp_atr_multiplier']
    
    def calculate_exit_levels(
        self, 
        data: pd.DataFrame, 
        entry_price: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates ATR-based SL/TP levels.
        """
        from app.strategy.indicators import calculate_atr
        
        atr_period = self.parameters['atr_period']
        sl_multiplier = self.parameters['sl_atr_multiplier']
        tp_multiplier = self.parameters['tp_atr_multiplier']
        
        # Calculate ATR
        atr = calculate_atr(data, atr_period)
        if atr is None or len(atr) == 0:
            logger.warning("ATR calculation failed, using default pip values")
            return None, None
        
        current_atr = atr.iloc[-1]
        
        if direction.upper() == 'BUY':
            stop_loss = entry_price - (current_atr * sl_multiplier)
            take_profit = entry_price + (current_atr * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (current_atr * sl_multiplier)
            take_profit = entry_price - (current_atr * tp_multiplier)
        
        logger.info(f"Calculated ATR-based exit levels: SL={stop_loss}, TP={take_profit} (ATR={current_atr})")
        return stop_loss, take_profit


class TrailingStopExitStrategy(ExitStrategy):
    """
    Exit strategy with trailing stop loss functionality.
    """
    
    def get_required_parameters(self) -> list[str]:
        return ['trailing_distance_pips', 'trailing_step_pips']
    
    def calculate_exit_levels(
        self, 
        data: pd.DataFrame, 
        entry_price: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates initial trailing stop level.
        """
        trailing_distance = self.parameters['trailing_distance_pips']
        point = symbol_info.get('point', 0.00001)
        pip_value = point * 10
        
        if direction.upper() == 'BUY':
            stop_loss = entry_price - (trailing_distance * pip_value)
        else:  # SELL
            stop_loss = entry_price + (trailing_distance * pip_value)
        
        # No take profit for trailing stops
        return stop_loss, None
    
    def update_trailing_stop(
        self, 
        current_price: float, 
        current_stop_loss: float, 
        direction: str,
        symbol_info: Dict[str, Any]
    ) -> Optional[float]:
        """
        Updates trailing stop level based on current price.
        """
        trailing_step = self.parameters['trailing_step_pips']
        point = symbol_info.get('point', 0.00001)
        pip_value = point * 10
        step_distance = trailing_step * pip_value
        
        if direction.upper() == 'BUY':
            # For long positions, trail up
            new_stop = current_price - step_distance
            if new_stop > current_stop_loss:
                logger.info(f"Updating trailing stop: {current_stop_loss} -> {new_stop}")
                return new_stop
        else:  # SELL
            # For short positions, trail down
            new_stop = current_price + step_distance
            if new_stop < current_stop_loss:
                logger.info(f"Updating trailing stop: {current_stop_loss} -> {new_stop}")
                return new_stop
        
        return current_stop_loss 