from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeRisk:
    stop_loss_pips: float
    risk_per_trade_pct: float # as a percentage of account balance
    position_size_lots: float
    estimated_loss: float

class RiskManager:
    """
    Manages trade risk, including position sizing.
    """
    def __init__(self, account_balance: float, risk_percentage: float, min_trade_volume_lots: float = 0.01, lot_step: float = 0.01):
        if not 0 < risk_percentage <= 100:
            raise ValueError("Risk percentage must be between 0 and 100.")
        self.account_balance = account_balance
        self.risk_percentage = risk_percentage
        self.min_trade_volume_lots = min_trade_volume_lots
        self.lot_step = lot_step

    def update_account_balance(self, new_balance: float):
        self.account_balance = new_balance

    def calculate_position_size(
        self,
        stop_loss_pips: float,
        pip_value_per_lot: float # e.g., $10 for EURUSD on a standard account
    ) -> Optional[TradeRisk]:
        """
        Calculates the position size in lots based on risk parameters.

        Args:
            stop_loss_pips: The stop loss distance in pips.
            pip_value_per_lot: The value of one pip per standard lot for the given symbol.

        Returns:
            A TradeRisk object with calculated values, or None if trade cannot be sized.
        """
        if stop_loss_pips <= 0:
            return None

        risk_per_trade_amount = self.account_balance * (self.risk_percentage / 100.0)
        
        risk_per_lot = stop_loss_pips * pip_value_per_lot
        
        if risk_per_lot <= 0:
            return None

        position_size_lots = risk_per_trade_amount / risk_per_lot

        # Round down to the nearest lot step
        if self.lot_step > 0:
            position_size_lots = (position_size_lots // self.lot_step) * self.lot_step

        if position_size_lots < self.min_trade_volume_lots:
            return None

        estimated_loss = position_size_lots * risk_per_lot

        return TradeRisk(
            stop_loss_pips=stop_loss_pips,
            risk_per_trade_pct=self.risk_percentage,
            position_size_lots=position_size_lots,
            estimated_loss=estimated_loss
        ) 