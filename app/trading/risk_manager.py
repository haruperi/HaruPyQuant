import os
import sys
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
import time
from datetime import datetime, timedelta

# Add project root to the Python path   
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.config.constants import RISK_PER_TRADE, INITIAL_CAPITAL, DEFAULT_TIMEFRAME, DEFAULT_START_CANDLE, DEFAULT_END_CANDLE, END_POS_D1, CORE_TIMEFRAME, VOLATILITY_PERIOD, CORRELATION_PERIOD, CONFIDENCE_LEVEL
from app.util.logger import get_logger
from app.data.mt5_client import MT5Client
from app.strategy.indicators import get_adr

logger = get_logger(__name__)

class RiskManager:
    """
    Manages trade risk, including position sizing.
    """
    def __init__(self, mt5_client: MT5Client, account_balance: float = INITIAL_CAPITAL, risk_percentage: float = RISK_PER_TRADE, start_pos: int = DEFAULT_START_CANDLE, end_pos: int = DEFAULT_END_CANDLE, timeframe: str = CORE_TIMEFRAME, input_date: str = None):
        self.start_pos = start_pos
        self.end_pos = end_pos 
        self.input_date = input_date 
        if input_date:
            self.end_date = datetime.strptime(input_date, "%Y-%m-%d")
            self.start_date = self.end_date - timedelta(days=END_POS_D1)
        else:
            self.start_date = None
            self.end_date = None

        self.timeframe = timeframe
        self.mt5_client = mt5_client
        self.positions = {}
        

        if not 0 < risk_percentage <= 100:
            raise ValueError("Risk percentage must be between 0 and 100.")
        self.account_balance = account_balance
        self.risk_percentage = risk_percentage

    def get_data(self, symbols: list[str], exclude_current_bar: bool = True) -> dict:
        """
        Fetches historical data from MetaTrader 5 for given symbols within a specified date range and timeframe.
        
        Args:
            symbols: List of symbols to fetch data for
            exclude_current_bar: If True, excludes the current incomplete bar to avoid look-ahead bias
            
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        all_data = {}
        for symbol in symbols:
            if self.input_date is None:
                df = self.mt5_client.fetch_data(symbol, timeframe=self.timeframe, start_pos=self.start_pos, end_pos=self.end_pos)
            else:
                df = self.mt5_client.fetch_data(symbol, timeframe=self.timeframe, start_date=self.start_date, end_date=self.end_date)

            if df is not None:
                # Remove the last bar if it's the current incomplete bar to avoid look-ahead bias
                if exclude_current_bar and len(df) > 1:
                    # Check if the last bar is from today (current incomplete bar)
                    last_bar_time = df.index[-1]
                    current_time = pd.Timestamp.now(tz='UTC')
                    
                    # For D1 timeframe, if last bar is from today, remove it
                    if self.timeframe == "D1":
                        if last_bar_time.date() == current_time.date():
                            df = df.iloc[:-1]  # Remove the last (incomplete) bar
                            logger.info(f"Removed current incomplete D1 bar for {symbol} to avoid look-ahead bias")
                    
                    # For other timeframes, remove if the bar is less than 1 timeframe old
                    else:
                        timeframe_minutes = {
                            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                            "H1": 60, "H4": 240
                        }
                        if self.timeframe in timeframe_minutes:
                            minutes_old = (current_time - last_bar_time).total_seconds() / 60
                            if minutes_old < timeframe_minutes[self.timeframe]:
                                df = df.iloc[:-1]  # Remove the last (incomplete) bar
                                logger.info(f"Removed current incomplete {self.timeframe} bar for {symbol} to avoid look-ahead bias")
                
                logger.info(f"Last bar for {symbol}: {df.index[-1]}")
                all_data[symbol] = df
        return all_data
    
    def get_positions(self):
        """
        Returns the current portfolio positions.    
        """
        return self.positions
    
    def get_current_positions(self):
        """
        Returns the current positions.
        """
        return self.mt5_client.get_positions()
    
    def add_position(self, symbol: str, lot_size: float):
        """
        Adds a position to the portfolio.
        """
        if symbol in self.positions:
            self.positions[symbol] += lot_size
            if self.positions[symbol] == 0:
                self.remove_position(symbol)
        else:
            self.positions[symbol] = lot_size
    
    def remove_position(self, symbol: str):
        """
        Removes a position from the portfolio.
        """
        if symbol in self.positions:
            del self.positions[symbol]

    def get_symbol_info(self, symbol: str):
        """
        Returns the symbol information.
        """
        return self.mt5_client.get_symbol_info(symbol)

    def update_account_balance(self, new_balance: float):
        """
        Updates the account balance.
        """
        self.account_balance = new_balance

    def calculate_position_size(
        self,
        stop_loss_pips: float,
        symbol_info: dict) -> float:
        """
        Calculates the position size in lots based on risk parameters.

        Args:
            stop_loss_pips: The stop loss distance in pips.
            pip_value_per_lot: The value of one pip per standard lot for the given symbol.

        Returns:
            A TradeRisk object with calculated values, or None if trade cannot be sized.
        """
        

        if stop_loss_pips <= 0:
            logger.error(f"Stop loss pips must be greater than 0. Stop loss pips: {stop_loss_pips}")
            return 0

        # Calculate the amount to risk
        risk_per_trade_amount = self.account_balance * self.risk_percentage 

        # Get the tick value and convert stop loss to money
        tick_value = symbol_info.trade_tick_value
        if tick_value <= 0:
            logger.error(f"Invalid tick value: {tick_value}")
            return 0

        # Convert stop loss pips to points
        #stop_loss_points = stop_loss_pips * 10

        # Convert stop loss pips to the symbol's base currency value
        point_size = symbol_info.point
        stop_loss_in_currency = stop_loss_pips * point_size * 10 
        risk_per_lot = stop_loss_in_currency * symbol_info.trade_tick_value / symbol_info.trade_tick_size

        position_size_lots = risk_per_trade_amount / risk_per_lot

        # Round down to the nearest valid lot step
        lot_step = symbol_info.volume_step
        position_size_lots = (position_size_lots // lot_step) * lot_step

        # Ensure minimum and maximum lot sizes
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        position_size_lots = max(min_lot, min(max_lot, position_size_lots))

        return position_size_lots

       
        # print(f"Point size: {point_size}")

        # # Calculate the position size in lots
        # position_size_lots = risk_per_trade_amount / (stop_loss_points * tick_value)

        # # Round down to the nearest lot step
        # position_size_lots = round(position_size_lots, 2)
        
        # # Ensure minimum lot size and check against maximum
        # min_lot = 0.01
        # max_lot = symbol_info.volume_max  # Use the maximum allowed by the broker
        # position_size_lots = max(min_lot, min(max_lot, position_size_lots))

        # return position_size_lots
    
    def run(self):
        """
        Runs the risk manager with stateless calculation methods and look-ahead bias protection.
        """
        # Get Data from MT5 (exclude current incomplete bar)
        symbols = list(self.get_positions().keys())
        data = self.get_data(symbols, exclude_current_bar=True)
        
        # Verify no look-ahead bias (data should already be clean from get_data)
        if not self.validate_no_look_ahead_bias(data):
            logger.error("Look-ahead bias detected despite data cleaning! This should not happen.")
            return 0.0

        # Calculate returns for all positions
        data_with_returns = self.calculate_returns(data)
        
        # Calculate volatility for all positions
        std_dev_returns = self.calculate_volatility(data_with_returns)
        
        # Calculate correlations between all positions
        correlations = self.calculate_correlations(data_with_returns)
        
        # Calculate nominal values for all positions (uses historical prices for backtesting, live for real-time)
        nominal_values, portfolio_nominal_value = self.calculate_nominal_values(self.positions, data)

        # Calculate position weights
        position_weights = self.calculate_position_weights(nominal_values, portfolio_nominal_value)

        # Calculate portfolio standard deviation
        portfolio_std_dev = self.calculate_portfolio_std_dev(position_weights, std_dev_returns, correlations)
        
        # Calculate Value at Risk
        portfolio_var = self.calculate_var(portfolio_std_dev, portfolio_nominal_value)

        return portfolio_var
    
    def calculate_returns(self, data: dict) -> dict:
        """
        Method to calculate returns from the fetched data
        
        Args:
            data: Dictionary of DataFrames with OHLC data
            
        Returns:
            dict: Dictionary of DataFrames with added log_returns column
        """
        for symbol, df in data.items():
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            logger.info(f"Returns for {symbol}: {df['log_returns'].iloc[-1]}")
        return data
    
    def calculate_volatility(self, data: dict) -> dict:
        """
        Method to calculate volatility from the returns data
        
        Args:
            data: Dictionary of DataFrames with log_returns column (already excludes current incomplete bar)
            
        Returns:
            dict: Dictionary with symbol as key and latest volatility as value
        """
        std_dev_returns = {}
        for symbol, df in data.items():
            if 'log_returns' not in df.columns:
                logger.error(f"Log returns have not been calculated for {symbol}")
                continue
            
            # Calculate rolling volatility (standard deviation of returns)
            # shift(1) ensures volatility at time T is based on returns from T-1 and earlier (no look-ahead)
            df['volatility'] = df['log_returns'].shift(1).rolling(window=VOLATILITY_PERIOD).std()
            
            # Use the last non-NaN value since data already excludes current incomplete bar
            last_valid_volatility = df['volatility'].dropna().iloc[-1] if not df['volatility'].dropna().empty else 0.0
            std_dev_returns[symbol] = last_valid_volatility
            logger.info(f"Volatility for {symbol}: {std_dev_returns[symbol]:.6f}")
        return std_dev_returns
    
    def calculate_correlations(self, data: dict) -> pd.DataFrame:
        """
        Method to calculate rolling correlation matrix based on the last N days
        
        Args:
            data: Dictionary of DataFrames with log_returns column (already excludes current incomplete bar)
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if len(data) < 2:
            logger.warning("Need at least 2 symbols to calculate correlations")
            return pd.DataFrame()
        
        # Get all symbols
        symbols = list(data.keys())
        
        # Create a DataFrame with log returns for all symbols
        returns_df = pd.DataFrame()
        
        for symbol in symbols:
            if 'log_returns' not in data[symbol].columns:
                logger.error(f"Log returns have not been calculated for {symbol}")
                continue
            returns_df[symbol] = data[symbol]['log_returns']
        
        if returns_df.empty:
            logger.error("No valid returns data available for correlation calculation")
            return pd.DataFrame()
        
        # Calculate rolling correlation matrix
        try:
            # Calculate rolling correlation matrix using the last CORRELATION_PERIOD days
            rolling_corr = returns_df.rolling(window=CORRELATION_PERIOD).corr()
            
            # Get the most recent correlation matrix (data already excludes current incomplete bar)
            if not rolling_corr.empty:
                # Get the last level of the multi-index since data is already clean
                timestamps = rolling_corr.index.get_level_values(0).unique()
                target_timestamp = timestamps[-1]
                correlation_matrix = rolling_corr.loc[target_timestamp]
                logger.info(f"Correlation matrix for {len(symbols)} symbols using {CORRELATION_PERIOD}-day window")
                print(correlation_matrix)
                
                return correlation_matrix
            else:
                logger.warning("No data available for rolling correlation calculation")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {str(e)}")
            return pd.DataFrame()
    
    def calculate_nominal_values(self, positions: dict, historical_data: dict) -> tuple[dict, float]:
        """
        Method to calculate the nominal value of each pair and the total portfolio nominal value
        Always uses the last close price from historical data for consistency between live and backtest.
        
        Args:
            positions: Dictionary with symbol as key and lot size as value
            historical_data: Dictionary of historical DataFrames (required for both live and backtest)
            
        Returns:
            tuple: (nominal_values_dict, total_portfolio_nominal_value)
        """
        nominal_values = {}
        portfolio_nominal_value = 0.0
        
        for symbol, lot_size in positions.items():
            try:
                # Get symbol info for contract size
                symbol_info = self.mt5_client.get_symbol_info(symbol)
                
                if symbol_info is None:
                    logger.error(f"Could not get symbol info for {symbol}")
                    continue
                
                # Always use the last close price from historical data for consistency
                if symbol not in historical_data:
                    logger.error(f"No historical data available for {symbol}")
                    continue
                    
                df = historical_data[symbol]
                if df.empty:
                    logger.error(f"Empty historical data for {symbol}")
                    continue
                    
                current_price = df['Close'].iloc[-1]  # Always use last historical close price
                logger.debug(f"Using last close price for {symbol}: {current_price:.5f}")
                
                # Calculate nominal value
                nominal_value_per_unit_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                nominal_value = lot_size * nominal_value_per_unit_per_lot * current_price
                
                nominal_values[symbol] = nominal_value
                portfolio_nominal_value += abs(nominal_value)
                
                logger.info(f"Nominal value for {symbol}: ${nominal_value:,.2f} "
                           f"(Lot size: {lot_size}, Price: {current_price:.5f})")
                
            except Exception as e:
                logger.error(f"Error calculating nominal value for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Total portfolio nominal value: ${portfolio_nominal_value:,.2f}")
        return nominal_values, portfolio_nominal_value

    def calculate_position_weights(self, nominal_values: dict, portfolio_nominal_value: float) -> dict:
        """
        Calculates the weight of each position in the portfolio based on nominal values.
        
        Args:
            nominal_values: Dictionary with symbol as key and nominal value as value
            portfolio_nominal_value: Total portfolio nominal value
            
        Returns:
            dict: Dictionary with symbol as key and weight as value
        """
        position_weights = {}
        if portfolio_nominal_value == 0:
            logger.warning("Total portfolio nominal value is zero. Cannot calculate weights.")
            return position_weights
        for symbol, nominal_value in nominal_values.items():
            weight = nominal_value / portfolio_nominal_value
            position_weights[symbol] = weight
            logger.info(f"Weight for {symbol}: {weight:.4%}")
        return position_weights

    def calculate_portfolio_std_dev(self, position_weights: dict, std_dev_returns: dict, correlations: pd.DataFrame) -> float:
        """
        Calculates the portfolio standard deviation (volatility) using position weights, std devs, and correlation matrix.
        
        Args:
            position_weights: Dictionary with symbol as key and weight as value
            std_dev_returns: Dictionary with symbol as key and std dev as value
            correlations: Correlation matrix DataFrame
            
        Returns:
            float: Portfolio standard deviation
        """
        symbols = list(position_weights.keys())
        n = len(symbols)
        if n == 0:
            logger.warning("No positions to calculate portfolio standard deviation.")
            return 0.0
        
        # Build weights vector
        weights = np.array([position_weights[s] for s in symbols])
        # Build std dev vector
        std_devs = np.array([std_dev_returns.get(s, 0.0) for s in symbols])
        # Build correlation matrix
        if isinstance(correlations, pd.DataFrame) and not correlations.empty:
            corr_matrix = correlations.reindex(index=symbols, columns=symbols).fillna(0).values
        else:
            corr_matrix = np.eye(n)
        # Covariance matrix
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
        # Portfolio variance
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std_dev = np.sqrt(port_var)
        logger.info(f"Portfolio standard deviation: {portfolio_std_dev:.6f}")
        return portfolio_std_dev
    
    def calculate_var(self, portfolio_std_dev: float, portfolio_nominal_value: float, confidence_level: float = CONFIDENCE_LEVEL, time_horizon: int = 1) -> float:
        """
        Calculates Value at Risk (VaR) using portfolio standard deviation and specified confidence level.
        
        Args:
            portfolio_std_dev: Portfolio standard deviation
            portfolio_nominal_value: Total portfolio nominal value
            confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95%)
            time_horizon (int): Time horizon in days for VaR calculation (default: 1 day)
            
        Returns:
            float: Portfolio VaR value
        """
        
        if portfolio_nominal_value == 0:
            logger.warning("Portfolio nominal value is zero. Cannot calculate VaR.")
            return 0.0
        
        if portfolio_std_dev == 0:
            logger.warning("Portfolio standard deviation is zero. Cannot calculate VaR.")
            return 0.0
        
        try:
            # Calculate the z-score for the confidence level
            z_score = stats.norm.ppf(confidence_level)
            
            # Calculate VaR: VaR = z_score * portfolio_std_dev * sqrt(time_horizon) * portfolio_value
            portfolio_var = z_score * portfolio_std_dev * np.sqrt(time_horizon) * portfolio_nominal_value
            
            logger.info(f"Portfolio VaR ({confidence_level:.0%} confidence, {time_horizon} day): ${portfolio_var:,.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            portfolio_var = 0.0
        
        return portfolio_var

    def validate_no_look_ahead_bias(self, data: dict) -> bool:
        """
        Validates that the data doesn't contain future information (look-ahead bias check).
        Since get_data() already excludes current incomplete bars, this is mainly for verification.
        
        Args:
            data: Dictionary of DataFrames with historical data (should already be clean)
            
        Returns:
            bool: True if no look-ahead bias detected, False otherwise
        """
        current_time = pd.Timestamp.now(tz='UTC')
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Check if the last bar timestamp is in the future (should never happen)
            last_bar_time = df.index[-1]
            if last_bar_time > current_time:
                logger.error(f"CRITICAL ERROR: {symbol} has future data: {last_bar_time} > {current_time}")
                return False
            
            # Verify that we're not using today's incomplete D1 bar
            if self.timeframe == "D1":
                if last_bar_time.date() == current_time.date():
                    logger.error(f"LOOK-AHEAD BIAS: {symbol} still has today's D1 bar: {last_bar_time}")
                    return False
        
        logger.debug("Data validation passed - no look-ahead bias detected")
        return True

    def analyze_hypothetical_portfolio(self, hypothetical_positions: dict) -> dict:
        """
        Analyzes a hypothetical portfolio without affecting the current state.
        This demonstrates the benefit of the stateless approach.
        
        Args:
            hypothetical_positions: Dictionary with symbol as key and lot size as value
            
        Returns:
            dict: Analysis results including VaR, weights, nominal values, etc.
        """
        # Get Data from MT5 for the hypothetical positions (exclude current incomplete bar)
        symbols = list(hypothetical_positions.keys())
        data = self.get_data(symbols, exclude_current_bar=True)
        
        # Verify no look-ahead bias (data should already be clean from get_data)
        if not self.validate_no_look_ahead_bias(data):
            logger.error("Look-ahead bias detected in hypothetical analysis despite data cleaning!")
            return {}

        # Calculate returns for all positions
        data_with_returns = self.calculate_returns(data)
        
        # Calculate volatility for all positions
        std_dev_returns = self.calculate_volatility(data_with_returns)
        
        # Calculate correlations between all positions
        correlations = self.calculate_correlations(data_with_returns)
        
        # Calculate nominal values for all positions (uses historical prices for backtesting, live for real-time)
        nominal_values, portfolio_nominal_value = self.calculate_nominal_values(hypothetical_positions, data)

        # Calculate position weights
        position_weights = self.calculate_position_weights(nominal_values, portfolio_nominal_value)

        # Calculate portfolio standard deviation
        portfolio_std_dev = self.calculate_portfolio_std_dev(position_weights, std_dev_returns, correlations)
        
        # Calculate Value at Risk
        portfolio_var = self.calculate_var(portfolio_std_dev, portfolio_nominal_value)

        return {
            'var': portfolio_var,
            'portfolio_std_dev': portfolio_std_dev,
            'portfolio_nominal_value': portfolio_nominal_value,
            'position_weights': position_weights,
            'nominal_values': nominal_values,
            'std_dev_returns': std_dev_returns,
            'correlations': correlations
        }

def test_risk_manager():
    """
    Tests the risk manager.
    """

    # Set specific date
    custom_date = None  # None for default date range
    #custom_date = "2023-07-03"
    

    # Initialize the MT5 client
    mt5_client = MT5Client()

    # Initialize the risk manager
    portfolio = RiskManager(mt5_client=mt5_client, input_date=custom_date)

    def add_position_to_portfolio(symbol, action, date_input: str = None):
        symbol_info = mt5_client.get_symbol_info(symbol)
        
        if date_input:
            end_day = datetime.strptime(date_input, "%Y-%m-%d")
            start_day = end_day - timedelta(days=END_POS_D1)
            d1_df = mt5_client.fetch_data(symbol, CORE_TIMEFRAME, start_date=start_day, end_date=end_day)
        else:
            d1_df = mt5_client.fetch_data(symbol, CORE_TIMEFRAME, start_pos=DEFAULT_START_CANDLE, end_pos=END_POS_D1)
        
        if d1_df is None or d1_df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return
            
        current_adr, current_daily_range_percentage, stop_loss = get_adr(d1_df, symbol_info)
        lots = portfolio.calculate_position_size(stop_loss, symbol_info)
        lots = lots if action == "Buy" else -lots
        portfolio.add_position(symbol, lots)
        logger.info(f"Added {lots} lots of {symbol}")
        logger.info(f"Stop loss: {stop_loss}")
        logger.info(f"Current ADR: {current_adr}")

    add_position_to_portfolio('GBPUSD', "Sell", custom_date)
    add_position_to_portfolio('USDJPY', "Sell", custom_date)

    curr_value_at_risk = portfolio.run()

    add_position_to_portfolio('GBPJPY', "Buy", custom_date)

    proposed_value_at_risk = portfolio.run()

    incr_var = ((proposed_value_at_risk - curr_value_at_risk) / curr_value_at_risk) * 100

    print(f"Current VaR: ${round(curr_value_at_risk):,.2f}" )
    print(f"Proposed VaR: ${round(proposed_value_at_risk):,.2f}")
    print(f"Increase in VaR: {round(incr_var)}%")

    # # Demonstrate the stateless approach with hypothetical portfolio analysis
    # print("\n" + "="*50)
    # print("DEMONSTRATING STATELESS APPROACH")
    # print("="*50)
    
    # # Create a hypothetical portfolio with different positions
    # hypothetical_positions = {
    #     'USDJPY': -0.05,  # Smaller position
    #     'GBPUSD': 0.08,   # New position
    #     'EURUSD': 0.06    # Another new position
    # }
    
    # # Analyze the hypothetical portfolio without affecting current state
    # hypothetical_analysis = portfolio.analyze_hypothetical_portfolio(hypothetical_positions)
    
    # print(f"Hypothetical Portfolio Analysis:")
    # print(f"VaR: ${round(hypothetical_analysis['var']):,.2f}")
    # print(f"Portfolio Std Dev: {hypothetical_analysis['portfolio_std_dev']:.6f}")
    # print(f"Portfolio Nominal Value: ${hypothetical_analysis['portfolio_nominal_value']:,.2f}")
    # print(f"Position Weights: {hypothetical_analysis['position_weights']}")
    
    # # Verify that the original portfolio state is unchanged
    # print(f"\nOriginal portfolio still has {len(portfolio.get_positions())} positions: {portfolio.get_positions()}")
    
    # # Compare with current portfolio
    # current_analysis = portfolio.analyze_hypothetical_portfolio(portfolio.get_positions())
    # print(f"Current portfolio VaR: ${round(current_analysis['var']):,.2f}")
    # print(f"Hypothetical portfolio VaR: ${round(hypothetical_analysis['var']):,.2f}")
    # print(f"VaR difference: ${round(hypothetical_analysis['var'] - current_analysis['var']):,.2f}")


    
    # portfolio.run()
    # print(portfolio.correlations)
    # print(f"\nNominal Values: {portfolio.nominal_values}")
    # print(f"Total Portfolio Nominal Value: ${portfolio.portfolio_nominal_value:,.2f}")
    # print(f"Position Weights: {portfolio.position_weights}")
    # print(f"Portfolio Standard Deviation: {portfolio.portfolio_std_dev:.6f}")
    # print(f"Portfolio VaR ({CONFIDENCE_LEVEL:.0%} confidence, 1 day): ${portfolio.portfolio_var:,.2f}")

if __name__ == "__main__":

    test_risk_manager()
 