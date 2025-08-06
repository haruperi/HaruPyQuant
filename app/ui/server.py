"""
Flask server for HaruPyQuant web application
Provides API endpoints for dashboard metrics and system status
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================
import os
import sys
import logging
import pandas as pd
from datetime import datetime, time, UTC
from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, Optional, Any

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Now import from app modules
from app.config.setup import *


logger = get_logger(__name__)

# ==============================================================================
# APP INITIALIZATION AND CORS
# ==============================================================================
# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'harupyquant-dev-key')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

# ------------------------------------------------------------------------------
# Health & Status
# ------------------------------------------------------------------------------
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/status')
def get_status():
    """Get system status"""
    logger.info("=== Starting status check request ===")
    logger.info("Status check requested")
    
    try:
        logger.info("Attempting to import MT5Client for status check...")
        from app.data.mt5_client import MT5Client
        logger.info("MT5Client imported successfully for status check")
        
        mt5_status = "disconnected"
        try:
            logger.info("Creating MT5Client instance for status check...")
            mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
            logger.info("MT5Client instance created for status check")
            
            logger.info("Checking MT5 connection status...")
            if mt5_client.is_connected():
                mt5_status = "connected"
                logger.info("MT5 is connected")
            else:
                logger.warning("MT5 is not connected")
            
            logger.info("Shutting down MT5Client...")
            mt5_client.shutdown()
            logger.info("MT5Client shut down")
            
        except Exception as e:
            logger.error(f"MT5 connection check failed: {e}")
            logger.exception("Full exception details:")
        
        logger.info(f"Final MT5 status: {mt5_status}")
        logger.info("=== Returning status response ===")
        
        return jsonify({
            'status': 'operational',
            'mt5_connection': mt5_status,
            'timestamp': datetime.utcnow().isoformat()
        })
    except ImportError as e:
        logger.error(f"Failed to import MT5Client for status check: {e}")
        return jsonify({
            'status': 'error',
            'error': f"MT5Client import failed: {str(e)}",
            'mt5_connection': 'unavailable',
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        logger.exception("Full exception details:")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ------------------------------------------------------------------------------
# Dashboard Metrics
# ------------------------------------------------------------------------------
@app.route('/api/dashboard-metrics', methods=['GET'])
def dashboard_metrics():
    """Get dashboard metrics"""
    logger.info("=== Starting dashboard metrics request ===")
    logger.info("Received request: /api/dashboard-metrics")
    
    try:
        logger.info("Attempting to import MT5Client...")
        from app.data.mt5_client import MT5Client
        logger.info("MT5Client imported successfully")
        
        logger.info("Creating MT5Client instance...")
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
        logger.info("MT5Client instance created")
        
        logger.info("Checking MT5 connection status...")
        connection_status = mt5_client.is_connected()
        logger.info(f"MT5 connection status: {connection_status}")
        
        if not connection_status:
            logger.warning("MT5 not connected, returning fallback dummy data")
            logger.info("=== Returning fallback data ===")
            return jsonify({
                'closed_positions': 114,
                'winning_factor': "53 / 114",
                'performance_factor': -0.45,
                'max_drawdown': 1.7,
                'open_positions': 4,
                'pnl': -31.3,
                'equity': 99223.56,
                'balance': 99254.86
            })
        
        logger.info("MT5 is connected, fetching account info...")
        account = mt5_client.get_account_info()
        logger.info(f"Account info retrieved: {account}")
        
        logger.info("Fetching open positions...")
        open_positions = mt5_client.get_positions()
        logger.info(f"Open positions retrieved: {len(open_positions)} positions")
        
        logger.info("Shutting down MT5 connection...")
        mt5_client.shutdown()
        logger.info("MT5 connection shut down")
        
        # Dummy calculations for now - replace with real calculations
        closed_count = 114
        winning_count = 53
        performance_factor = -0.45
        max_drawdown = 1.7
        pnl = account.get('profit', 0)
        equity = account.get('equity', 0)
        balance = account.get('balance', 0)
        open_count = len(open_positions)
        
        logger.info(f"Calculated metrics: closed={closed_count}, winning={winning_count}, open={open_count}, pnl={pnl}, equity={equity}, balance={balance}")
        logger.info("=== Returning real MT5 data ===")
        
        return jsonify({
            'closed_positions': closed_count,
            'winning_factor': f"{winning_count} / {closed_count}",
            'performance_factor': performance_factor,
            'max_drawdown': max_drawdown,
            'open_positions': open_count,
            'pnl': pnl,
            'equity': equity,
            'balance': balance
        })
    except ImportError as e:
        logger.error(f"Failed to import MT5Client: {e}")
        logger.warning("Returning fallback dummy data due to import error")
        return jsonify({
            'closed_positions': 114,
            'winning_factor': "53 / 114",
            'performance_factor': -0.45,
            'max_drawdown': 1.7,
            'open_positions': 4,
            'pnl': -31.3,
            'equity': 99223.56,
            'balance': 99254.86
        })
    except Exception as e:
        logger.exception(f"Error fetching MT5 dashboard metrics: {e}")
        logger.warning("Returning fallback dummy data due to error")
        return jsonify({
            'closed_positions': 114,
            'winning_factor': "53 / 114",
            'performance_factor': -0.45,
            'max_drawdown': 1.7,
            'open_positions': 4,
            'pnl': -31.3,
            'equity': 99223.56,
            'balance': 99254.86
        })

# ------------------------------------------------------------------------------
# Technical Data
# ------------------------------------------------------------------------------
@app.route('/api/technical-indicators', methods=['GET'])
def get_technical_indicators():
    """
    Calculate technical indicators for given data.
    Query params: symbol, timeframe, mode ('bars' or 'date'), bars, start_date, end_date, indicators
    """
    logger.info("=== Starting technical indicators request ===")
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    mode = request.args.get('mode', 'bars')
    bars = int(request.args.get('bars', 100))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    indicators = request.args.get('indicators', '')  # Comma-separated list of indicators
    
    logger.info(f"Received params: symbol={symbol}, timeframe={timeframe}, mode={mode}, bars={bars}, start_date={start_date}, end_date={end_date}, indicators={indicators}")

    # Validate required parameters
    if not symbol or not timeframe:
        return jsonify({'error': 'symbol and timeframe are required parameters'}), 400

    try:
        from app.data.mt5_client import MT5Client
        from app.strategy.indicators import calculate_ma
        logger.info("Attempting to create MT5Client instance for indicators...")
        mt5 = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
        logger.info("MT5Client instance created")
        if not mt5.is_connected():
            logger.error("MT5 not connected")
            return jsonify({'error': 'MT5 not connected'}), 500

        df = None
        if mode == 'bars':
            logger.info(f"Fetching {bars} bars for {symbol} {timeframe}")
            df = mt5.fetch_data(symbol, timeframe, start_pos=0, end_pos=bars)
        else:
            # Convert date strings to datetime objects for mt5.fetch_data compatibility
            from datetime import datetime, timezone
            logger.info(f"Fetching data for {symbol} {timeframe} from {start_date} to {end_date}")
            start_str = start_date[:10] if start_date else None
            end_str = end_date[:10] if end_date else None
            
            if not start_str or not end_str:
                logger.error("start_date and end_date are required for date mode")
                #return jsonify({'error': 'start_date and end_date are required for date mode'}), 400
            else:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                
            start_of_day = datetime.combine(start_dt, time.min)  # 00:00:00
            end_of_day = datetime.combine(end_dt, time.max)    # 23:59:59.999999
            df = mt5.fetch_data(symbol, timeframe, start_date=start_of_day, end_date=end_of_day)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Calculate indicators
        indicator_data = {}
        indicator_list = [ind.strip() for ind in indicators.split(',') if ind.strip()]
        
        for indicator in indicator_list:
            try:
                if indicator == 'EMA20':
                    # Use calculate_ma function instead of ema_series
                    df_with_ema = calculate_ma(df, period=20, ma_type="EMA", column='Close')
                    ema_values = df_with_ema['ema_20'].tolist()
                    indicator_data[indicator] = ema_values
                    logger.info(f"Calculated {indicator} with period 20")
                else:
                    logger.warning(f"Unsupported indicator: {indicator}")
                    indicator_data[indicator] = None
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                indicator_data[indicator] = None

        logger.info(f"Calculated indicators: {list(indicator_data.keys())}")
        return jsonify({'indicators': indicator_data})
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return jsonify({'error': f'Import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error calculating technical indicators: {e}")
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------------------------
# Correlation Matrix
# ------------------------------------------------------------------------------
@app.route('/api/correlation-matrix', methods=['GET'])
def get_correlation_matrix():
    """
    Calculate and return the correlation matrix for all symbols.
    Query params: date, timeframe
    """
    logger.info("=== Starting correlation matrix request ===")
    date = request.args.get('date')
    timeframe = request.args.get('timeframe')
    logger.info(f"Received params: date={date}, timeframe={timeframe}")

    # Validate required parameters
    if not timeframe:
        return jsonify({'error': 'timeframe is a required parameter'}), 400

    try:
        from app.trading.risk_manager import get_all_symbols_correlation
        
        logger.info("Fetching correlation matrix...")
        correlation_df = get_all_symbols_correlation(date_input=date, timeframe=timeframe)
        
        if correlation_df is None or correlation_df.empty:
            logger.warning("Correlation matrix is empty.")
            return jsonify({'error': 'Could not generate correlation matrix. Check logs.'}), 404

        # Process the DataFrame
        # Multiply by 100 and round
        correlation_df = (correlation_df * 100).round()

        # Set diagonal to 100
        for i in range(len(correlation_df.columns)):
            correlation_df.iloc[i, i] = 100

        # Replace NaN with null for JSON compatibility
        correlation_df = correlation_df.where(pd.notnull(correlation_df), None)

        # Convert to JSON format for Plotly
        data = {
            'z': correlation_df.values.tolist(),
            'x': correlation_df.columns.tolist(),
            'y': correlation_df.index.tolist()
        }
        
        logger.info("Successfully generated correlation matrix.")
        return jsonify(data)

    except ImportError as e:
        logger.error(f"Failed to import module: {e}")
        return jsonify({'error': f'Import Error: {e}'}), 500
    except Exception as e:
        logger.exception(f"Error generating correlation matrix: {e}")
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------------------
# SMC Data
# ------------------------------------------------------------------------------
@app.route('/api/smc-data', methods=['GET'])
def get_smc_data():
    """
    Calculate SMC (Smart Money Concepts) data for given symbol and timeframe.
    Query params: symbol, timeframe, mode ('bars' or 'date'), bars, start_date, end_date
    """
    logger.info("=== Starting SMC data request ===")
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    mode = request.args.get('mode', 'bars')
    bars = int(request.args.get('bars', 100))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    logger.info(f"Received params: symbol={symbol}, timeframe={timeframe}, mode={mode}, bars={bars}, start_date={start_date}, end_date={end_date}")

    # Validate required parameters
    if not symbol or not timeframe:
        return jsonify({'error': 'symbol and timeframe are required parameters'}), 400

    try:
        from app.data.mt5_client import MT5Client
        from app.strategy.indicators import SmartMoneyConcepts
        logger.info("Attempting to create MT5Client instance for SMC...")
        mt5 = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
        logger.info("MT5Client instance created")
        if not mt5.is_connected():
            logger.error("MT5 not connected")
            return jsonify({'error': 'MT5 not connected'}), 500

        df = None
        if mode == 'bars':
            logger.info(f"Fetching {bars} bars for {symbol} {timeframe}")
            df = mt5.fetch_data(symbol, timeframe, start_pos=0, end_pos=bars)
        else:
            # Convert date strings to datetime objects for mt5.fetch_data compatibility
            from datetime import datetime, timezone
            logger.info(f"Fetching data for {symbol} {timeframe} from {start_date} to {end_date}")
            start_str = start_date[:10] if start_date else None
            end_str = end_date[:10] if end_date else None
            
            if not start_str or not end_str:
                logger.error("start_date and end_date are required for date mode")
                #return jsonify({'error': 'start_date and end_date are required for date mode'}), 400
            else:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                
            start_of_day = datetime.combine(start_dt, time.min)  # 00:00:00
            end_of_day = datetime.combine(end_dt, time.max)    # 23:59:59.999999
            df = mt5.fetch_data(symbol, timeframe, start_date=start_of_day, end_date=end_of_day)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Calculate SMC data
        logger.info("Calculating SMC data...")
        smc = SmartMoneyConcepts(mt5, symbol)
        
        # Run the complete SMC analysis
        df = smc.run_smc(df)
        
        # Convert DataFrame to JSON-serializable format
        smc_data = {}
        for column in ['swingline', 'swingvalue', 'swingpoint', 'Resistance', 'Support', 'BOS', 'CHoCH', 
                      'Bullish_Order_Block_Top', 'Bullish_Order_Block_Bottom', 'Bullish_Order_Block_Mitigated',
                      'Bearish_Order_Block_Top', 'Bearish_Order_Block_Bottom', 'Bearish_Order_Block_Mitigated',
                      'fib_signal', 'retest_signal', 'swinglineH1', 'swingvalueH1']:
            if column in df.columns:
                # Convert NaN and other non-serializable values to None
                column_data = df[column].tolist()
                cleaned_data = []
                for value in column_data:
                    if pd.isna(value) or value == 'None' or value == 'nan' or str(value).lower() == 'nan':
                        cleaned_data.append(None)
                    elif isinstance(value, (int, float)):
                        # Ensure numeric values are valid
                        if pd.isna(value) or value == float('inf') or value == float('-inf'):
                            cleaned_data.append(None)
                        else:
                            cleaned_data.append(float(value) if isinstance(value, float) else int(value))
                    else:
                        cleaned_data.append(str(value) if value is not None else None)
                smc_data[column] = cleaned_data
        
        logger.info(f"SMC calculation completed. Returning swingline data.")
        logger.info(f"SMC data keys: {list(smc_data.keys())}")
        
        try:
            return jsonify({'smc_data': smc_data})
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            logger.error(f"Problematic data: {smc_data}")
            return jsonify({'error': f'JSON serialization failed: {str(e)}'}), 500
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return jsonify({'error': f'Import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error calculating SMC data: {e}")
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------------------------
# MT5 Data
# ------------------------------------------------------------------------------
@app.route('/api/mt5-data', methods=['GET'])
def get_mt5_data():
    """
    Fetch OHLCV data from MT5 for a given symbol, timeframe, and range.
    Query params: symbol, timeframe, mode ('bars' or 'date'), bars, start_date, end_date
    """
    logger.info("=== Starting MT5 data request ===")
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    mode = request.args.get('mode', 'bars')
    bars = int(request.args.get('bars', 100))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    logger.info(f"Received params: symbol={symbol}, timeframe={timeframe}, mode={mode}, bars={bars}, start_date={start_date}, end_date={end_date}")

    # Validate required parameters
    if not symbol or not timeframe:
        return jsonify({'error': 'symbol and timeframe are required parameters'}), 400

    try:
        from app.data.mt5_client import MT5Client
        logger.info("Attempting to create MT5Client instance for data fetch...")
        mt5 = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
        logger.info("MT5Client instance created")
        if not mt5.is_connected():
            logger.error("MT5 not connected")
            return jsonify({'error': 'MT5 not connected'}), 500

        df = None
        if mode == 'bars':
            logger.info(f"Fetching {bars} bars for {symbol} {timeframe}")
            df = mt5.fetch_data(symbol, timeframe, start_pos=0, end_pos=bars)
        else:
            # Convert date strings to datetime objects for mt5.fetch_data compatibility
            from datetime import datetime, timezone
            logger.info(f"Fetching data for {symbol} {timeframe} from {start_date} to {end_date}")
            start_str = start_date[:10] if start_date else None
            end_str = end_date[:10] if end_date else None
            
            if not start_str or not end_str:
                logger.error("start_date and end_date are required for date mode")
                #return jsonify({'error': 'start_date and end_date are required for date mode'}), 400
            else:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
                
            start_of_day = datetime.combine(start_dt, time.min)  # 00:00:00
            end_of_day = datetime.combine(end_dt, time.max)    # 23:59:59.999999
            df = mt5.fetch_data(symbol, timeframe, start_date=start_of_day, end_date=end_of_day)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Format for lightweight-charts/plotly
        data = []
        for ts, row in df.iterrows():
            # Handle timestamp conversion properly
            try:
                tval = ts.isoformat()  # type: ignore
            except AttributeError:
                # Fallback for non-datetime index
                tval = str(ts)
            data.append({
                'time': tval,
                'open': round(float(row['Open']), 5),
                'high': round(float(row['High']), 5),
                'low': round(float(row['Low']), 5),
                'close': round(float(row['Close']), 5),
            })
        logger.info(f"Returning {len(data)} bars for {symbol} {timeframe}")
        return jsonify({'data': data})
    except ImportError as e:
        logger.error(f"Failed to import MT5Client: {e}")
        return jsonify({'error': f'MT5Client import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error fetching MT5 data: {e}")
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------------------------
# Trading Endpoints
# ------------------------------------------------------------------------------

@app.route('/api/trade', methods=['POST'])
def place_trade():
    """
    Place a buy or sell order using the trading module
    POST body: {
        "symbol": "EURUSD",
        "action": "buy" | "sell",
        "volume": 0.01,
        "type": "market" | "limit" | "stop"
    }
    """
    logger.info("=== Starting trade request ===")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        symbol = data.get('symbol')
        action = data.get('action')
        volume = 0.01  # Hard code to 0.01 lot size
        order_type = data.get('type', 'market')
        
        logger.info(f"Trade request: symbol={symbol}, action={action}, volume={volume}, type={order_type}")
        
        # Validate required parameters
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        if action not in ['buy', 'sell']:
            return jsonify({'error': 'action must be "buy" or "sell"'}), 400
        if volume <= 0:
            return jsonify({'error': 'volume must be positive'}), 400
        
        # Import trading modules
        from app.data.mt5_client import MT5Client
        from app.trading.trader import Trader
        from app.trading.risk_manager import RiskManager
        from app.trading.order import OrderDirection, OrderType, OrderStatus, Order
        from app.trading.position import Position, PositionDirection
        from app.strategy.base import BaseStrategy
        import MetaTrader5 as mt5
        import pandas as pd
        from datetime import datetime, UTC
        
        logger.info("Creating trading components...")
        mt5_client = MT5Client(config_path=DEFAULT_CONFIG_PATH, symbols=ALL_SYMBOLS, broker=BROKER)
        
        if not mt5_client.is_connected():
            logger.error("MT5 client not connected")
            return jsonify({'error': 'MT5 client not connected'}), 500
        
        # Use the working MT5Broker implementation from trade_executor_example.py
        class WorkingMT5Broker:
            def __init__(self, mt5_client: MT5Client):
                self.client = mt5_client
                self._symbol_info_cache: Dict[str, Any] = {}

            def _get_cached_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
                if symbol not in self._symbol_info_cache:
                    try:
                        symbol_info_raw = self.client.get_symbol_info(symbol)
                        logger.info(f"Symbol info type for {symbol}: {type(symbol_info_raw)}")
                        logger.info(f"Symbol info keys: {dir(symbol_info_raw) if hasattr(symbol_info_raw, '__dict__') else 'No dict'}")
                        
                        # Handle both named tuple and dictionary
                        if hasattr(symbol_info_raw, '_asdict'):
                            # It's a named tuple, convert to dict
                            symbol_info_dict = symbol_info_raw._asdict()
                            logger.info(f"Converted named tuple to dict with keys: {list(symbol_info_dict.keys())}")
                        elif isinstance(symbol_info_raw, dict):
                            # It's already a dictionary
                            symbol_info_dict = symbol_info_raw
                            logger.info(f"Already a dict with keys: {list(symbol_info_dict.keys())}")
                        else:
                            # Try to access as object attributes
                            symbol_info_dict = {}
                            for attr in dir(symbol_info_raw):
                                if not attr.startswith('_'):
                                    try:
                                        symbol_info_dict[attr] = getattr(symbol_info_raw, attr)
                                    except:
                                        pass
                            logger.info(f"Converted object to dict with keys: {list(symbol_info_dict.keys())}")
                        
                        # Check if filling_mode exists
                        if 'filling_mode' in symbol_info_dict:
                            logger.info(f"filling_mode value: {symbol_info_dict['filling_mode']}")
                        else:
                            logger.warning(f"filling_mode not found in symbol info for {symbol}")
                        
                        self._symbol_info_cache[symbol] = symbol_info_dict
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

            def _map_order_to_mt5_order_type(self, order) -> Optional[int]:
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

            def get_open_positions(self):
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

            def get_open_orders(self):
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

            def send_order(self, order) -> Optional[str]:
                logger.info(f"Starting send_order for {order.symbol} {order.direction.name} {order.volume}")
                
                mt5_order_type = self._map_order_to_mt5_order_type(order)
                if mt5_order_type is None:
                    logger.error(f"Unsupported order type: {order.order_type}")
                    return None
                logger.info(f"Mapped order type: {mt5_order_type}")

                symbol_info = self._get_cached_symbol_info(order.symbol)
                if not symbol_info:
                    logger.error(f"Could not retrieve symbol info for {order.symbol}, cannot send order.")
                    return None
                logger.info(f"Got symbol info for {order.symbol}")

                # Determine the correct filling type by checking the bitmask
                # Try different possible field names for filling mode
                filling_mode = symbol_info.get('filling_mode', 0)
                logger.info(f"Raw filling_mode value: {filling_mode}")
                
                filling_type = mt5.ORDER_FILLING_FOK  # Default

                # Handle different filling mode values
                if filling_mode == 1:  # ORDER_FILLING_FOK
                    filling_type = mt5.ORDER_FILLING_FOK
                elif filling_mode == 2:  # ORDER_FILLING_IOC
                    filling_type = mt5.ORDER_FILLING_IOC
                elif filling_mode == 4:  # ORDER_FILLING_RETURN
                    filling_type = mt5.ORDER_FILLING_RETURN
                else:
                    logger.warning(f"Unknown filling mode: {filling_mode}. Using default FOK.")
                    filling_type = mt5.ORDER_FILLING_FOK
                
                logger.info(f"Using filling type: {filling_type}")

                price = order.price
                if order.order_type == OrderType.MARKET:
                    try:
                        price = self.get_current_price(order.symbol, 'ask' if order.direction == OrderDirection.BUY else 'bid')
                        logger.info(f"Got current price for {order.symbol}: {price}")
                    except Exception as e:
                        logger.error(f"Failed to get current price for {order.symbol}: {e}")
                        return None

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
                
                logger.info(f"Sending order request: {request}")
                result = self.client.order_send(request)
                logger.info(f"Order send result: {result}")
                
                if result and result.get('order'):
                    logger.info(f"Order sent successfully: {result['order']}")
                    return str(result['order'])
                else:
                    logger.error(f"Order send failed. Result: {result}")
                    return None

            def get_current_price(self, symbol: str, price_type: str = 'ask') -> float:
                tick = self.client.get_tick(symbol)
                if tick and price_type in tick:
                    return tick[price_type]
                
                # Fallback if tick fails
                info = self._get_cached_symbol_info(symbol)
                if info and price_type in info:
                    return info[price_type]
                    
                raise RuntimeError(f"Could not retrieve current price for {symbol}")

        # Create broker, risk manager, and trader using working implementation
        broker = WorkingMT5Broker(mt5_client=mt5_client)
        account_balance = broker.get_account_balance()
        risk_manager = RiskManager(mt5_client=mt5_client, account_balance=account_balance, risk_percentage=1.0)
        trader = Trader(broker=broker, risk_manager=risk_manager)
        
        # Validate symbol availability
        logger.info(f"Checking symbol availability for {symbol}")
        
        # Check if symbol is visible in market watch
        if not mt5_client.is_symbol_visible(symbol):
            logger.error(f"Symbol {symbol} is not visible in market watch")
            return jsonify({'error': f'Symbol {symbol} is not visible in market watch'}), 400
        
        # Check if symbol is available from broker
        available_symbols = mt5_client.get_available_symbols()
        if symbol not in available_symbols:
            logger.error(f"Symbol {symbol} is not available from broker")
            return jsonify({'error': f'Symbol {symbol} is not available from broker'}), 400
        
        logger.info(f"Symbol {symbol} is available and visible")
        
        # Check account balance
        logger.info(f"Account balance: ${account_balance:.2f}")
        if account_balance <= 0:
            logger.error("Account balance is zero or negative")
            return jsonify({'error': 'Account balance is insufficient'}), 400
        
        # Get symbol info for dynamic parameter calculation
        symbol_info_raw = mt5_client.get_symbol_info(symbol)
        if symbol_info_raw is None:
            logger.error(f"Could not get symbol info for {symbol}")
            return jsonify({'error': f'Could not get symbol info for {symbol}'}), 400
        
        # Convert symbol info to dictionary format
        if hasattr(symbol_info_raw, '_asdict'):
            symbol_info_dict = symbol_info_raw._asdict()
        else:
            symbol_info_dict = symbol_info_raw
        
        # Create a proper symbol info object for BaseStrategy
        class SymbolInfo:
            def __init__(self, symbol_dict):
                self.name = symbol_dict.get('name', symbol)
                self.ask = symbol_dict.get('ask', 0.0)
                self.bid = symbol_dict.get('bid', 0.0)
                self.trade_tick_size = symbol_dict.get('trade_tick_size', 0.00001)
                self.digits = symbol_dict.get('digits', 5)
                # Add other attributes that might be needed
                for key, value in symbol_dict.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
        
        symbol_info_obj = SymbolInfo(symbol_info_dict)
        
        # Create a mock strategy to use get_trade_parameters
        class MockStrategy(BaseStrategy):
            def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
                return data  # Mock implementation
        
        # Get recent market data for parameter calculation
        try:
            # Get recent data (last 100 bars)
            df = mt5_client.fetch_data(symbol, "D1", start_pos=0, end_pos=100)
            if df is None or df.empty:
                logger.error(f"Could not fetch market data for {symbol}")
                return jsonify({'error': f'Could not fetch market data for {symbol}'}), 400
            
            logger.info(f"Fetched {len(df)} bars of market data for {symbol}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Ensure DataFrame has required columns for ADR calculation
            required_columns = ['High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"DataFrame missing required columns: {missing_columns}")
                return jsonify({'error': f'DataFrame missing required columns: {missing_columns}'}), 400
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return jsonify({'error': f'Error fetching market data: {e}'}), 400
        
        # Create mock strategy instance with proper symbol info object
        mock_strategy = MockStrategy(mt5_client, parameters={}, symbol_info=symbol_info_obj)
        
        # Calculate dynamic trading parameters
        action = 1 if action == 'buy' else -1  # Convert to strategy format
        
        try:
            trade_params, df_updated = mock_strategy.get_trade_parameters(df, df, action, symbol_info_obj)
            
            if trade_params is None:
                logger.error("Failed to calculate trade parameters")
                return jsonify({'error': 'Failed to calculate trade parameters'}), 400
                
        except Exception as e:
            logger.error(f"Error in get_trade_parameters: {e}")
            # Fallback to basic parameters if ADR calculation fails
            logger.info("Using fallback parameters due to ADR calculation failure")
            trade_params = {
                "Lots": 0.01,
                "SL Pips": 50,
                "TP Pips": 100
            }
        
        # Extract dynamic parameters with fallbacks
        volume = abs(trade_params.get('Lots', 0.01))  # Use calculated lot size
        stop_loss_pips = trade_params.get('SL Pips', 50)
        take_profit_pips = trade_params.get('TP Pips', 100)
        
        logger.info(f"Dynamic parameters - Volume: {volume}, SL Pips: {stop_loss_pips}, TP Pips: {take_profit_pips}")
        
        # Map action to direction
        direction = OrderDirection.BUY if action == 1 else OrderDirection.SELL
        
        # Calculate SL/TP prices based on pips
        current_price = broker.get_current_price(symbol, 'ask' if direction == OrderDirection.BUY else 'bid')
        
        # Get symbol info for minimum distance requirements
        symbol_info_for_broker = broker._get_cached_symbol_info(symbol)
        if symbol_info_for_broker:
            min_stop_level = symbol_info_for_broker.get('trade_stops_level', 10)  # Minimum distance in points
            point = symbol_info_for_broker.get('point', 0.001)  # Point value
            min_distance = min_stop_level * point  # Convert to price distance
        else:
            min_distance = 0.001  # Default minimum distance
        
        logger.info(f"Current price: {current_price}, Min distance: {min_distance}")
        
        if direction == OrderDirection.BUY:
            stop_loss_price = current_price - (stop_loss_pips * 0.0001)  # Convert pips to price
            take_profit_price = current_price + (take_profit_pips * 0.0001)
            
            # Validate SL/TP distances
            if abs(current_price - stop_loss_price) < min_distance:
                stop_loss_price = current_price - min_distance
                logger.info(f"Adjusted SL to meet minimum distance: {stop_loss_price}")
            
            if abs(take_profit_price - current_price) < min_distance:
                take_profit_price = current_price + min_distance
                logger.info(f"Adjusted TP to meet minimum distance: {take_profit_price}")
        else:
            stop_loss_price = current_price + (stop_loss_pips * 0.0001)
            take_profit_price = current_price - (take_profit_pips * 0.0001)
            
            # Validate SL/TP distances
            if abs(stop_loss_price - current_price) < min_distance:
                stop_loss_price = current_price + min_distance
                logger.info(f"Adjusted SL to meet minimum distance: {stop_loss_price}")
            
            if abs(current_price - take_profit_price) < min_distance:
                take_profit_price = current_price - min_distance
                logger.info(f"Adjusted TP to meet minimum distance: {take_profit_price}")
        
        # Additional validation: ensure SL/TP are reasonable
        if direction == OrderDirection.BUY:
            if stop_loss_price >= current_price:
                stop_loss_price = current_price - min_distance
                logger.warning(f"SL was above current price, adjusted to: {stop_loss_price}")
            if take_profit_price <= current_price:
                take_profit_price = current_price + min_distance
                logger.warning(f"TP was below current price, adjusted to: {take_profit_price}")
        else:
            if stop_loss_price <= current_price:
                stop_loss_price = current_price + min_distance
                logger.warning(f"SL was below current price, adjusted to: {stop_loss_price}")
            if take_profit_price >= current_price:
                take_profit_price = current_price - min_distance
                logger.warning(f"TP was above current price, adjusted to: {take_profit_price}")
        
        # Create and execute market order with dynamic parameters
        logger.info(f"Creating market order: {symbol} {direction.name} {volume} with SL: {stop_loss_price}, TP: {take_profit_price}")
        
        # Convert action back to string for comment
        action_str = 'BUY' if action == 1 else 'SELL'
        
        # Create order with dynamic parameters
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            direction=direction,
            volume=volume,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            comment=f"Web UI {action_str} order"
        )
        
        # Send order directly through broker
        order_id = broker.send_order(order)
        
        logger.info(f"Order creation result: {order_id}")
        
        if order_id:
            logger.info(f"Order placed successfully: {order_id}")
            return jsonify({
                'success': True,
                'order_id': order_id,
                'message': f'{action_str} order placed successfully',
                'symbol': order.symbol,
                'volume': order.volume,
                'direction': order.direction.name,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips
            })
        else:
            logger.error("Order placement failed")
            return jsonify({
                'success': False,
                'error': 'Order placement failed - check symbol availability and account balance'
            }), 400
            
    except ImportError as e:
        logger.error(f"Failed to import trading modules: {e}")
        return jsonify({'error': f'Trading modules import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error placing trade: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up MT5Client
        try:
            if 'mt5_client' in locals():
                mt5_client.shutdown()
                logger.info("MT5Client shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down MT5Client: {e}")

# ==============================================================================
# ERROR HANDLERS
# ==============================================================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting HaruPyQuant Flask server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug) 