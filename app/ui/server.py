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
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add the parent directory to the path to import app modules
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

# Also add the project root to the path
project_root = os.path.dirname(app_dir)
sys.path.insert(0, project_root)

try:
    from app.util.logger import get_logger
    logger = get_logger(__name__)
    print(f"Successfully imported logger from {app_dir}")
except ImportError as e:
    print(f"Warning: Could not import app logger: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback logging if app logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
        logger.info("✓ MT5Client imported successfully for status check")
        
        mt5_status = "disconnected"
        try:
            logger.info("Creating MT5Client instance for status check...")
            mt5_client = MT5Client()
            logger.info("✓ MT5Client instance created for status check")
            
            logger.info("Checking MT5 connection status...")
            if mt5_client.is_connected():
                mt5_status = "connected"
                logger.info("✓ MT5 is connected")
            else:
                logger.warning("❌ MT5 is not connected")
            
            logger.info("Shutting down MT5Client...")
            mt5_client.shutdown()
            logger.info("✓ MT5Client shut down")
            
        except Exception as e:
            logger.error(f"❌ MT5 connection check failed: {e}")
            logger.exception("Full exception details:")
        
        logger.info(f"Final MT5 status: {mt5_status}")
        logger.info("=== Returning status response ===")
        
        return jsonify({
            'status': 'operational',
            'mt5_connection': mt5_status,
            'timestamp': datetime.utcnow().isoformat()
        })
    except ImportError as e:
        logger.error(f"❌ Failed to import MT5Client for status check: {e}")
        return jsonify({
            'status': 'error',
            'error': f"MT5Client import failed: {str(e)}",
            'mt5_connection': 'unavailable',
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
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
        logger.info("✓ MT5Client imported successfully")
        
        logger.info("Creating MT5Client instance...")
        mt5_client = MT5Client()
        logger.info("✓ MT5Client instance created")
        
        logger.info("Checking MT5 connection status...")
        connection_status = mt5_client.is_connected()
        logger.info(f"MT5 connection status: {connection_status}")
        
        if not connection_status:
            logger.warning("❌ MT5 not connected, returning fallback dummy data")
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
        
        logger.info("✓ MT5 is connected, fetching account info...")
        account = mt5_client.get_account_info()
        logger.info(f"Account info retrieved: {account}")
        
        logger.info("Fetching open positions...")
        open_positions = mt5_client.get_positions()
        logger.info(f"Open positions retrieved: {len(open_positions)} positions")
        
        logger.info("Shutting down MT5 connection...")
        mt5_client.shutdown()
        logger.info("✓ MT5 connection shut down")
        
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
        logger.error(f"❌ Failed to import MT5Client: {e}")
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
        logger.exception(f"❌ Error fetching MT5 dashboard metrics: {e}")
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

    try:
        from app.data.mt5_client import MT5Client
        from app.strategy.indicators import ema_series
        logger.info("Attempting to create MT5Client instance for indicators...")
        mt5 = MT5Client()
        logger.info("✓ MT5Client instance created")
        if not mt5.is_connected():
            logger.error("❌ MT5 not connected")
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
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_str else None
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_str else None
            df = mt5.fetch_data(symbol, timeframe, start_date=start_dt, end_date=end_dt)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Calculate indicators
        indicator_data = {}
        indicator_list = [ind.strip() for ind in indicators.split(',') if ind.strip()]
        
        for indicator in indicator_list:
            try:
                if indicator == 'EMA20':
                    ema_values = ema_series(df['Close'], 20)
                    indicator_data[indicator] = ema_values.tolist()
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
        logger.error(f"❌ Failed to import required modules: {e}")
        return jsonify({'error': f'Import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error calculating technical indicators: {e}")
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

    try:
        from app.data.mt5_client import MT5Client
        from app.strategy.indicators import SmartMoneyConcepts
        logger.info("Attempting to create MT5Client instance for SMC...")
        mt5 = MT5Client()
        logger.info("✓ MT5Client instance created")
        if not mt5.is_connected():
            logger.error("❌ MT5 not connected")
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
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_str else None
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_str else None
            df = mt5.fetch_data(symbol, timeframe, start_date=start_dt, end_date=end_dt)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Calculate SMC data
        logger.info("Calculating SMC data...")
        smc = SmartMoneyConcepts(symbol)
        
        # Run the complete SMC analysis
        df = smc.run_smc(df)
        
        # Convert DataFrame to JSON-serializable format
        smc_data = {}
        for column in ['swingline', 'swing_value']:
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
        logger.error(f"❌ Failed to import required modules: {e}")
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

    try:
        from app.data.mt5_client import MT5Client
        logger.info("Attempting to create MT5Client instance for data fetch...")
        mt5 = MT5Client()
        logger.info("✓ MT5Client instance created")
        if not mt5.is_connected():
            logger.error("❌ MT5 not connected")
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
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_str else None
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_str else None
            df = mt5.fetch_data(symbol, timeframe, start_date=start_dt, end_date=end_dt)

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return jsonify({'error': 'No data returned from MT5'}), 404

        # Format for lightweight-charts/plotly
        data = []
        for ts, row in df.iterrows():
            tval = ts.isoformat()
            data.append({
                'time': tval,
                'open': round(float(row['Open']), 5),
                'high': round(float(row['High']), 5),
                'low': round(float(row['Low']), 5),
                'close': round(float(row['Close']), 5),
            })
        logger.info(f"Returning {len(data)} bars for {symbol} {timeframe}")
        #logger.info(f"Data: {data}")
        return jsonify({'data': data})
    except ImportError as e:
        logger.error(f"❌ Failed to import MT5Client: {e}")
        return jsonify({'error': f'MT5Client import failed: {str(e)}'}), 500
    except Exception as e:
        logger.exception(f"Error fetching MT5 data: {e}")
        return jsonify({'error': str(e)}), 500

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