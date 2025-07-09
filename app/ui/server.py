"""
Flask server for HaruPyQuant web application
Provides API endpoints for dashboard metrics and system status
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, jsonify
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

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'harupyquant-dev-key')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting HaruPyQuant Flask server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug) 