#!/usr/bin/env python3
"""
Watchdog service runner for HaruPyQuant.

This script runs the watchdog service that monitors the main application
and automatically restarts it if it crashes or becomes unresponsive.

Usage:
    python scripts/run_watchdog.py [options]

Options:
    --app-script PATH          Application script to monitor (default: main.py)
    --max-restarts N           Maximum restart attempts (default: 10)
    --restart-delay SECONDS    Delay between restarts (default: 30)
    --health-check-interval SECONDS  Health check interval (default: 30)
    --max-memory MB            Maximum memory usage (default: 1024)
    --max-cpu PERCENT          Maximum CPU usage (default: 90)
    --daemon                   Run as daemon process
    --log-file PATH            Log file path
    --pid-file PATH            PID file path
"""

import os
import sys
import argparse
import signal
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.util.watchdog import WatchdogService
from app.util.logger import get_logger

logger = get_logger(__name__)


def setup_signal_handlers(watchdog):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, stopping watchdog")
        watchdog.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def write_pid_file(pid_file):
    """Write PID to file."""
    try:
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"PID written to {pid_file}")
    except Exception as e:
        logger.error(f"Could not write PID file: {e}")


def remove_pid_file(pid_file):
    """Remove PID file."""
    try:
        if pid_file.exists():
            pid_file.unlink()
            logger.info(f"PID file {pid_file} removed")
    except Exception as e:
        logger.error(f"Could not remove PID file: {e}")


def check_pid_file(pid_file):
    """Check if PID file exists and process is running."""
    if not pid_file.exists():
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is running
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="HaruPyQuant Watchdog Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--app-script",
        default="main.py",
        help="Application script to monitor (default: main.py)"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="Maximum restart attempts (default: 10)"
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=30,
        help="Delay between restarts in seconds (default: 30)"
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=30,
        help="Health check interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=1024,
        help="Maximum memory usage in MB (default: 1024)"
    )
    parser.add_argument(
        "--max-cpu",
        type=int,
        default=90,
        help="Maximum CPU usage percentage (default: 90)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon process"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    parser.add_argument(
        "--pid-file",
        default="watchdog.pid",
        help="PID file path (default: watchdog.pid)"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running watchdog service"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show watchdog status"
    )
    
    args = parser.parse_args()
    
    pid_file = Path(args.pid_file)
    
    # Handle stop command
    if args.stop:
        if check_pid_file(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent stop signal to watchdog process {pid}")
                time.sleep(2)
                if check_pid_file(pid_file):
                    logger.warning("Watchdog did not stop gracefully, forcing kill")
                    os.kill(pid, signal.SIGKILL)
                remove_pid_file(pid_file)
            except Exception as e:
                logger.error(f"Error stopping watchdog: {e}")
        else:
            logger.info("No watchdog process found")
        return
    
    # Handle status command
    if args.status:
        if check_pid_file(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                logger.info(f"Watchdog is running with PID {pid}")
                
                # Try to get status from the process
                import psutil
                try:
                    process = psutil.Process(pid)
                    logger.info(f"Process status: {process.status()}")
                    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
                    logger.info(f"CPU usage: {process.cpu_percent():.1f}%")
                except psutil.NoSuchProcess:
                    logger.warning("Process not found")
            except Exception as e:
                logger.error(f"Error getting status: {e}")
        else:
            logger.info("Watchdog is not running")
        return
    
    # Check if already running
    if check_pid_file(pid_file):
        logger.error("Watchdog is already running")
        logger.info("Use --stop to stop the running watchdog")
        sys.exit(1)
    
    # Create watchdog service
    watchdog = WatchdogService(
        app_script=args.app_script,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        health_check_interval=args.health_check_interval,
        max_memory_mb=args.max_memory,
        max_cpu_percent=args.max_cpu
    )
    
    # Setup signal handlers
    setup_signal_handlers(watchdog)
    
    # Write PID file
    write_pid_file(pid_file)
    
    try:
        # Start watchdog
        logger.info("Starting watchdog service...")
        watchdog.start()
        
        # Keep running
        while watchdog.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Watchdog error: {e}")
    finally:
        # Cleanup
        watchdog.stop()
        remove_pid_file(pid_file)
        logger.info("Watchdog service stopped")


if __name__ == "__main__":
    main() 