import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import psutil
import json

from .logger import get_logger

logger = get_logger(__name__)


class WatchdogService:
    """
    Watchdog service that monitors the main application and restarts it if needed.
    
    Features:
    - Process monitoring
    - Automatic restart on crash
    - Health checks
    - Log monitoring
    - Configuration management
    """
    
    def __init__(self, 
                 app_script: str = "main.py",
                 max_restarts: int = 10,
                 restart_delay: int = 30,
                 health_check_interval: int = 30,
                 max_memory_mb: int = 1024,
                 max_cpu_percent: int = 90):
        """
        Initialize the watchdog service.
        
        Args:
            app_script: Path to the main application script
            max_restarts: Maximum number of restart attempts
            restart_delay: Delay between restart attempts (seconds)
            health_check_interval: Interval for health checks (seconds)
            max_memory_mb: Maximum memory usage before restart (MB)
            max_cpu_percent: Maximum CPU usage before restart (%)
        """
        self.app_script = Path(app_script)
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        
        # Process tracking
        self.app_process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.is_running = False
        self.monitor_thread = None
        
        # Configuration
        self.watchdog_config_file = Path("watchdog_config.json")
        self._load_config()
        
        logger.info("WatchdogService initialized")
    
    def _load_config(self):
        """Load watchdog configuration from file."""
        try:
            if self.watchdog_config_file.exists():
                with open(self.watchdog_config_file, 'r') as f:
                    config = json.load(f)
                    # Update instance variables with config values
                    for key, value in config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                logger.info(f"Loaded watchdog config from {self.watchdog_config_file}")
        except Exception as e:
            logger.warning(f"Could not load watchdog config: {e}")
    
    def _save_config(self):
        """Save current watchdog configuration to file."""
        try:
            config = {
                "max_restarts": self.max_restarts,
                "restart_delay": self.restart_delay,
                "health_check_interval": self.health_check_interval,
                "max_memory_mb": self.max_memory_mb,
                "max_cpu_percent": self.max_cpu_percent,
                "restart_count": self.restart_count
            }
            
            with open(self.watchdog_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.debug("Watchdog config saved")
        except Exception as e:
            logger.error(f"Could not save watchdog config: {e}")
    
    def start(self):
        """Start the watchdog service and the monitored application."""
        if self.is_running:
            logger.warning("Watchdog service already running")
            return
        
        logger.info("Starting watchdog service")
        self.is_running = True
        
        # Start the application
        self._start_application()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="WatchdogMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("Watchdog service started")
    
    def stop(self):
        """Stop the watchdog service and the monitored application."""
        if not self.is_running:
            return
        
        logger.info("Stopping watchdog service")
        self.is_running = False
        
        # Stop the application
        self._stop_application()
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Watchdog service stopped")
    
    def _start_application(self):
        """Start the monitored application."""
        try:
            if self.app_process and self.app_process.poll() is None:
                logger.warning("Application already running")
                return
            
            # Start the application
            cmd = [sys.executable, str(self.app_script)]
            self.app_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"Started application with PID {self.app_process.pid}")
            
            # Reset restart count on successful start
            self.restart_count = 0
            self._save_config()
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            self._handle_startup_failure()
    
    def _stop_application(self):
        """Stop the monitored application."""
        if not self.app_process:
            return
        
        try:
            # Try graceful shutdown first
            logger.info("Attempting graceful shutdown of application")
            self.app_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.app_process.wait(timeout=10)
                logger.info("Application stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("Graceful shutdown failed, forcing kill")
                self.app_process.kill()
                self.app_process.wait()
                logger.info("Application force killed")
                
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
        finally:
            self.app_process = None
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check if application is still running
                if not self._is_application_healthy():
                    logger.warning("Application health check failed")
                    self._restart_application()
                
                # Check system resources
                if self._check_system_resources():
                    logger.warning("System resource limits exceeded")
                    self._restart_application()
                
                # Check for application output
                self._check_application_output()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _is_application_healthy(self) -> bool:
        """Check if the application is healthy."""
        if not self.app_process:
            return False
        
        # Check if process is still running
        if self.app_process.poll() is not None:
            logger.warning(f"Application process terminated with code {self.app_process.returncode}")
            return False
        
        # Check if process is responsive
        try:
            process = psutil.Process(self.app_process.pid)
            if not process.is_running():
                logger.warning("Application process is not running")
                return False
            
            # Check if process is not zombie
            if process.status() == psutil.STATUS_ZOMBIE:
                logger.warning("Application process is zombie")
                return False
                
        except psutil.NoSuchProcess:
            logger.warning("Application process not found")
            return False
        except Exception as e:
            logger.error(f"Error checking process health: {e}")
            return False
        
        return True
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are within limits."""
        if not self.app_process:
            return False
        
        try:
            process = psutil.Process(self.app_process.pid)
            
            # Check memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                logger.warning(f"Memory usage exceeded limit: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                return True
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > self.max_cpu_percent:
                logger.warning(f"CPU usage exceeded limit: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                return True
            
        except psutil.NoSuchProcess:
            logger.warning("Cannot check system resources - process not found")
            return True
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False
        
        return False
    
    def _check_application_output(self):
        """Check application output for errors or warnings."""
        if not self.app_process:
            return
        
        try:
            # Check stdout
            if self.app_process.stdout:
                line = self.app_process.stdout.readline()
                if line:
                    line = line.strip()
                    if "ERROR" in line.upper() or "CRITICAL" in line.upper():
                        logger.warning(f"Application error detected: {line}")
                    elif "WARNING" in line.upper():
                        logger.info(f"Application warning: {line}")
            
            # Check stderr
            if self.app_process.stderr:
                line = self.app_process.stderr.readline()
                if line:
                    line = line.strip()
                    logger.warning(f"Application stderr: {line}")
                    
        except Exception as e:
            logger.debug(f"Error reading application output: {e}")
    
    def _restart_application(self):
        """Restart the application."""
        if self.restart_count >= self.max_restarts:
            logger.critical(f"Maximum restart attempts ({self.max_restarts}) exceeded")
            self.stop()
            return
        
        logger.info(f"Restarting application (attempt {self.restart_count + 1}/{self.max_restarts})")
        
        # Stop current application
        self._stop_application()
        
        # Increment restart count
        self.restart_count += 1
        self._save_config()
        
        # Wait before restart
        time.sleep(self.restart_delay)
        
        # Start new application
        self._start_application()
    
    def _handle_startup_failure(self):
        """Handle application startup failure."""
        logger.error("Application startup failed")
        
        if self.restart_count >= self.max_restarts:
            logger.critical("Maximum restart attempts exceeded, stopping watchdog")
            self.stop()
        else:
            # Try to restart after a longer delay
            logger.info("Scheduling restart after startup failure")
            time.sleep(self.restart_delay * 2)
            self._restart_application()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current watchdog status."""
        status = {
            "is_running": self.is_running,
            "restart_count": self.restart_count,
            "max_restarts": self.max_restarts,
            "app_script": str(self.app_script)
        }
        
        if self.app_process:
            status.update({
                "app_pid": self.app_process.pid,
                "app_returncode": self.app_process.returncode,
                "app_poll": self.app_process.poll()
            })
            
            # Get process info if available
            try:
                process = psutil.Process(self.app_process.pid)
                status.update({
                    "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "status": process.status()
                })
            except psutil.NoSuchProcess:
                status["process_status"] = "not_found"
        
        return status


def run_watchdog():
    """Run the watchdog service as a standalone process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HaruPyQuant Watchdog Service")
    parser.add_argument("--app-script", default="main.py", help="Application script to monitor")
    parser.add_argument("--max-restarts", type=int, default=10, help="Maximum restart attempts")
    parser.add_argument("--restart-delay", type=int, default=30, help="Delay between restarts (seconds)")
    parser.add_argument("--health-check-interval", type=int, default=30, help="Health check interval (seconds)")
    parser.add_argument("--max-memory", type=int, default=1024, help="Maximum memory usage (MB)")
    parser.add_argument("--max-cpu", type=int, default=90, help="Maximum CPU usage (%)")
    
    args = parser.parse_args()
    
    # Create and start watchdog
    watchdog = WatchdogService(
        app_script=args.app_script,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        health_check_interval=args.health_check_interval,
        max_memory_mb=args.max_memory,
        max_cpu_percent=args.max_cpu
    )
    
    try:
        watchdog.start()
        
        # Keep the watchdog running
        while watchdog.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping watchdog")
        watchdog.stop()
    except Exception as e:
        logger.error(f"Watchdog error: {e}")
        watchdog.stop()


if __name__ == "__main__":
    run_watchdog() 