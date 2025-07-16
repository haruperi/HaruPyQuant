import os
import sys
import signal
import time
import json
import pickle
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import atexit

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemState:
    """Represents the current state of the trading system."""
    timestamp: str
    status: str  # 'running', 'stopped', 'error', 'recovering'
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    active_positions: int
    last_error: Optional[str] = None
    restart_count: int = 0
    max_restarts: int = 5
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class CrashRecoveryManager:
    """
    Manages crash recovery mechanisms for the HaruPyQuant trading system.
    
    Features:
    - Exception handling and logging
    - State persistence and recovery
    - Automatic restart mechanisms
    - Health monitoring
    - Graceful shutdown procedures
    - Resource cleanup
    """
    
    def __init__(self, 
                 state_file: str = "system_state.json",
                 max_restarts: int = 5,
                 restart_delay: int = 30,
                 health_check_interval: int = 60):
        """
        Initialize the crash recovery manager.
        
        Args:
            state_file: File to persist system state
            max_restarts: Maximum number of restart attempts
            restart_delay: Delay between restart attempts (seconds)
            health_check_interval: Interval for health checks (seconds)
        """
        self.state_file = Path(state_file)
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        
        # Initialize state
        self.state = SystemState(
            timestamp=datetime.now().isoformat(),
            status="initializing",
            uptime=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            active_connections=0,
            active_positions=0,
            max_restarts=max_restarts
        )
        
        # Recovery state
        self.start_time = time.time()
        self.is_shutting_down = False
        self.health_monitor_thread = None
        self.recovery_callbacks = []
        self.cleanup_callbacks = []
        
        # Load previous state if exists
        self._load_state()
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        logger.info("CrashRecoveryManager initialized")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers registered")
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.graceful_shutdown()
    
    def _load_state(self):
        """Load system state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Update state with loaded data
                    for key, value in data.items():
                        if hasattr(self.state, key):
                            setattr(self.state, key, value)
                logger.info(f"Loaded system state from {self.state_file}")
        except Exception as e:
            logger.warning(f"Could not load system state: {e}")
    
    def _save_state(self):
        """Save current system state to file."""
        try:
            # Update state before saving
            self.state.timestamp = datetime.now().isoformat()
            self.state.uptime = time.time() - self.start_time
            
            # Get system metrics
            process = psutil.Process()
            self.state.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.state.cpu_usage = process.cpu_percent()
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
            
            logger.debug("System state saved")
        except Exception as e:
            logger.error(f"Could not save system state: {e}")
    
    def register_recovery_callback(self, callback: Callable):
        """Register a callback to be called during recovery."""
        self.recovery_callbacks.append(callback)
        logger.debug(f"Registered recovery callback: {callback.__name__}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be called during cleanup."""
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def start_health_monitoring(self):
        """Start the health monitoring thread."""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.health_monitor_thread.start()
        logger.info("Health monitoring started")
    
    def _health_monitor_loop(self):
        """Health monitoring loop."""
        while not self.is_shutting_down:
            try:
                self._perform_health_check()
                self._save_state()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)  # Short delay on error
    
    def _perform_health_check(self):
        """Perform system health check."""
        try:
            process = psutil.Process()
            
            # Check memory usage
            memory_percent = process.memory_percent()
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 90:
                logger.warning(f"Low disk space: {disk_usage.percent:.1f}% used")
            
            logger.debug(f"Health check completed - Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    @contextmanager
    def exception_handler(self, context: str = "unknown"):
        """
        Context manager for handling exceptions with recovery.
        
        Args:
            context: Context description for logging
        """
        try:
            yield
        except Exception as e:
            self._handle_exception(e, context)
            raise
    
    def _handle_exception(self, exception: Exception, context: str):
        """Handle exceptions and initiate recovery if needed."""
        error_msg = f"Exception in {context}: {str(exception)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state
        self.state.status = "error"
        self.state.last_error = error_msg
        self.state.recovery_attempts += 1
        
        # Log full traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Check if recovery is possible
        if self.state.recovery_attempts <= self.state.max_recovery_attempts:
            logger.info(f"Attempting recovery (attempt {self.state.recovery_attempts}/{self.state.max_recovery_attempts})")
            self._attempt_recovery()
        else:
            logger.critical("Maximum recovery attempts exceeded, shutting down")
            self.graceful_shutdown()
    
    def _attempt_recovery(self):
        """Attempt system recovery."""
        try:
            self.state.status = "recovering"
            self._save_state()
            
            # Call recovery callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback()
                    logger.debug(f"Recovery callback {callback.__name__} completed")
                except Exception as e:
                    logger.error(f"Recovery callback {callback.__name__} failed: {e}")
            
            # Reset recovery state
            self.state.status = "running"
            self.state.recovery_attempts = 0
            self.state.last_error = None
            
            logger.info("Recovery completed successfully")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self.state.status = "error"
    
    def restart_application(self):
        """Restart the application."""
        if self.state.restart_count >= self.max_restarts:
            logger.critical("Maximum restart attempts exceeded")
            self.graceful_shutdown()
            return
        
        logger.info(f"Restarting application (attempt {self.state.restart_count + 1}/{self.max_restarts})")
        
        # Update restart count
        self.state.restart_count += 1
        self._save_state()
        
        # Wait before restart
        time.sleep(self.restart_delay)
        
        # Restart the application
        try:
            os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart application: {e}")
            self.graceful_shutdown()
    
    def graceful_shutdown(self):
        """Perform graceful shutdown."""
        if self.is_shutting_down:
            return
        
        logger.info("Initiating graceful shutdown")
        self.is_shutting_down = True
        self.state.status = "stopping"
        
        try:
            # Call cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                    logger.debug(f"Cleanup callback {callback.__name__} completed")
                except Exception as e:
                    logger.error(f"Cleanup callback {callback.__name__} failed: {e}")
            
            # Save final state
            self.state.status = "stopped"
            self._save_state()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            # Force exit if needed
            sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if not self.is_shutting_down:
                self.graceful_shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "uptime_seconds": time.time() - self.start_time,
                "status": self.state.status,
                "restart_count": self.state.restart_count,
                "recovery_attempts": self.state.recovery_attempts,
                "last_error": self.state.last_error
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}


# Global instance
recovery_manager = None


def get_recovery_manager() -> CrashRecoveryManager:
    """Get the global recovery manager instance."""
    global recovery_manager
    if recovery_manager is None:
        recovery_manager = CrashRecoveryManager()
    return recovery_manager


def initialize_recovery_manager(**kwargs) -> CrashRecoveryManager:
    """Initialize the global recovery manager."""
    global recovery_manager
    recovery_manager = CrashRecoveryManager(**kwargs)
    return recovery_manager 