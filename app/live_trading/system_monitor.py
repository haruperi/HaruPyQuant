"""
System Monitor

This module monitors system health, resources, and trading system status.
"""

import time
import threading
import psutil
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class SystemStatus(Enum):
    """System status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime: timedelta
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemAlert:
    """System alert event."""
    id: str
    level: str
    message: str
    component: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class SystemMonitor:
    """Monitors system health and resources."""
    
    def __init__(self, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize system monitor.
        
        Args:
            notification_manager: Notification manager for alerts
        """
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Monitoring data
        self.system_metrics: List[SystemMetrics] = []
        self.system_alerts: List[SystemAlert] = []
        self.thresholds: Dict[str, float] = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'disk_usage': 90.0,
            'process_count': 1000
        }
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._running = False
        self._start_time = datetime.now()
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'total_alerts': 0,
            'resolved_alerts': 0
        }
        
        self.logger.info("System monitor initialized")
    
    def start(self):
        """Start the system monitor."""
        if self._running:
            self.logger.warning("System monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("System monitor started")
    
    def stop(self):
        """Stop the system monitor."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        self.logger.info("System monitor stopped")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        try:
            metrics = self._get_system_metrics()
            
            if (metrics.cpu_usage > self.thresholds['cpu_usage'] or
                metrics.memory_usage > self.thresholds['memory_usage'] or
                metrics.disk_usage > self.thresholds['disk_usage']):
                return SystemStatus.CRITICAL
            elif (metrics.cpu_usage > self.thresholds['cpu_usage'] * 0.7 or
                  metrics.memory_usage > self.thresholds['memory_usage'] * 0.7):
                return SystemStatus.WARNING
            else:
                return SystemStatus.HEALTHY
                
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return SystemStatus.OFFLINE
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._get_system_metrics()
    
    def get_recent_metrics(self, limit: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics."""
        with self._lock:
            return self.system_metrics[-limit:] if self.system_metrics else []
    
    def get_system_alerts(self, resolved: Optional[bool] = None) -> List[SystemAlert]:
        """Get system alerts."""
        with self._lock:
            alerts = self.system_alerts.copy()
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a system alert."""
        with self._lock:
            for alert in self.system_alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    self.stats['resolved_alerts'] += 1
                    self.logger.info(f"System alert resolved: {alert_id}")
                    return True
        return False
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set a monitoring threshold."""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            self.logger.info(f"System threshold updated: {threshold_name} = {value}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self._get_system_metrics()
                self._store_metrics(metrics)
                self._check_alerts(metrics)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in system monitor loop: {e}", exc_info=True)
                time.sleep(60)  # Wait longer on error
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime = datetime.now() - self._start_time
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(0.0, 0.0, 0.0, {}, 0, timedelta(0))
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store system metrics."""
        with self._lock:
            self.system_metrics.append(metrics)
            self.stats['total_checks'] += 1
            
            # Keep only recent metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check for system alerts."""
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append(SystemAlert(
                id=f"cpu_{int(time.time())}",
                level="CRITICAL" if metrics.cpu_usage > 90 else "WARNING",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                component="CPU",
                value=metrics.cpu_usage,
                threshold=self.thresholds['cpu_usage']
            ))
        
        # Check memory usage
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append(SystemAlert(
                id=f"memory_{int(time.time())}",
                level="CRITICAL" if metrics.memory_usage > 90 else "WARNING",
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                component="Memory",
                value=metrics.memory_usage,
                threshold=self.thresholds['memory_usage']
            ))
        
        # Check disk usage
        if metrics.disk_usage > self.thresholds['disk_usage']:
            alerts.append(SystemAlert(
                id=f"disk_{int(time.time())}",
                level="CRITICAL" if metrics.disk_usage > 95 else "WARNING",
                message=f"High disk usage: {metrics.disk_usage:.1f}%",
                component="Disk",
                value=metrics.disk_usage,
                threshold=self.thresholds['disk_usage']
            ))
        
        # Check process count
        if metrics.process_count > self.thresholds['process_count']:
            alerts.append(SystemAlert(
                id=f"process_{int(time.time())}",
                level="WARNING",
                message=f"High process count: {metrics.process_count}",
                component="Processes",
                value=metrics.process_count,
                threshold=self.thresholds['process_count']
            ))
        
        # Add alerts and send notifications
        for alert in alerts:
            self._add_system_alert(alert)
            self._send_system_notification(alert)
    
    def _add_system_alert(self, alert: SystemAlert):
        """Add a system alert."""
        with self._lock:
            self.system_alerts.append(alert)
            self.stats['total_alerts'] += 1
        
        self.logger.warning(f"System alert: {alert.message}")
    
    def _send_system_notification(self, alert: SystemAlert):
        """Send system notification."""
        if not self.notification_manager:
            return
        
        try:
            self.notification_manager.send_system_alert(
                level=alert.level,
                message=alert.message,
                details=f"Value: {alert.value:.1f}, Threshold: {alert.threshold:.1f}",
                component=alert.component,
                status="Active"
            )
        except Exception as e:
            self.logger.error(f"Failed to send system notification: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system monitor statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'current_status': self.get_system_status().value,
                'active_alerts': len([a for a in self.system_alerts if not a.resolved]),
                'total_alerts': len(self.system_alerts),
                'recent_metrics': len(self.system_metrics),
                'uptime': str(datetime.now() - self._start_time)
            })
            return stats 