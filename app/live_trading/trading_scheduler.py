"""
Trading Scheduler

This module manages trading hours, schedules, and market timing.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
import pytz

from app.notifications.manager import NotificationManager
from app.util import get_logger

logger = get_logger(__name__)


class MarketStatus(Enum):
    """Market status."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class TradingSession:
    """Trading session configuration."""
    name: str
    start_time: dt_time
    end_time: dt_time
    timezone: str
    enabled: bool = True


@dataclass
class MarketHoliday:
    """Market holiday."""
    date: datetime
    name: str
    description: str


class TradingScheduler:
    """Manages trading schedules and market timing."""
    
    def __init__(self, notification_manager: Optional[NotificationManager] = None):
        """
        Initialize trading scheduler.
        
        Args:
            notification_manager: Notification manager for alerts
        """
        self.notification_manager = notification_manager
        self.logger = get_logger(__name__)
        
        # Trading sessions
        self.trading_sessions: Dict[str, TradingSession] = {}
        self.market_holidays: List[MarketHoliday] = []
        
        # Callbacks
        self.on_market_open: Optional[Callable] = None
        self.on_market_close: Optional[Callable] = None
        self.on_session_change: Optional[Callable] = None
        
        # Threading
        self._lock = threading.Lock()
        self._scheduler_thread = None
        self._running = False
        self._current_status = MarketStatus.CLOSED
        
        # Configuration
        self.timezone = pytz.UTC
        self.weekend_trading = False
        self.holiday_trading = False
        
        # Statistics
        self.stats = {
            'market_opens': 0,
            'market_closes': 0,
            'session_changes': 0
        }
        
        self._setup_default_sessions()
        self.logger.info("Trading scheduler initialized")
    
    def _setup_default_sessions(self):
        """Setup default trading sessions."""
        # Forex market sessions (24/5)
        self.add_trading_session(
            "forex_sydney",
            dt_time(22, 0),  # 22:00 UTC
            dt_time(7, 0),   # 07:00 UTC
            "UTC"
        )
        
        self.add_trading_session(
            "forex_tokyo",
            dt_time(0, 0),   # 00:00 UTC
            dt_time(9, 0),   # 09:00 UTC
            "UTC"
        )
        
        self.add_trading_session(
            "forex_london",
            dt_time(8, 0),   # 08:00 UTC
            dt_time(17, 0),  # 17:00 UTC
            "UTC"
        )
        
        self.add_trading_session(
            "forex_newyork",
            dt_time(13, 0),  # 13:00 UTC
            dt_time(22, 0),  # 22:00 UTC
            "UTC"
        )
    
    def add_trading_session(self, name: str, start_time: dt_time, end_time: dt_time, 
                          timezone: str = "UTC", enabled: bool = True):
        """Add a trading session."""
        session = TradingSession(
            name=name,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
            enabled=enabled
        )
        
        with self._lock:
            self.trading_sessions[name] = session
        
        self.logger.info(f"Trading session added: {name} ({start_time} - {end_time})")
    
    def remove_trading_session(self, name: str) -> bool:
        """Remove a trading session."""
        with self._lock:
            if name in self.trading_sessions:
                del self.trading_sessions[name]
                self.logger.info(f"Trading session removed: {name}")
                return True
        return False
    
    def add_market_holiday(self, date: datetime, name: str, description: str = ""):
        """Add a market holiday."""
        holiday = MarketHoliday(
            date=date,
            name=name,
            description=description
        )
        
        with self._lock:
            self.market_holidays.append(holiday)
        
        self.logger.info(f"Market holiday added: {name} on {date.date()}")
    
    def remove_market_holiday(self, date: datetime) -> bool:
        """Remove a market holiday."""
        with self._lock:
            for i, holiday in enumerate(self.market_holidays):
                if holiday.date.date() == date.date():
                    del self.market_holidays[i]
                    self.logger.info(f"Market holiday removed: {holiday.name}")
                    return True
        return False
    
    def start(self):
        """Start the trading scheduler."""
        if self._running:
            self.logger.warning("Trading scheduler already running")
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TradingScheduler"
        )
        self._scheduler_thread.start()
        self.logger.info("Trading scheduler started")
    
    def stop(self):
        """Stop the trading scheduler."""
        if not self._running:
            return
        
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        
        self.logger.info("Trading scheduler stopped")
    
    def get_market_status(self) -> MarketStatus:
        """Get current market status."""
        now = datetime.now(self.timezone)
        
        # Check if it's weekend
        if now.weekday() >= 5 and not self.weekend_trading:
            return MarketStatus.WEEKEND
        
        # Check if it's a holiday
        if self._is_market_holiday(now) and not self.holiday_trading:
            return MarketStatus.HOLIDAY
        
        # Check if any trading session is active
        if self._is_market_open(now):
            return MarketStatus.OPEN
        else:
            return MarketStatus.CLOSED
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        return self.get_market_status() == MarketStatus.OPEN
    
    def get_next_market_open(self) -> Optional[datetime]:
        """Get next market open time."""
        now = datetime.now(self.timezone)
        
        # Check next 7 days
        for i in range(7):
            check_date = now + timedelta(days=i)
            
            # Skip weekends if weekend trading is disabled
            if check_date.weekday() >= 5 and not self.weekend_trading:
                continue
            
            # Skip holidays if holiday trading is disabled
            if self._is_market_holiday(check_date) and not self.holiday_trading:
                continue
            
            # Check each session
            for session in self.trading_sessions.values():
                if not session.enabled:
                    continue
                
                session_tz = pytz.timezone(session.timezone)
                session_start = session_tz.localize(
                    datetime.combine(check_date.date(), session.start_time)
                )
                
                if session_start > now:
                    return session_start
        
        return None
    
    def get_next_market_close(self) -> Optional[datetime]:
        """Get next market close time."""
        now = datetime.now(self.timezone)
        
        # Check current day and next few days
        for i in range(7):
            check_date = now + timedelta(days=i)
            
            # Check each session
            for session in self.trading_sessions.values():
                if not session.enabled:
                    continue
                
                session_tz = pytz.timezone(session.timezone)
                session_end = session_tz.localize(
                    datetime.combine(check_date.date(), session.end_time)
                )
                
                if session_end > now:
                    return session_end
        
        return None
    
    def get_trading_sessions(self) -> Dict[str, TradingSession]:
        """Get all trading sessions."""
        with self._lock:
            return self.trading_sessions.copy()
    
    def get_market_holidays(self) -> List[MarketHoliday]:
        """Get all market holidays."""
        with self._lock:
            return self.market_holidays.copy()
    
    def set_timezone(self, timezone: str):
        """Set the timezone for the scheduler."""
        try:
            self.timezone = pytz.timezone(timezone)
            self.logger.info(f"Timezone set to: {timezone}")
        except Exception as e:
            self.logger.error(f"Invalid timezone: {timezone}")
    
    def enable_weekend_trading(self, enabled: bool = True):
        """Enable or disable weekend trading."""
        self.weekend_trading = enabled
        self.logger.info(f"Weekend trading {'enabled' if enabled else 'disabled'}")
    
    def enable_holiday_trading(self, enabled: bool = True):
        """Enable or disable holiday trading."""
        self.holiday_trading = enabled
        self.logger.info(f"Holiday trading {'enabled' if enabled else 'disabled'}")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                old_status = self._current_status
                new_status = self.get_market_status()
                
                # Check for status changes
                if new_status != old_status:
                    self._handle_status_change(old_status, new_status)
                    self._current_status = new_status
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                time.sleep(120)  # Wait longer on error
    
    def _handle_status_change(self, old_status: MarketStatus, new_status: MarketStatus):
        """Handle market status changes."""
        self.stats['session_changes'] += 1
        
        if new_status == MarketStatus.OPEN and old_status != MarketStatus.OPEN:
            self.stats['market_opens'] += 1
            self.logger.info("Market opened")
            
            if self.on_market_open:
                try:
                    self.on_market_open()
                except Exception as e:
                    self.logger.error(f"Error in market open callback: {e}")
            
            self._send_market_notification("Market Opened", "Trading is now active")
            
        elif old_status == MarketStatus.OPEN and new_status != MarketStatus.OPEN:
            self.stats['market_closes'] += 1
            self.logger.info("Market closed")
            
            if self.on_market_close:
                try:
                    self.on_market_close()
                except Exception as e:
                    self.logger.error(f"Error in market close callback: {e}")
            
            self._send_market_notification("Market Closed", "Trading is now inactive")
        
        if self.on_session_change:
            try:
                self.on_session_change(old_status, new_status)
            except Exception as e:
                self.logger.error(f"Error in session change callback: {e}")
    
    def _is_market_open(self, check_time: datetime) -> bool:
        """Check if market is open at the given time."""
        for session in self.trading_sessions.values():
            if not session.enabled:
                continue
            
            session_tz = pytz.timezone(session.timezone)
            session_start = session_tz.localize(
                datetime.combine(check_time.date(), session.start_time)
            )
            session_end = session_tz.localize(
                datetime.combine(check_time.date(), session.end_time)
            )
            
            # Handle sessions that span midnight
            if session_end < session_start:
                session_end += timedelta(days=1)
            
            if session_start <= check_time <= session_end:
                return True
        
        return False
    
    def _is_market_holiday(self, check_date: datetime) -> bool:
        """Check if the given date is a market holiday."""
        for holiday in self.market_holidays:
            if holiday.date.date() == check_date.date():
                return True
        return False
    
    def _send_market_notification(self, title: str, message: str):
        """Send market notification."""
        if not self.notification_manager:
            return
        
        try:
            self.notification_manager.send_system_alert(
                level="INFO",
                message=title,
                details=message,
                component="Trading Scheduler",
                status="Active"
            )
        except Exception as e:
            self.logger.error(f"Failed to send market notification: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading scheduler statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'current_status': self._current_status.value,
                'trading_sessions': len(self.trading_sessions),
                'market_holidays': len(self.market_holidays),
                'weekend_trading': self.weekend_trading,
                'holiday_trading': self.holiday_trading,
                'timezone': str(self.timezone)
            })
            return stats 