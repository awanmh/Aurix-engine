"""
AURIX Overtrading Detector

Detects profit compression, overtrading, revenge trading, and drawdown spirals.
Implements automatic cooldowns and position size adjustments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of overtrading alerts."""
    PROFIT_COMPRESSION = "profit_compression"
    OVERTRADING = "overtrading"
    REVENGE_TRADING = "revenge_trading"
    WIN_CHASING = "win_chasing"
    DRAWDOWN_SPIRAL = "drawdown_spiral"
    SIZE_CREEP = "size_creep"


class AlertSeverity(Enum):
    """Severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class OvertradingAlert:
    """Alert for potential overtrading behavior."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    action: str  # Recommended action
    confidence_modifier: float  # Multiplier for confidence threshold
    size_modifier: float  # Multiplier for position size
    cooldown_minutes: int  # Suggested cooldown
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'action': self.action,
            'confidence_modifier': self.confidence_modifier,
            'size_modifier': self.size_modifier,
            'cooldown_minutes': self.cooldown_minutes,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TradeEvent:
    """Simplified trade event for pattern detection."""
    timestamp: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    position_size: float
    hold_time_minutes: float
    was_winner: bool
    exit_reason: str  # TP, SL, SIGNAL, MANUAL


class OvertradingDetector:
    """
    Detects overtrading patterns and recommends actions.
    
    Patterns Detected:
    - Profit Compression: Decreasing average win size
    - Overtrading: Trade count > 2σ above mean
    - Revenge Trading: Loss → immediate new trade
    - Win Chasing: Multiple wins → oversized position
    - Drawdown Spiral: 3+ losses with increasing size
    - Size Creep: Gradual position size increases
    """
    
    def __init__(
        self,
        max_trades_per_day: int = 10,
        min_time_between_trades_minutes: int = 15,
        compression_threshold: float = 0.3,
        lookback_trades: int = 50
    ):
        """
        Initialize detector.
        
        Args:
            max_trades_per_day: Maximum trades allowed per day
            min_time_between_trades_minutes: Minimum time between trades
            compression_threshold: Threshold for profit compression detection
            lookback_trades: Number of trades to analyze
        """
        self.max_trades_day = max_trades_per_day
        self.min_time_between = min_time_between_trades_minutes
        self.compression_threshold = compression_threshold
        self.lookback = lookback_trades
        
        # Trade history
        self.trades: List[TradeEvent] = []
        
        # Cooldown state
        self.cooldown_until: Optional[datetime] = None
        self.cooldown_reason: str = ""
        
        # Baseline metrics (established during normal trading)
        self.baseline_trade_count_24h: Optional[float] = None
        self.baseline_win_size: Optional[float] = None
        self.baseline_position_size: Optional[float] = None
    
    def add_trade(self, trade: TradeEvent):
        """Add a completed trade to history."""
        self.trades.append(trade)
        
        # Keep only recent trades
        if len(self.trades) > self.lookback * 2:
            self.trades = self.trades[-self.lookback:]
        
        # Update baselines if we have enough data
        self._update_baselines()
    
    def check_all_patterns(self) -> List[OvertradingAlert]:
        """
        Check for all overtrading patterns.
        
        Returns:
            List of detected alerts
        """
        alerts = []
        
        if len(self.trades) < 5:
            return alerts
        
        # Check each pattern
        patterns = [
            self._check_profit_compression,
            self._check_overtrading,
            self._check_revenge_trading,
            self._check_win_chasing,
            self._check_drawdown_spiral,
            self._check_size_creep
        ]
        
        for check_fn in patterns:
            alert = check_fn()
            if alert:
                alerts.append(alert)
                logger.warning(f"Overtrading alert: {alert.alert_type.value} - {alert.message}")
        
        return alerts
    
    def should_trade(self) -> Tuple[bool, str, float, float]:
        """
        Check if trading should be allowed.
        
        Returns:
            Tuple of (allowed, reason, confidence_modifier, size_modifier)
        """
        # Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
            return False, f"Cooldown active ({remaining:.0f}m): {self.cooldown_reason}", 0, 0
        
        # Check minimum time between trades
        if self.trades:
            last_trade = self.trades[-1]
            minutes_since = (datetime.now() - last_trade.timestamp).total_seconds() / 60
            if minutes_since < self.min_time_between:
                return False, f"Too soon after last trade ({minutes_since:.0f}m)", 0, 0
        
        # Check all patterns
        alerts = self.check_all_patterns()
        
        if not alerts:
            return True, "OK", 1.0, 1.0
        
        # Aggregate modifiers from alerts
        worst_confidence_mod = 1.0
        worst_size_mod = 1.0
        
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                # Set cooldown
                self.set_cooldown(alert.cooldown_minutes, alert.message)
                return False, f"Blocked: {alert.message}", 0, 0
            
            worst_confidence_mod = min(worst_confidence_mod, alert.confidence_modifier)
            worst_size_mod = min(worst_size_mod, alert.size_modifier)
        
        return True, "OK with modifiers", worst_confidence_mod, worst_size_mod
    
    def set_cooldown(self, minutes: int, reason: str):
        """Set a trading cooldown."""
        self.cooldown_until = datetime.now() + timedelta(minutes=minutes)
        self.cooldown_reason = reason
        logger.warning(f"Cooldown set for {minutes}m: {reason}")
    
    def clear_cooldown(self):
        """Clear any active cooldown."""
        self.cooldown_until = None
        self.cooldown_reason = ""
    
    def _update_baselines(self):
        """Update baseline metrics from stable trading periods."""
        if len(self.trades) < 20:
            return
        
        recent = self.trades[-20:]
        
        # Baseline trade count (trades per 24h)
        time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds() / 86400
        if time_span > 0:
            self.baseline_trade_count_24h = len(recent) / time_span
        
        # Baseline win size
        winners = [t for t in recent if t.was_winner]
        if winners:
            self.baseline_win_size = np.mean([t.pnl for t in winners])
        
        # Baseline position size
        self.baseline_position_size = np.mean([t.position_size for t in recent])
    
    def _check_profit_compression(self) -> Optional[OvertradingAlert]:
        """
        Detect decreasing average win size over time.
        
        This often indicates deteriorating edge or overtrading.
        """
        if len(self.trades) < 10:
            return None
        
        winners = [t for t in self.trades if t.was_winner]
        if len(winners) < 5:
            return None
        
        # Compare recent wins to older wins
        mid = len(winners) // 2
        old_wins = winners[:mid]
        new_wins = winners[mid:]
        
        old_avg = np.mean([t.pnl for t in old_wins])
        new_avg = np.mean([t.pnl for t in new_wins])
        
        if old_avg <= 0:
            return None
        
        compression = (old_avg - new_avg) / old_avg
        
        if compression > self.compression_threshold:
            return OvertradingAlert(
                alert_type=AlertType.PROFIT_COMPRESSION,
                severity=AlertSeverity.WARNING,
                message=f"Win size compressed by {compression:.0%}",
                action="Raise confidence threshold",
                confidence_modifier=0.95,  # +5% threshold
                size_modifier=0.9,
                cooldown_minutes=0
            )
        
        return None
    
    def _check_overtrading(self) -> Optional[OvertradingAlert]:
        """
        Detect trade count significantly above normal.
        """
        if self.baseline_trade_count_24h is None:
            return None
        
        # Count trades in last 24h
        cutoff = datetime.now() - timedelta(hours=24)
        recent_count = sum(1 for t in self.trades if t.timestamp > cutoff)
        
        # Check if > 2 standard deviations above baseline
        threshold = self.baseline_trade_count_24h * 1.5  # Simple threshold
        
        if recent_count > self.max_trades_day:
            return OvertradingAlert(
                alert_type=AlertType.OVERTRADING,
                severity=AlertSeverity.CRITICAL,
                message=f"Trade count {recent_count} exceeds max {self.max_trades_day}",
                action="Pause new entries",
                confidence_modifier=0,
                size_modifier=0,
                cooldown_minutes=240  # 4 hour cooldown
            )
        
        if recent_count > threshold:
            return OvertradingAlert(
                alert_type=AlertType.OVERTRADING,
                severity=AlertSeverity.WARNING,
                message=f"Trade count {recent_count} above baseline {threshold:.1f}",
                action="Reduce trade frequency",
                confidence_modifier=0.9,  # +10% threshold
                size_modifier=0.85,
                cooldown_minutes=0
            )
        
        return None
    
    def _check_revenge_trading(self) -> Optional[OvertradingAlert]:
        """
        Detect trading immediately after a loss.
        """
        if len(self.trades) < 2:
            return None
        
        last_trade = self.trades[-1]
        prev_trade = self.trades[-2]
        
        # Check if last trade was a loss
        if not prev_trade.was_winner:
            # Check time between trades
            gap_minutes = (last_trade.timestamp - prev_trade.timestamp).total_seconds() / 60
            
            # Traded within 5 minutes of a loss = revenge trading
            if gap_minutes < 5:
                return OvertradingAlert(
                    alert_type=AlertType.REVENGE_TRADING,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Traded {gap_minutes:.0f}m after loss",
                    action="Block trade, 30min cooldown",
                    confidence_modifier=0,
                    size_modifier=0,
                    cooldown_minutes=30
                )
            
            # Traded within 15 minutes = potential revenge
            if gap_minutes < 15:
                return OvertradingAlert(
                    alert_type=AlertType.REVENGE_TRADING,
                    severity=AlertSeverity.WARNING,
                    message=f"Quick trade {gap_minutes:.0f}m after loss",
                    action="Reduce size",
                    confidence_modifier=0.9,
                    size_modifier=0.7,
                    cooldown_minutes=0
                )
        
        return None
    
    def _check_win_chasing(self) -> Optional[OvertradingAlert]:
        """
        Detect oversized positions after winning streaks.
        """
        if len(self.trades) < 4:
            return None
        
        recent = self.trades[-4:]
        
        # Check for 3+ consecutive wins
        win_streak = sum(1 for t in recent[:-1] if t.was_winner)
        
        if win_streak >= 3:
            # Check if latest trade has larger position
            last_size = recent[-1].position_size
            avg_size = np.mean([t.position_size for t in recent[:-1]])
            
            if last_size > avg_size * 1.3:  # 30% larger
                return OvertradingAlert(
                    alert_type=AlertType.WIN_CHASING,
                    severity=AlertSeverity.WARNING,
                    message=f"Position size increased {(last_size/avg_size - 1):.0%} after win streak",
                    action="Cap position to 75% normal",
                    confidence_modifier=1.0,
                    size_modifier=0.75,
                    cooldown_minutes=0
                )
        
        return None
    
    def _check_drawdown_spiral(self) -> Optional[OvertradingAlert]:
        """
        Detect pattern of consecutive losses with increasing size.
        """
        if len(self.trades) < 3:
            return None
        
        recent = self.trades[-5:]
        
        # Find consecutive losses
        consecutive_losses = 0
        increasing_size = True
        prev_size = 0
        
        for trade in reversed(recent):
            if not trade.was_winner:
                consecutive_losses += 1
                if prev_size > 0 and trade.position_size <= prev_size:
                    increasing_size = False
                prev_size = trade.position_size
            else:
                break
        
        if consecutive_losses >= 3 and increasing_size:
            return OvertradingAlert(
                alert_type=AlertType.DRAWDOWN_SPIRAL,
                severity=AlertSeverity.CRITICAL,
                message=f"{consecutive_losses} losses with increasing size",
                action="Reduce size 50%, pause 2h",
                confidence_modifier=0,
                size_modifier=0.5,
                cooldown_minutes=120
            )
        
        if consecutive_losses >= 3:
            return OvertradingAlert(
                alert_type=AlertType.DRAWDOWN_SPIRAL,
                severity=AlertSeverity.WARNING,
                message=f"{consecutive_losses} consecutive losses",
                action="Reduce size",
                confidence_modifier=0.9,
                size_modifier=0.7,
                cooldown_minutes=0
            )
        
        return None
    
    def _check_size_creep(self) -> Optional[OvertradingAlert]:
        """
        Detect gradual increase in position size over time.
        """
        if len(self.trades) < 10 or self.baseline_position_size is None:
            return None
        
        recent_avg = np.mean([t.position_size for t in self.trades[-5:]])
        
        # Check if recent average is significantly above baseline
        creep_ratio = recent_avg / self.baseline_position_size
        
        if creep_ratio > 1.5:  # 50% above baseline
            return OvertradingAlert(
                alert_type=AlertType.SIZE_CREEP,
                severity=AlertSeverity.WARNING,
                message=f"Position size {creep_ratio:.0%} of baseline",
                action="Reset to baseline size",
                confidence_modifier=1.0,
                size_modifier=1 / creep_ratio,  # Return to baseline
                cooldown_minutes=0
            )
        
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of overtrading detection state."""
        cutoff_24h = datetime.now() - timedelta(hours=24)
        trades_24h = [t for t in self.trades if t.timestamp > cutoff_24h]
        
        win_rate = 0
        if trades_24h:
            win_rate = sum(1 for t in trades_24h if t.was_winner) / len(trades_24h)
        
        return {
            'total_trades_tracked': len(self.trades),
            'trades_24h': len(trades_24h),
            'win_rate_24h': win_rate,
            'baseline_trades_24h': self.baseline_trade_count_24h,
            'baseline_win_size': self.baseline_win_size,
            'baseline_position_size': self.baseline_position_size,
            'cooldown_active': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'cooldown_reason': self.cooldown_reason,
            'active_alerts': [a.to_dict() for a in self.check_all_patterns()]
        }
