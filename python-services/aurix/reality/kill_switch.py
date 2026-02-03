"""
AURIX Kill Switch

Hard stop trading under dangerous conditions to prevent catastrophic loss.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""
    is_active: bool = False
    reason: str = ""
    triggered_at: Optional[datetime] = None
    resume_conditions: List[str] = field(default_factory=list)
    consecutive_failures: int = 0
    current_drawdown_pct: float = 0.0
    avg_confidence: float = 1.0


class KillSwitch:
    """
    Kill Switch - Hard stop trading under dangerous conditions.
    
    Triggers on:
    1. Drawdown breach (exceeds max allowed)
    2. Consecutive failures (too many losses in a row)
    3. Confidence collapse (avg confidence too low)
    
    Clear logging explains WHY trading is stopped.
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 0.08,
        max_consecutive_losses: int = 5,
        min_avg_confidence: float = 0.55,
        confidence_window_size: int = 20,
        auto_resume_enabled: bool = False,
        auto_resume_cooldown_hours: float = 24.0
    ):
        """
        Initialize kill switch.
        
        Args:
            max_drawdown_pct: Maximum drawdown before trigger (e.g., 0.08 = 8%)
            max_consecutive_losses: Max losses in a row before trigger
            min_avg_confidence: Min rolling avg confidence before trigger
            confidence_window_size: Window size for confidence averaging
            auto_resume_enabled: Whether to auto-resume after cooldown
            auto_resume_cooldown_hours: Hours before auto-resume consideration
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.min_avg_confidence = min_avg_confidence
        self.confidence_window_size = confidence_window_size
        self.auto_resume_enabled = auto_resume_enabled
        self.auto_resume_cooldown_hours = auto_resume_cooldown_hours
        
        self._state = KillSwitchState()
        self._confidence_history: List[float] = []
        self._manual_override: bool = False
    
    @property
    def is_active(self) -> bool:
        """Check if kill switch is currently active."""
        return self._state.is_active
    
    @property
    def state(self) -> KillSwitchState:
        """Get current kill switch state."""
        return self._state
    
    def check_all(
        self,
        current_drawdown_pct: float,
        consecutive_losses: int,
        last_confidence: Optional[float] = None
    ) -> KillSwitchState:
        """
        Run all kill switch checks.
        
        Args:
            current_drawdown_pct: Current drawdown as percentage (e.g., 0.05 = 5%)
            consecutive_losses: Current streak of consecutive losses
            last_confidence: Most recent prediction confidence
            
        Returns:
            Updated KillSwitchState
        """
        # Update confidence history
        if last_confidence is not None:
            self._confidence_history.append(last_confidence)
            if len(self._confidence_history) > self.confidence_window_size:
                self._confidence_history = self._confidence_history[-self.confidence_window_size:]
        
        # Calculate rolling average confidence
        avg_confidence = (
            sum(self._confidence_history) / len(self._confidence_history)
            if self._confidence_history else 1.0
        )
        
        # Update state metrics
        self._state.current_drawdown_pct = current_drawdown_pct
        self._state.consecutive_failures = consecutive_losses
        self._state.avg_confidence = avg_confidence
        
        # Check for manual override
        if self._manual_override:
            return self._state
        
        # Check for auto-resume if currently active
        if self._state.is_active and self.auto_resume_enabled:
            if self._check_auto_resume():
                self.reset("Auto-resume after cooldown")
                return self._state
        
        # Run individual checks
        if self.check_drawdown(current_drawdown_pct):
            return self._state
        
        if self.check_consecutive_failures(consecutive_losses):
            return self._state
        
        if self.check_confidence_collapse(avg_confidence):
            return self._state
        
        return self._state
    
    def check_drawdown(self, current_drawdown_pct: float) -> bool:
        """
        Check if drawdown exceeds maximum allowed.
        
        Args:
            current_drawdown_pct: Current drawdown as percentage
            
        Returns:
            True if kill switch triggered
        """
        if current_drawdown_pct >= self.max_drawdown_pct:
            self._trigger(
                f"DRAWDOWN BREACH: {current_drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%} max allowed",
                [
                    f"Reduce drawdown below {self.max_drawdown_pct:.1%}",
                    "Manual reset required"
                ]
            )
            return True
        return False
    
    def check_consecutive_failures(self, count: int) -> bool:
        """
        Check if consecutive losses exceed threshold.
        
        Args:
            count: Number of consecutive losses
            
        Returns:
            True if kill switch triggered
        """
        if count >= self.max_consecutive_losses:
            self._trigger(
                f"CONSECUTIVE LOSSES: {count} >= {self.max_consecutive_losses} max allowed",
                [
                    "Win at least 1 trade to reset counter",
                    "Review recent trade quality"
                ]
            )
            return True
        return False
    
    def check_confidence_collapse(self, avg_confidence: float) -> bool:
        """
        Check if average confidence has collapsed.
        
        Args:
            avg_confidence: Rolling average confidence
            
        Returns:
            True if kill switch triggered
        """
        # Need enough samples to make this determination
        if len(self._confidence_history) < self.confidence_window_size // 2:
            return False
        
        if avg_confidence < self.min_avg_confidence:
            self._trigger(
                f"CONFIDENCE COLLAPSE: {avg_confidence:.1%} avg < {self.min_avg_confidence:.1%} minimum",
                [
                    "Model may need retraining",
                    "Market conditions may have shifted",
                    "Wait for confidence recovery"
                ]
            )
            return True
        return False
    
    def _trigger(self, reason: str, resume_conditions: List[str]):
        """Trigger the kill switch."""
        if self._state.is_active:
            return  # Already triggered
        
        self._state.is_active = True
        self._state.reason = reason
        self._state.triggered_at = datetime.now()
        self._state.resume_conditions = resume_conditions
        
        logger.critical("=" * 60)
        logger.critical("[KILL SWITCH TRIGGERED]")
        logger.critical(f"Reason: {reason}")
        logger.critical("Trading is HALTED. No new positions will be opened.")
        logger.critical("Resume conditions:")
        for i, cond in enumerate(resume_conditions, 1):
            logger.critical(f"  {i}. {cond}")
        logger.critical("=" * 60)
    
    def _check_auto_resume(self) -> bool:
        """Check if conditions are met for auto-resume."""
        if not self._state.triggered_at:
            return False
        
        hours_since_trigger = (datetime.now() - self._state.triggered_at).total_seconds() / 3600
        
        if hours_since_trigger < self.auto_resume_cooldown_hours:
            return False
        
        # Check if conditions have improved
        if self._state.current_drawdown_pct >= self.max_drawdown_pct * 0.8:
            return False  # Still too close to drawdown limit
        
        if self._state.avg_confidence < self.min_avg_confidence:
            return False  # Confidence still too low
        
        return True
    
    def reset(self, reason: str = "Manual reset"):
        """
        Reset the kill switch to allow trading.
        
        Args:
            reason: Reason for reset (for logging)
        """
        if not self._state.is_active:
            logger.info("Kill switch reset called but was not active")
            return
        
        logger.warning("=" * 60)
        logger.warning("[KILL SWITCH RESET]")
        logger.warning(f"Reason: {reason}")
        logger.warning("Trading is RESUMED. New positions may be opened.")
        logger.warning("=" * 60)
        
        self._state = KillSwitchState()
        self._confidence_history = []
    
    def manual_override(self, enable: bool):
        """
        Set manual override to prevent auto-triggering.
        
        WARNING: Use with extreme caution. Disabling kill switch
        removes critical protection against catastrophic loss.
        
        Args:
            enable: True to enable override (disable kill switch)
        """
        if enable:
            logger.warning("[DANGER] Kill switch manual override ENABLED")
            logger.warning("All safety checks are disabled. Trade at your own risk.")
        else:
            logger.info("Kill switch manual override disabled")
        
        self._manual_override = enable
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        lines = [
            "=== Kill Switch Status ===",
            f"Active: {'YES - TRADING HALTED' if self._state.is_active else 'No'}",
            f"Current Drawdown: {self._state.current_drawdown_pct:.2%} (max: {self.max_drawdown_pct:.2%})",
            f"Consecutive Losses: {self._state.consecutive_failures} (max: {self.max_consecutive_losses})",
            f"Avg Confidence: {self._state.avg_confidence:.2%} (min: {self.min_avg_confidence:.2%})",
        ]
        
        if self._state.is_active:
            lines.append(f"Triggered At: {self._state.triggered_at}")
            lines.append(f"Reason: {self._state.reason}")
            lines.append("Resume Conditions:")
            for cond in self._state.resume_conditions:
                lines.append(f"  - {cond}")
        
        return "\n".join(lines)
