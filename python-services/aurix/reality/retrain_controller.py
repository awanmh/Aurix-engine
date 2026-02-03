"""
AURIX Retrain Controller

Disciplined model retraining - only retrain when truly necessary.
Prevents over-adaptation and frequent model switching.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrainRecord:
    """Record of a model retrain event."""
    version: str
    timestamp: datetime
    reason: str
    train_accuracy: float
    validation_accuracy: float
    sample_count: int


@dataclass
class RetrainDecision:
    """Decision from retrain controller."""
    should_retrain: bool
    reason: str
    cooldown_remaining_hours: float
    performance_delta: float
    regime_stable: bool


class RetrainController:
    """
    Retrain Controller
    
    Enforces disciplined retraining to prevent over-adaptation.
    
    Retrain is only allowed when:
    1. Minimum cooldown has passed since last retrain
    2. Regime change is sustained (not temporary)
    3. Performance decay exceeds threshold
    4. Max retrains per month not exceeded
    
    Model weights are LOCKED between retrain cycles.
    """
    
    def __init__(
        self,
        min_cooldown_days: int = 7,
        performance_decay_threshold: float = 0.10,
        regime_confirmation_bars: int = 48,
        max_retrains_per_month: int = 4
    ):
        """
        Initialize retrain controller.
        
        Args:
            min_cooldown_days: Minimum days between retrains
            performance_decay_threshold: Performance drop that triggers retrain
            regime_confirmation_bars: Bars to confirm regime change
            max_retrains_per_month: Maximum retrains allowed per month
        """
        self.min_cooldown_days = min_cooldown_days
        self.decay_threshold = performance_decay_threshold
        self.regime_confirmation_bars = regime_confirmation_bars
        self.max_retrains_per_month = max_retrains_per_month
        
        self._last_retrain: Optional[datetime] = None
        self._retrain_history: List[RetrainRecord] = []
        self._model_locked: bool = True  # Start locked
        self._peak_performance: float = 0.0
        self._current_performance: float = 0.0
        self._regime_change_bar_count: int = 0
        self._last_regime: Optional[str] = None
    
    @property
    def is_model_locked(self) -> bool:
        """Check if model is currently locked (no retraining)."""
        return self._model_locked
    
    def check_retrain_needed(
        self,
        current_regime: str,
        current_accuracy: float,
        current_auc: float = 0.5
    ) -> RetrainDecision:
        """
        Check if model retraining is needed.
        
        Args:
            current_regime: Current detected market regime
            current_accuracy: Current rolling accuracy
            current_auc: Current rolling AUC-ROC
            
        Returns:
            RetrainDecision with recommendation
        """
        # Combined performance metric
        current_perf = (current_accuracy + current_auc) / 2
        self._current_performance = current_perf
        
        # Update peak
        if current_perf > self._peak_performance:
            self._peak_performance = current_perf
        
        # Check cooldown
        cooldown_remaining = self._get_cooldown_remaining_hours()
        
        # Check monthly limit
        retrains_this_month = self._get_retrains_this_month()
        
        # Check regime change
        regime_stable = self._check_regime_stability(current_regime)
        
        # Calculate performance delta from peak
        perf_delta = self._peak_performance - current_perf
        
        # Decision logic
        should_retrain = False
        reason = ""
        
        # Can't retrain if in cooldown
        if cooldown_remaining > 0:
            reason = f"In cooldown period ({cooldown_remaining:.1f}h remaining)"
            return RetrainDecision(
                should_retrain=False,
                reason=reason,
                cooldown_remaining_hours=cooldown_remaining,
                performance_delta=perf_delta,
                regime_stable=regime_stable
            )
        
        # Can't retrain if monthly limit exceeded
        if retrains_this_month >= self.max_retrains_per_month:
            reason = f"Monthly limit reached ({retrains_this_month}/{self.max_retrains_per_month})"
            return RetrainDecision(
                should_retrain=False,
                reason=reason,
                cooldown_remaining_hours=0,
                performance_delta=perf_delta,
                regime_stable=regime_stable
            )
        
        # Check performance decay
        if perf_delta >= self.decay_threshold:
            should_retrain = True
            reason = f"Performance decay: {perf_delta:.1%} from peak"
            logger.warning(f"Retrain recommended: {reason}")
        
        # Check sustained regime change
        if not regime_stable and self._regime_change_bar_count >= self.regime_confirmation_bars:
            should_retrain = True
            reason = f"Sustained regime change: {self._last_regime} -> {current_regime}"
            logger.warning(f"Retrain recommended: {reason}")
        
        if not should_retrain:
            reason = "No retrain needed - performance stable"
        
        return RetrainDecision(
            should_retrain=should_retrain,
            reason=reason,
            cooldown_remaining_hours=cooldown_remaining,
            performance_delta=perf_delta,
            regime_stable=regime_stable
        )
    
    def _get_cooldown_remaining_hours(self) -> float:
        """Get remaining cooldown time in hours."""
        if self._last_retrain is None:
            return 0.0
        
        cooldown_end = self._last_retrain + timedelta(days=self.min_cooldown_days)
        remaining = cooldown_end - datetime.now()
        
        if remaining.total_seconds() <= 0:
            return 0.0
        
        return remaining.total_seconds() / 3600
    
    def _get_retrains_this_month(self) -> int:
        """Count retrains in the current month."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return sum(
            1 for r in self._retrain_history
            if r.timestamp >= month_start
        )
    
    def _check_regime_stability(self, current_regime: str) -> bool:
        """Check if regime is stable or changing."""
        if self._last_regime is None:
            self._last_regime = current_regime
            return True
        
        if current_regime == self._last_regime:
            self._regime_change_bar_count = 0
            return True
        else:
            self._regime_change_bar_count += 1
            
            # Only update last regime once change is confirmed
            if self._regime_change_bar_count >= self.regime_confirmation_bars:
                self._last_regime = current_regime
            
            return False
    
    def lock_model(self):
        """Lock model to prevent retraining."""
        self._model_locked = True
        logger.info("Model weights LOCKED - retraining disabled")
    
    def unlock_model(self):
        """Unlock model to allow retraining."""
        self._model_locked = False
        logger.info("Model weights UNLOCKED - retraining enabled")
    
    def record_retrain(
        self,
        version: str,
        reason: str,
        train_accuracy: float,
        validation_accuracy: float,
        sample_count: int
    ):
        """
        Record a retrain event.
        
        Args:
            version: Model version string
            reason: Reason for retrain
            train_accuracy: Training accuracy achieved
            validation_accuracy: Validation accuracy achieved
            sample_count: Number of training samples
        """
        record = RetrainRecord(
            version=version,
            timestamp=datetime.now(),
            reason=reason,
            train_accuracy=train_accuracy,
            validation_accuracy=validation_accuracy,
            sample_count=sample_count
        )
        
        self._retrain_history.append(record)
        self._last_retrain = datetime.now()
        
        # Reset peak performance after retrain
        self._peak_performance = (train_accuracy + validation_accuracy) / 2
        
        # Keep history manageable
        if len(self._retrain_history) > 100:
            self._retrain_history = self._retrain_history[-50:]
        
        logger.info(f"Recorded retrain: version={version}, reason={reason}")
        logger.info(f"  Train acc: {train_accuracy:.2%}, Val acc: {validation_accuracy:.2%}")
        
        # Lock model after retrain
        self.lock_model()
    
    def force_retrain(self, reason: str = "Manual trigger"):
        """
        Force unlock model for immediate retrain.
        
        Use with caution - bypasses cooldown checks.
        
        Args:
            reason: Reason for forced retrain
        """
        logger.warning(f"FORCED RETRAIN: {reason}")
        logger.warning("Bypassing cooldown and lock checks")
        self.unlock_model()
    
    def get_retrain_history(self, limit: int = 10) -> List[RetrainRecord]:
        """Get recent retrain history."""
        return self._retrain_history[-limit:]
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        lines = [
            "=== Retrain Controller Status ===",
            f"Model Locked: {'YES' if self._model_locked else 'No'}",
            f"Last Retrain: {self._last_retrain or 'Never'}",
            f"Cooldown Remaining: {self._get_cooldown_remaining_hours():.1f} hours",
            f"Retrains This Month: {self._get_retrains_this_month()}/{self.max_retrains_per_month}",
            f"Current Performance: {self._current_performance:.2%}",
            f"Peak Performance: {self._peak_performance:.2%}",
            f"Performance Delta: {self._peak_performance - self._current_performance:.2%}",
            f"Current Regime: {self._last_regime or 'Unknown'}",
            f"Regime Change Bars: {self._regime_change_bar_count}",
        ]
        
        if self._retrain_history:
            last = self._retrain_history[-1]
            lines.append(f"Last Retrain Version: {last.version}")
            lines.append(f"Last Retrain Reason: {last.reason}")
        
        return "\n".join(lines)
