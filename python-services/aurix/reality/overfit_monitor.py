"""
AURIX Overfitting Monitor

Detects performance divergence between train/validation/forward windows
to identify overfitting before it causes real losses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowMetrics:
    """Metrics for a single evaluation window."""
    accuracy: float = 0.0
    auc_roc: float = 0.5
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0
    sample_count: int = 0
    recorded_at: datetime = field(default_factory=datetime.now)


@dataclass
class OverfitState:
    """Current overfitting detection state."""
    train_metrics: Optional[WindowMetrics] = None
    validation_metrics: Optional[WindowMetrics] = None
    forward_metrics: Optional[WindowMetrics] = None
    
    divergence: float = 0.0  # Train accuracy - Forward accuracy
    auc_drop: float = 0.0    # Train AUC - Forward AUC
    is_overfitting: bool = False
    confidence_penalty: float = 0.0
    should_freeze_model: bool = False
    warning_message: str = ""


class OverfitMonitor:
    """
    Overfitting Monitor
    
    Tracks performance across training, validation, and forward windows.
    Detects overfitting when forward performance diverges from training.
    
    Key metrics:
    1. Accuracy divergence (train - forward)
    2. AUC collapse (significant drop after retrain)
    3. Performance decay over time
    
    Actions:
    - Apply confidence penalty when overfitting detected
    - Recommend model freeze when severe overfitting
    """
    
    def __init__(
        self,
        max_train_forward_divergence: float = 0.15,
        auc_collapse_threshold: float = 0.10,
        overfit_confidence_penalty: float = 0.20,
        min_forward_samples: int = 50,
        history_size: int = 10
    ):
        """
        Initialize overfitting monitor.
        
        Args:
            max_train_forward_divergence: Max allowed train-forward accuracy gap
            auc_collapse_threshold: AUC drop threshold for warning
            overfit_confidence_penalty: Confidence reduction when overfitting
            min_forward_samples: Min samples before divergence check
            history_size: Number of historical metrics to keep
        """
        self.max_divergence = max_train_forward_divergence
        self.auc_threshold = auc_collapse_threshold
        self.confidence_penalty = overfit_confidence_penalty
        self.min_forward_samples = min_forward_samples
        self.history_size = history_size
        
        self._state = OverfitState()
        self._train_history: List[WindowMetrics] = []
        self._forward_history: List[WindowMetrics] = []
    
    @property
    def state(self) -> OverfitState:
        """Get current overfitting state."""
        return self._state
    
    def record_train_metrics(
        self,
        accuracy: float,
        auc_roc: float,
        profit_factor: float = 0.0,
        sharpe_ratio: float = 0.0,
        expectancy: float = 0.0,
        sample_count: int = 0
    ):
        """
        Record metrics from training window.
        
        Args:
            accuracy: Training accuracy
            auc_roc: Training AUC-ROC
            profit_factor: Training profit factor
            sharpe_ratio: Training Sharpe ratio
            expectancy: Training expectancy per trade
            sample_count: Number of training samples
        """
        metrics = WindowMetrics(
            accuracy=accuracy,
            auc_roc=auc_roc,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            expectancy=expectancy,
            sample_count=sample_count
        )
        
        self._state.train_metrics = metrics
        self._train_history.append(metrics)
        
        if len(self._train_history) > self.history_size:
            self._train_history = self._train_history[-self.history_size:]
        
        logger.info(f"Recorded train metrics: acc={accuracy:.2%}, AUC={auc_roc:.3f}")
    
    def record_validation_metrics(
        self,
        accuracy: float,
        auc_roc: float,
        profit_factor: float = 0.0,
        sharpe_ratio: float = 0.0,
        expectancy: float = 0.0,
        sample_count: int = 0
    ):
        """Record metrics from validation window."""
        metrics = WindowMetrics(
            accuracy=accuracy,
            auc_roc=auc_roc,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            expectancy=expectancy,
            sample_count=sample_count
        )
        
        self._state.validation_metrics = metrics
        logger.info(f"Recorded validation metrics: acc={accuracy:.2%}, AUC={auc_roc:.3f}")
    
    def record_forward_metrics(
        self,
        accuracy: float,
        auc_roc: float,
        profit_factor: float = 0.0,
        sharpe_ratio: float = 0.0,
        expectancy: float = 0.0,
        sample_count: int = 0
    ):
        """Record metrics from forward (out-of-sample) window."""
        metrics = WindowMetrics(
            accuracy=accuracy,
            auc_roc=auc_roc,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            expectancy=expectancy,
            sample_count=sample_count
        )
        
        self._state.forward_metrics = metrics
        self._forward_history.append(metrics)
        
        if len(self._forward_history) > self.history_size:
            self._forward_history = self._forward_history[-self.history_size:]
        
        logger.info(f"Recorded forward metrics: acc={accuracy:.2%}, AUC={auc_roc:.3f}")
        
        # Auto-check after forward metrics update
        self.check_divergence()
    
    def check_divergence(self) -> OverfitState:
        """
        Check for overfitting based on train/forward divergence.
        
        Returns:
            Updated OverfitState
        """
        train = self._state.train_metrics
        forward = self._state.forward_metrics
        
        # Need both metrics to compare
        if train is None or forward is None:
            return self._state
        
        # Need enough forward samples
        if forward.sample_count < self.min_forward_samples:
            logger.debug(f"Insufficient forward samples: {forward.sample_count} < {self.min_forward_samples}")
            return self._state
        
        # Calculate divergence
        acc_divergence = train.accuracy - forward.accuracy
        auc_drop = train.auc_roc - forward.auc_roc
        
        self._state.divergence = acc_divergence
        self._state.auc_drop = auc_drop
        
        # Check for overfitting
        is_overfitting = False
        warnings = []
        
        if acc_divergence > self.max_divergence:
            is_overfitting = True
            warnings.append(
                f"Accuracy divergence: train={train.accuracy:.1%} vs forward={forward.accuracy:.1%} "
                f"(gap={acc_divergence:.1%} > {self.max_divergence:.1%} threshold)"
            )
        
        if auc_drop > self.auc_threshold:
            is_overfitting = True
            warnings.append(
                f"AUC collapse: train={train.auc_roc:.3f} vs forward={forward.auc_roc:.3f} "
                f"(drop={auc_drop:.3f} > {self.auc_threshold:.3f} threshold)"
            )
        
        # Check for severe overfitting (should freeze model)
        severe_overfit = acc_divergence > self.max_divergence * 1.5 or auc_drop > self.auc_threshold * 1.5
        
        self._state.is_overfitting = is_overfitting
        self._state.should_freeze_model = severe_overfit
        self._state.confidence_penalty = self.confidence_penalty if is_overfitting else 0.0
        self._state.warning_message = " | ".join(warnings)
        
        if is_overfitting:
            logger.warning("=" * 60)
            logger.warning("[OVERFITTING DETECTED]")
            for w in warnings:
                logger.warning(f"  {w}")
            logger.warning(f"Applying confidence penalty: -{self.confidence_penalty:.1%}")
            if severe_overfit:
                logger.warning("SEVERE: Recommend freezing model until retrain")
            logger.warning("=" * 60)
        
        return self._state
    
    def get_confidence_penalty(self) -> float:
        """
        Get confidence penalty to apply to predictions.
        
        Returns:
            Penalty to subtract from confidence (0.0 if no overfitting)
        """
        return self._state.confidence_penalty
    
    def should_freeze_model(self) -> bool:
        """
        Check if model should be frozen (no trading).
        
        Returns:
            True if severe overfitting detected
        """
        return self._state.should_freeze_model
    
    def get_performance_trend(self) -> str:
        """
        Analyze performance trend across history.
        
        Returns:
            Trend description: 'improving', 'stable', 'degrading'
        """
        if len(self._forward_history) < 3:
            return "insufficient_data"
        
        recent = self._forward_history[-3:]
        accuracies = [m.accuracy for m in recent]
        
        trend = accuracies[-1] - accuracies[0]
        
        if trend > 0.02:
            return "improving"
        elif trend < -0.02:
            return "degrading"
        else:
            return "stable"
    
    def reset(self):
        """Reset overfitting monitor state."""
        self._state = OverfitState()
        logger.info("Overfitting monitor reset")
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        lines = [
            "=== Overfitting Monitor Status ===",
            f"Overfitting Detected: {'YES' if self._state.is_overfitting else 'No'}",
        ]
        
        if self._state.train_metrics:
            lines.append(f"Train Accuracy: {self._state.train_metrics.accuracy:.2%}")
            lines.append(f"Train AUC: {self._state.train_metrics.auc_roc:.3f}")
        
        if self._state.forward_metrics:
            lines.append(f"Forward Accuracy: {self._state.forward_metrics.accuracy:.2%}")
            lines.append(f"Forward AUC: {self._state.forward_metrics.auc_roc:.3f}")
            lines.append(f"Forward Samples: {self._state.forward_metrics.sample_count}")
        
        lines.append(f"Divergence: {self._state.divergence:.2%}")
        lines.append(f"AUC Drop: {self._state.auc_drop:.3f}")
        lines.append(f"Confidence Penalty: {self._state.confidence_penalty:.1%}")
        lines.append(f"Performance Trend: {self.get_performance_trend()}")
        
        if self._state.warning_message:
            lines.append(f"Warning: {self._state.warning_message}")
        
        return "\n".join(lines)
