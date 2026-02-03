"""
AURIX Reality Layer Configuration

Centralized configuration for all anti-overfitting thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RealityConfig:
    """
    Configuration for Reality Validation & Anti-Overfitting Layer.
    
    All thresholds are tuned for CONSERVATIVE behavior.
    Better to trade less and survive than trade more and blow up.
    """
    
    # ==================== Overfitting Detection ====================
    # Maximum allowed divergence between train and forward accuracy
    max_train_forward_divergence: float = 0.15  # 15%
    
    # AUC drop threshold that triggers warning
    auc_collapse_threshold: float = 0.10  # 10% AUC drop
    
    # Confidence penalty when overfitting detected (0.0 - 0.5)
    overfit_confidence_penalty: float = 0.20
    
    # Number of forward samples required before divergence check
    min_forward_samples: int = 50
    
    # ==================== Retrain Discipline ====================
    # Minimum days between retrains (to prevent over-adaptation)
    min_retrain_cooldown_days: int = 7
    
    # Performance decay that triggers retrain consideration
    performance_decay_threshold: float = 0.10  # 10% drop from peak
    
    # Bars to confirm regime change before allowing retrain
    regime_change_confirmation_bars: int = 48  # 12 hours at 15m
    
    # Maximum retrains per month
    max_retrains_per_month: int = 4
    
    # ==================== Stress Testing ====================
    # Enable stress testing during backtest
    enable_stress_testing: bool = True
    
    # Stress intensity (0.0 = none, 1.0 = maximum)
    stress_intensity: float = 0.5
    
    # Wick noise as percentage of ATR
    wick_noise_pct: float = 0.10
    
    # Probability of gap injection per candle
    gap_probability: float = 0.005
    
    # Execution delay range (milliseconds)
    min_execution_delay_ms: int = 50
    max_execution_delay_ms: int = 500
    
    # Slippage multiplier during high volatility
    high_volatility_slippage_multiplier: float = 3.0
    
    # ==================== Kill Switch ====================
    # Maximum drawdown before hard stop
    max_drawdown_pct: float = 0.08  # 8%
    
    # Consecutive losses before hard stop
    max_consecutive_losses: int = 5
    
    # Minimum average confidence (below = confidence collapse)
    min_avg_confidence: float = 0.55
    
    # Rolling window for confidence average (trades)
    confidence_window_size: int = 20
    
    # Auto-resume after kill switch (False = manual only)
    auto_resume_enabled: bool = False
    
    # Hours to wait before auto-resume consideration
    auto_resume_cooldown_hours: float = 24.0
    
    # ==================== Transaction Costs ====================
    # Taker fee (market orders)
    taker_fee_pct: float = 0.075  # 0.075% Binance
    
    # Maker fee (limit orders)
    maker_fee_pct: float = 0.025
    
    # Default spread as percentage of price
    default_spread_pct: float = 0.01  # 1 bps
    
    # Base slippage for normal volatility
    base_slippage_pct: float = 0.02  # 2 bps
    
    # ==================== Data Guard ====================
    # Lookahead tolerance (0 = zero tolerance)
    max_lookahead_tolerance_sec: int = 0
    
    # Minimum data quality score (0-1) to proceed
    min_data_quality_score: float = 0.95
    
    # Maximum allowed gap in data (candles)
    max_allowed_gap_candles: int = 3
    
    # ==================== Reality Score ====================
    # Soft threshold for position size reduction
    soft_threshold_position: float = 0.7  # Below: reduce position
    
    # Soft threshold for frequency reduction
    soft_threshold_frequency: float = 0.6  # Below: reduce trade frequency
    
    # Hard threshold - triggers kill switch
    hard_threshold: float = 0.4  # Below: halt trading
    
    # Component weights (must sum to 1.0)
    weight_data_quality: float = 0.20
    weight_slippage: float = 0.15
    weight_stress: float = 0.20
    weight_overfit: float = 0.25
    weight_confidence: float = 0.20
    
    # ==================== Recovery Protocol ====================
    # Cooldown hours after halt
    recovery_cooldown_hours: float = 12.0
    
    # Validation trades required
    recovery_validation_trades: int = 10
    
    # Position size during validation (0-1)
    recovery_validation_position_pct: float = 0.10
    
    # Minimum win rate to pass validation
    recovery_validation_min_winrate: float = 0.50
    
    # Hours per ramp-up stage
    recovery_rampup_hours_per_stage: float = 6.0
    
    def validate(self) -> bool:
        """Validate configuration values are within acceptable ranges."""
        errors = []
        
        if not 0 <= self.max_train_forward_divergence <= 0.5:
            errors.append("max_train_forward_divergence must be 0-0.5")
        
        if not 0 <= self.overfit_confidence_penalty <= 0.5:
            errors.append("overfit_confidence_penalty must be 0-0.5")
        
        if self.min_retrain_cooldown_days < 1:
            errors.append("min_retrain_cooldown_days must be >= 1")
        
        if not 0 <= self.stress_intensity <= 1:
            errors.append("stress_intensity must be 0-1")
        
        if not 0 <= self.max_drawdown_pct <= 0.20:
            errors.append("max_drawdown_pct must be 0-0.20")
        
        if self.max_consecutive_losses < 3:
            errors.append("max_consecutive_losses must be >= 3")
        
        if errors:
            for e in errors:
                logger.error(f"Config validation error: {e}")
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "overfitting": {
                "max_divergence": self.max_train_forward_divergence,
                "auc_collapse_threshold": self.auc_collapse_threshold,
                "confidence_penalty": self.overfit_confidence_penalty,
            },
            "retrain": {
                "cooldown_days": self.min_retrain_cooldown_days,
                "decay_threshold": self.performance_decay_threshold,
                "max_per_month": self.max_retrains_per_month,
            },
            "stress": {
                "enabled": self.enable_stress_testing,
                "intensity": self.stress_intensity,
            },
            "kill_switch": {
                "max_drawdown": self.max_drawdown_pct,
                "max_consecutive_losses": self.max_consecutive_losses,
                "min_confidence": self.min_avg_confidence,
            },
            "costs": {
                "taker_fee": self.taker_fee_pct,
                "maker_fee": self.maker_fee_pct,
                "spread": self.default_spread_pct,
                "base_slippage": self.base_slippage_pct,
            }
        }


# Default conservative configuration
DEFAULT_REALITY_CONFIG = RealityConfig()
