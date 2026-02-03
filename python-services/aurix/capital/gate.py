"""
AURIX Capital Efficiency Gate

Unified gating logic that combines all efficiency checks:
- Capital Efficiency Score
- Pair Manager
- Overtrading Detection
- Psychological Drift
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .scorer import CapitalEfficiencyScorer, EfficiencyScore
from .pair_manager import PairManager, PairStatus
from .overtrading import OvertradingDetector, OvertradingAlert
from .psych_drift import PsychDriftDetector, DriftState, DriftLevel

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of the capital efficiency gate check."""
    approved: bool
    reason: str
    confidence_modifier: float  # Multiply base confidence by this
    size_modifier: float  # Multiply base position size by this
    pair_weight: float  # Additional pair-specific weight
    
    # Component details
    ces_score: Optional[float] = None
    pair_rank: Optional[int] = None
    drift_level: Optional[str] = None
    overtrading_alerts: int = 0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'approved': self.approved,
            'reason': self.reason,
            'confidence_modifier': self.confidence_modifier,
            'size_modifier': self.size_modifier,
            'pair_weight': self.pair_weight,
            'ces_score': self.ces_score,
            'pair_rank': self.pair_rank,
            'drift_level': self.drift_level,
            'overtrading_alerts': self.overtrading_alerts,
            'timestamp': self.timestamp.isoformat()
        }


class CapitalEfficiencyGate:
    """
    Unified gate for all capital efficiency checks.
    
    Combines:
    1. Capital Efficiency Score (CES) - minimum threshold
    2. Pair Manager - is pair active and ranked high enough
    3. Overtrading Detection - no active cooldowns or alerts
    4. Psychological Drift - mental state factor
    
    Final confidence adjustment:
        adjusted_confidence = base_confidence * ces_mod * drift_mod * overtrading_mod
        
    Final size adjustment:
        adjusted_size = base_size * ces_size_mod * pair_weight * overtrading_size_mod
    """
    
    def __init__(
        self,
        efficiency_scorer: CapitalEfficiencyScorer,
        pair_manager: PairManager,
        overtrading_detector: OvertradingDetector,
        psych_drift_detector: PsychDriftDetector,
        min_ces_score: float = 0.4,
        enable_ces: bool = True,
        enable_pair_filter: bool = True,
        enable_overtrading: bool = True,
        enable_psych_drift: bool = True
    ):
        """
        Initialize the gate.
        
        Args:
            efficiency_scorer: Capital efficiency scorer instance
            pair_manager: Pair manager instance
            overtrading_detector: Overtrading detector instance
            psych_drift_detector: Psychological drift detector
            min_ces_score: Minimum CES required to trade
            enable_*: Flags to enable/disable each component
        """
        self.scorer = efficiency_scorer
        self.pair_manager = pair_manager
        self.overtrading = overtrading_detector
        self.psych_drift = psych_drift_detector
        
        self.min_ces = min_ces_score
        self.enable_ces = enable_ces
        self.enable_pairs = enable_pair_filter
        self.enable_overtrading = enable_overtrading
        self.enable_psych = enable_psych_drift
    
    def should_trade(
        self,
        symbol: str,
        base_confidence: float,
        base_size: float
    ) -> GateResult:
        """
        Main entry point: check if a trade should be allowed.
        
        Args:
            symbol: Trading pair symbol
            base_confidence: ML model's raw confidence
            base_size: Calculated position size before adjustments
            
        Returns:
            GateResult with approval status and modifiers
        """
        # Initialize modifiers
        conf_mod = 1.0
        size_mod = 1.0
        pair_weight = 1.0
        ces_score = None
        pair_rank = None
        drift_level = None
        overtrading_alerts = 0
        
        reasons = []
        
        # 1. Check Capital Efficiency Score
        if self.enable_ces:
            ces_result = self._check_ces()
            if not ces_result[0]:
                return GateResult(
                    approved=False,
                    reason=ces_result[1],
                    confidence_modifier=0,
                    size_modifier=0,
                    pair_weight=0,
                    ces_score=ces_result[2]
                )
            conf_mod *= ces_result[3]
            size_mod *= ces_result[4]
            ces_score = ces_result[2]
        
        # 2. Check Pair Manager
        if self.enable_pairs:
            pair_result = self._check_pair(symbol)
            if not pair_result[0]:
                return GateResult(
                    approved=False,
                    reason=pair_result[1],
                    confidence_modifier=0,
                    size_modifier=0,
                    pair_weight=0,
                    ces_score=ces_score
                )
            pair_weight = pair_result[2]
            pair_rank = pair_result[3]
        
        # 3. Check Overtrading
        if self.enable_overtrading:
            ot_result = self._check_overtrading()
            if not ot_result[0]:
                return GateResult(
                    approved=False,
                    reason=ot_result[1],
                    confidence_modifier=0,
                    size_modifier=0,
                    pair_weight=0,
                    ces_score=ces_score,
                    pair_rank=pair_rank,
                    overtrading_alerts=ot_result[4]
                )
            conf_mod *= ot_result[2]
            size_mod *= ot_result[3]
            overtrading_alerts = ot_result[4]
        
        # 4. Check Psychological Drift
        if self.enable_psych:
            drift_result = self._check_psych_drift()
            conf_mod *= drift_result[0]
            drift_level = drift_result[1]
            if drift_result[2]:
                reasons.append(drift_result[2])
        
        # Compile result
        reason = "Approved"
        if reasons:
            reason = f"Approved with caveats: {'; '.join(reasons)}"
        
        result = GateResult(
            approved=True,
            reason=reason,
            confidence_modifier=conf_mod,
            size_modifier=size_mod,
            pair_weight=pair_weight,
            ces_score=ces_score,
            pair_rank=pair_rank,
            drift_level=drift_level,
            overtrading_alerts=overtrading_alerts
        )
        
        logger.debug(f"Gate result for {symbol}: {result}")
        
        return result
    
    def _check_ces(self) -> Tuple[bool, str, Optional[float], float, float]:
        """
        Check capital efficiency score.
        
        Returns:
            Tuple of (allowed, reason, score, conf_mod, size_mod)
        """
        score = self.scorer.calculate_score()
        
        if score is None:
            # No data yet, allow with default modifiers
            return True, "CES: Insufficient data", None, 1.0, 1.0
        
        if score.composite < self.min_ces:
            return (
                False,
                f"CES too low: {score.composite:.2f} < {self.min_ces}",
                score.composite,
                0,
                0
            )
        
        return (
            True,
            "CES OK",
            score.composite,
            1.0,  # CES doesn't modify confidence
            score.position_multiplier  # But does affect size
        )
    
    def _check_pair(self, symbol: str) -> Tuple[bool, str, float, Optional[int]]:
        """
        Check pair ranking and status.
        
        Returns:
            Tuple of (allowed, reason, weight, rank)
        """
        can_trade, reason = self.pair_manager.can_trade(symbol)
        
        if not can_trade:
            return False, f"Pair check failed: {reason}", 0, None
        
        # Get pair weight and rank
        weight = self.pair_manager.get_position_weight(symbol)
        
        if symbol in self.pair_manager.pairs:
            rank = self.pair_manager.pairs[symbol].rank
        else:
            rank = None
        
        return True, "Pair OK", weight, rank
    
    def _check_overtrading(self) -> Tuple[bool, str, float, float, int]:
        """
        Check for overtrading patterns.
        
        Returns:
            Tuple of (allowed, reason, conf_mod, size_mod, alert_count)
        """
        allowed, reason, conf_mod, size_mod = self.overtrading.should_trade()
        alerts = self.overtrading.check_all_patterns()
        
        return allowed, reason, conf_mod, size_mod, len(alerts)
    
    def _check_psych_drift(self) -> Tuple[float, str, Optional[str]]:
        """
        Check psychological drift.
        
        Returns:
            Tuple of (conf_mod, drift_level, warning_message)
        """
        state = self.psych_drift.calculate_drift()
        
        conf_mod = state.confidence_modifier
        level = state.level.value
        
        warning = None
        if state.level in [DriftLevel.MODERATE, DriftLevel.HIGH]:
            warning = state.recommended_action
        
        return conf_mod, level, warning
    
    def get_full_summary(self) -> Dict:
        """Get complete summary of all gate components."""
        return {
            'ces': self.scorer.get_efficiency_summary(),
            'pairs': self.pair_manager.get_summary(),
            'overtrading': self.overtrading.get_summary(),
            'psych_drift': self.psych_drift.get_summary(),
            'config': {
                'min_ces_score': self.min_ces,
                'ces_enabled': self.enable_ces,
                'pair_filter_enabled': self.enable_pairs,
                'overtrading_enabled': self.enable_overtrading,
                'psych_drift_enabled': self.enable_psych
            }
        }
    
    def apply_modifiers(
        self,
        base_confidence: float,
        base_size: float,
        gate_result: GateResult
    ) -> Tuple[float, float]:
        """
        Apply gate result modifiers to confidence and size.
        
        Args:
            base_confidence: Original ML confidence
            base_size: Original calculated position size
            gate_result: Result from should_trade()
            
        Returns:
            Tuple of (adjusted_confidence, adjusted_size)
        """
        if not gate_result.approved:
            return 0, 0
        
        adjusted_conf = base_confidence * gate_result.confidence_modifier
        adjusted_size = (
            base_size * 
            gate_result.size_modifier * 
            gate_result.pair_weight
        )
        
        return adjusted_conf, adjusted_size


def create_default_gate(
    window_days: int = 30,
    max_active_pairs: int = 5,
    max_trades_per_day: int = 10
) -> CapitalEfficiencyGate:
    """
    Create a gate with default configuration.
    
    Args:
        window_days: Rolling window for CES calculation
        max_active_pairs: Maximum actively traded pairs
        max_trades_per_day: Maximum trades allowed per day
        
    Returns:
        Configured CapitalEfficiencyGate
    """
    scorer = CapitalEfficiencyScorer(window_days=window_days)
    pair_manager = PairManager(max_active_pairs=max_active_pairs)
    overtrading = OvertradingDetector(max_trades_per_day=max_trades_per_day)
    psych_drift = PsychDriftDetector()
    
    return CapitalEfficiencyGate(
        efficiency_scorer=scorer,
        pair_manager=pair_manager,
        overtrading_detector=overtrading,
        psych_drift_detector=psych_drift
    )
