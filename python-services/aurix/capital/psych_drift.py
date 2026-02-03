"""
AURIX Psychological Drift Proxy

Infers trader mental state from trading behavior patterns.
Since we can't measure actual psychology, we use behavioral proxies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DriftLevel(Enum):
    """Psychological drift severity levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftState:
    """Current psychological drift state."""
    overall_score: float  # 0-1, higher = more drift
    level: DriftLevel
    impatience_score: float  # Trades closed before TP/SL
    stubbornness_score: float  # Trades held past SL
    hesitation_score: float  # High-confidence signals not taken
    overconfidence_score: float  # Low-confidence signals taken
    inconsistency_score: float  # Deviation from strategy rules
    confidence_modifier: float  # Multiplier for confidence threshold
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'overall_score': self.overall_score,
            'level': self.level.value,
            'impatience_score': self.impatience_score,
            'stubbornness_score': self.stubbornness_score,
            'hesitation_score': self.hesitation_score,
            'overconfidence_score': self.overconfidence_score,
            'inconsistency_score': self.inconsistency_score,
            'confidence_modifier': self.confidence_modifier,
            'recommended_action': self.recommended_action,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TradeWithContext:
    """Trade with context for psychological analysis."""
    timestamp: datetime
    direction: str
    signal_confidence: float  # What confidence triggered the trade
    entry_price: float
    exit_price: float
    take_profit: float
    stop_loss: float
    exit_reason: str  # TP, SL, SIGNAL, MANUAL
    hold_time_minutes: float
    planned_hold_minutes: float  # Expected hold time
    pnl: float
    was_winner: bool


@dataclass
class SkippedSignal:
    """Record of a signal that was not taken."""
    timestamp: datetime
    direction: str
    confidence: float
    reason: str  # Why skipped (cooldown, manual, etc.)
    hypothetical_pnl: Optional[float] = None  # What would have happened


class PsychDriftDetector:
    """
    Detects psychological drift from trading behavior.
    
    Drift Signals:
    - Impatience: Closing trades early (before TP/SL)
    - Stubbornness: Holding past stop loss
    - Hesitation: Not taking high-confidence signals
    - Overconfidence: Taking low-confidence signals
    - Inconsistency: Deviation from strategy rules
    """
    
    # Drift score weights
    WEIGHTS = {
        'impatience': 0.20,
        'stubbornness': 0.25,
        'hesitation': 0.20,
        'overconfidence': 0.20,
        'inconsistency': 0.15
    }
    
    # Thresholds
    THRESHOLD_MODERATE = 0.4
    THRESHOLD_HIGH = 0.7
    
    def __init__(
        self,
        lookback_trades: int = 30,
        min_confidence_threshold: float = 0.65,
        high_confidence_threshold: float = 0.75
    ):
        """
        Initialize detector.
        
        Args:
            lookback_trades: Number of trades to analyze
            min_confidence_threshold: Threshold below which trades are "low confidence"
            high_confidence_threshold: Threshold above which signals are "high confidence"
        """
        self.lookback = lookback_trades
        self.min_conf = min_confidence_threshold
        self.high_conf = high_confidence_threshold
        
        # History
        self.trades: List[TradeWithContext] = []
        self.skipped_signals: List[SkippedSignal] = []
        
        # Baseline metrics
        self.baseline_hold_time: Optional[float] = None
        self.baseline_exit_ratio: Optional[Dict[str, float]] = None
    
    def add_trade(self, trade: TradeWithContext):
        """Add a completed trade."""
        self.trades.append(trade)
        
        if len(self.trades) > self.lookback * 2:
            self.trades = self.trades[-self.lookback:]
        
        self._update_baselines()
    
    def add_skipped_signal(self, signal: SkippedSignal):
        """Record a signal that was not taken."""
        self.skipped_signals.append(signal)
        
        # Keep only recent
        cutoff = datetime.now() - timedelta(days=7)
        self.skipped_signals = [s for s in self.skipped_signals if s.timestamp > cutoff]
    
    def calculate_drift(self) -> DriftState:
        """
        Calculate current psychological drift state.
        
        Returns:
            DriftState with all component scores
        """
        if len(self.trades) < 5:
            return DriftState(
                overall_score=0.0,
                level=DriftLevel.NONE,
                impatience_score=0.0,
                stubbornness_score=0.0,
                hesitation_score=0.0,
                overconfidence_score=0.0,
                inconsistency_score=0.0,
                confidence_modifier=1.0,
                recommended_action="Insufficient data"
            )
        
        # Calculate component scores
        impatience = self._calculate_impatience()
        stubbornness = self._calculate_stubbornness()
        hesitation = self._calculate_hesitation()
        overconfidence = self._calculate_overconfidence()
        inconsistency = self._calculate_inconsistency()
        
        # Weighted overall score
        overall = (
            impatience * self.WEIGHTS['impatience'] +
            stubbornness * self.WEIGHTS['stubbornness'] +
            hesitation * self.WEIGHTS['hesitation'] +
            overconfidence * self.WEIGHTS['overconfidence'] +
            inconsistency * self.WEIGHTS['inconsistency']
        )
        
        # Determine level
        if overall >= self.THRESHOLD_HIGH:
            level = DriftLevel.HIGH
        elif overall >= self.THRESHOLD_MODERATE:
            level = DriftLevel.MODERATE
        elif overall > 0.2:
            level = DriftLevel.LOW
        else:
            level = DriftLevel.NONE
        
        # Calculate confidence modifier
        conf_mod = self._calculate_confidence_modifier(overall, level)
        
        # Recommended action
        action = self._get_recommended_action(level, impatience, stubbornness, hesitation, overconfidence, inconsistency)
        
        state = DriftState(
            overall_score=overall,
            level=level,
            impatience_score=impatience,
            stubbornness_score=stubbornness,
            hesitation_score=hesitation,
            overconfidence_score=overconfidence,
            inconsistency_score=inconsistency,
            confidence_modifier=conf_mod,
            recommended_action=action
        )
        
        if level in [DriftLevel.MODERATE, DriftLevel.HIGH]:
            logger.warning(f"Psychological drift detected: {level.value} ({overall:.2f})")
        
        return state
    
    def get_confidence_adjustment(self) -> Tuple[float, str]:
        """
        Get confidence threshold adjustment based on drift.
        
        Returns:
            Tuple of (adjustment_amount, reason)
        """
        state = self.calculate_drift()
        
        if state.level == DriftLevel.HIGH:
            return 0.10, f"HIGH drift ({state.overall_score:.2f})"
        elif state.level == DriftLevel.MODERATE:
            return 0.05, f"MODERATE drift ({state.overall_score:.2f})"
        else:
            return 0.0, "Normal"
    
    def _update_baselines(self):
        """Update baseline metrics."""
        if len(self.trades) < 10:
            return
        
        recent = self.trades[-10:]
        
        # Baseline hold time
        self.baseline_hold_time = np.mean([t.hold_time_minutes for t in recent])
        
        # Baseline exit ratio
        exit_counts = {}
        for t in recent:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        
        total = len(recent)
        self.baseline_exit_ratio = {k: v / total for k, v in exit_counts.items()}
    
    def _calculate_impatience(self) -> float:
        """
        Calculate impatience score.
        
        Indicators:
        - Closing trades before TP or SL hit (MANUAL exits)
        - Average hold time decreasing
        - Closing winners early (below TP)
        """
        if len(self.trades) < 5:
            return 0.0
        
        recent = self.trades[-10:]
        
        # Manual exits (neither TP nor SL)
        manual_exits = sum(1 for t in recent if t.exit_reason == 'MANUAL')
        manual_ratio = manual_exits / len(recent)
        
        # Early winner exits (exited positive but below TP)
        early_winner_exits = 0
        for t in recent:
            if t.was_winner and t.direction == 'LONG':
                if t.exit_price < t.take_profit * 0.8:  # Exited below 80% of TP
                    early_winner_exits += 1
            elif t.was_winner and t.direction == 'SHORT':
                if t.exit_price > t.take_profit * 1.2:  # Similar for shorts
                    early_winner_exits += 1
        
        early_ratio = early_winner_exits / len(recent)
        
        # Hold time decreasing
        if self.baseline_hold_time and self.baseline_hold_time > 0:
            current_avg = np.mean([t.hold_time_minutes for t in recent[-5:]])
            hold_decline = max(0, 1 - current_avg / self.baseline_hold_time)
        else:
            hold_decline = 0
        
        return min(1.0, manual_ratio * 0.4 + early_ratio * 0.4 + hold_decline * 0.2)
    
    def _calculate_stubbornness(self) -> float:
        """
        Calculate stubbornness score.
        
        Indicators:
        - Holding past stop loss
        - Large average loss (relative to planned)
        - Holding losers too long
        """
        if len(self.trades) < 5:
            return 0.0
        
        recent = self.trades[-10:]
        losers = [t for t in recent if not t.was_winner]
        
        if not losers:
            return 0.0
        
        # Trades that held past SL
        past_sl_count = 0
        for t in losers:
            if t.direction == 'LONG' and t.exit_price < t.stop_loss:
                past_sl_count += 1
            elif t.direction == 'SHORT' and t.exit_price > t.stop_loss:
                past_sl_count += 1
        
        past_sl_ratio = past_sl_count / len(losers) if losers else 0
        
        # Average loss larger than expected from SL
        expected_losses = []
        actual_losses = []
        for t in losers:
            if t.direction == 'LONG':
                expected = t.entry_price - t.stop_loss
            else:
                expected = t.stop_loss - t.entry_price
            
            actual = abs(t.pnl)
            expected_losses.append(expected)
            actual_losses.append(actual)
        
        if expected_losses and sum(expected_losses) > 0:
            loss_ratio = sum(actual_losses) / sum(expected_losses)
            excess_loss = max(0, loss_ratio - 1)  # How much worse than expected
        else:
            excess_loss = 0
        
        return min(1.0, past_sl_ratio * 0.6 + excess_loss * 0.4)
    
    def _calculate_hesitation(self) -> float:
        """
        Calculate hesitation score.
        
        Indicators:
        - High-confidence signals that were skipped
        - Profitable signals that were not taken
        """
        if not self.skipped_signals:
            return 0.0
        
        # High-confidence signals skipped
        high_conf_skipped = [
            s for s in self.skipped_signals 
            if s.confidence >= self.high_conf and s.reason not in ['cooldown', 'risk_limit']
        ]
        
        if not high_conf_skipped:
            return 0.0
        
        # What percentage of skipped high-conf signals would have been profitable?
        profitable_skipped = sum(
            1 for s in high_conf_skipped 
            if s.hypothetical_pnl and s.hypothetical_pnl > 0
        )
        
        profitable_ratio = profitable_skipped / len(high_conf_skipped) if high_conf_skipped else 0
        
        # Ratio of high-conf signals skipped to total signals
        total_high_conf = len(high_conf_skipped)
        total_trades = len(self.trades)
        
        skip_ratio = total_high_conf / (total_trades + total_high_conf) if (total_trades + total_high_conf) > 0 else 0
        
        return min(1.0, skip_ratio * 0.5 + profitable_ratio * 0.5)
    
    def _calculate_overconfidence(self) -> float:
        """
        Calculate overconfidence score.
        
        Indicators:
        - Taking low-confidence signals
        - Taking trades during cooldown/unfavorable conditions
        """
        if len(self.trades) < 5:
            return 0.0
        
        recent = self.trades[-10:]
        
        # Trades taken at low confidence
        low_conf_trades = [t for t in recent if t.signal_confidence < self.min_conf]
        low_conf_ratio = len(low_conf_trades) / len(recent)
        
        # What was the win rate on low-conf trades?
        if low_conf_trades:
            low_conf_win_rate = sum(1 for t in low_conf_trades if t.was_winner) / len(low_conf_trades)
            # If win rate is low, overconfidence is confirmed
            overconf_penalty = 1 - low_conf_win_rate
        else:
            overconf_penalty = 0
        
        return min(1.0, low_conf_ratio * 0.6 + overconf_penalty * 0.4)
    
    def _calculate_inconsistency(self) -> float:
        """
        Calculate inconsistency score.
        
        Indicators:
        - Variance in hold times
        - Variance in position sizes
        - Exit reason distribution change
        """
        if len(self.trades) < 10:
            return 0.0
        
        recent = self.trades[-10:]
        
        # Hold time variance
        hold_times = [t.hold_time_minutes for t in recent]
        if np.mean(hold_times) > 0:
            hold_cv = np.std(hold_times) / np.mean(hold_times)  # Coefficient of variation
        else:
            hold_cv = 0
        
        # Position size variance (would need to track this)
        # For now, use exit reason consistency
        
        # Exit reason distribution vs baseline
        if self.baseline_exit_ratio:
            current_exits = {}
            for t in recent:
                current_exits[t.exit_reason] = current_exits.get(t.exit_reason, 0) + 1
            
            total = len(recent)
            current_ratio = {k: v / total for k, v in current_exits.items()}
            
            # Calculate difference from baseline
            diff = 0
            for reason in set(list(self.baseline_exit_ratio.keys()) + list(current_ratio.keys())):
                baseline_val = self.baseline_exit_ratio.get(reason, 0)
                current_val = current_ratio.get(reason, 0)
                diff += abs(baseline_val - current_val)
            
            exit_inconsistency = min(1.0, diff)
        else:
            exit_inconsistency = 0
        
        return min(1.0, hold_cv * 0.5 + exit_inconsistency * 0.5)
    
    def _calculate_confidence_modifier(self, overall: float, level: DriftLevel) -> float:
        """Calculate confidence threshold modifier."""
        if level == DriftLevel.HIGH:
            return 0.90  # Effectively +10% threshold
        elif level == DriftLevel.MODERATE:
            return 0.95  # Effectively +5% threshold
        else:
            return 1.0
    
    def _get_recommended_action(
        self,
        level: DriftLevel,
        impatience: float,
        stubbornness: float,
        hesitation: float,
        overconfidence: float,
        inconsistency: float
    ) -> str:
        """Get recommended action based on drift components."""
        if level == DriftLevel.NONE:
            return "Continue normal operation"
        
        # Find dominant drift factor
        factors = [
            ('impatience', impatience, "Let trades run to TP/SL"),
            ('stubbornness', stubbornness, "Respect stop losses"),
            ('hesitation', hesitation, "Trust high-confidence signals"),
            ('overconfidence', overconfidence, "Avoid low-confidence trades"),
            ('inconsistency', inconsistency, "Follow strategy rules consistently")
        ]
        
        dominant = max(factors, key=lambda x: x[1])
        
        if level == DriftLevel.HIGH:
            return f"CRITICAL: {dominant[2]}. Consider taking a break."
        elif level == DriftLevel.MODERATE:
            return f"WARNING: {dominant[2]}"
        else:
            return f"Note: {dominant[2]}"
    
    def get_summary(self) -> Dict:
        """Get summary of psychological drift detection."""
        state = self.calculate_drift()
        
        return {
            'trade_count': len(self.trades),
            'skipped_signals': len(self.skipped_signals),
            'current_state': state.to_dict(),
            'baseline_hold_time': self.baseline_hold_time,
            'baseline_exit_ratio': self.baseline_exit_ratio
        }
