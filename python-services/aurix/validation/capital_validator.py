"""
AURIX Capital Validation Mode

Validates real-time trading behavior before live capital deployment.
Runs on testnet for 14+ days to build trust in the system.

Features:
- Locked model logic (no learning changes during validation)
- Equity Curve Analytics (slope, volatility, max DD, recovery time)
- Live vs Backtest Expectancy Drift tracking
- Capital Trust Score (CTS) based on stability metrics
- Equity curve degradation pattern detection

Goal: Ensure system behaves consistently with backtest results
before risking real capital.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ValidationPhase(Enum):
    """Phases of capital validation."""
    WARMUP = "warmup"           # First 3 days - gathering baseline
    OBSERVATION = "observation" # Days 4-10 - tracking metrics
    CONFIRMATION = "confirmation"  # Days 11-14 - confirming stability
    COMPLETE = "complete"       # Validation finished


class DegradationPattern(Enum):
    """Types of equity curve degradation."""
    NONE = "none"
    CHOP = "chop"           # Oscillating around breakeven, high trade count
    OVERTRADE = "overtrade" # Too many trades, death by fees
    BLEED = "bleed"         # Slow consistent losses
    SPIKE_LOSS = "spike_loss"  # Sudden large drawdown
    STAGNATION = "stagnation"  # No meaningful gains, flat equity


class TrustLevel(Enum):
    """Capital trust levels."""
    UNTRUSTED = "untrusted"     # Not validated
    LOW = "low"                 # CTS < 40
    MEDIUM = "medium"           # CTS 40-60
    HIGH = "high"               # CTS 60-80
    VALIDATED = "validated"     # CTS > 80 after 14 days


@dataclass
class EquityCurveMetrics:
    """Metrics computed from equity curve."""
    # Basic stats
    total_return_pct: float
    current_equity: float
    peak_equity: float
    
    # Slope and trend
    slope_daily: float          # Daily return trend
    slope_r_squared: float      # How linear is the curve
    slope_consistency: float    # % of days with positive returns
    
    # Volatility
    daily_volatility: float
    downside_volatility: float
    volatility_ratio: float     # Down vol / total vol
    
    # Drawdown
    current_drawdown_pct: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_duration_days: int
    max_drawdown_recovery_days: Optional[int]
    
    # Recovery
    time_underwater_pct: float  # % of time in drawdown
    recovery_factor: float      # Return / Max DD
    
    # Computed at timestamp
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExpectancyDrift:
    """Tracks drift between backtest and live expectancy."""
    backtest_expectancy: float
    live_expectancy: float
    drift_pct: float
    drift_significance: str  # "acceptable", "warning", "critical"
    
    backtest_win_rate: float
    live_win_rate: float
    win_rate_drift_pct: float
    
    backtest_profit_factor: float
    live_profit_factor: float
    profit_factor_drift_pct: float
    
    # By confidence bucket
    confidence_drift: Dict[str, float] = field(default_factory=dict)


@dataclass
class CapitalTrustScore:
    """
    Capital Trust Score (CTS)
    
    A composite score (0-100) measuring system stability,
    NOT raw returns. High trust = consistent, predictable behavior.
    """
    total_score: float
    
    # Component scores (0-100 each)
    stability_score: float      # Equity curve smoothness
    consistency_score: float    # Performance vs expectations
    risk_score: float           # Drawdown management
    recovery_score: float       # Ability to recover from losses
    pattern_score: float        # Absence of degradation patterns
    
    # Weights used
    weights: Dict[str, float] = field(default_factory=lambda: {
        'stability': 0.25,
        'consistency': 0.20,
        'risk': 0.25,
        'recovery': 0.15,
        'pattern': 0.15
    })
    
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    recommendation: str = ""


@dataclass
class ValidationState:
    """Current state of capital validation."""
    phase: ValidationPhase
    start_date: datetime
    days_elapsed: int
    
    # Model lock status
    model_version_locked: str
    model_locked_at: datetime
    learning_disabled: bool
    
    # Metrics
    equity_metrics: Optional[EquityCurveMetrics]
    expectancy_drift: Optional[ExpectancyDrift]
    trust_score: Optional[CapitalTrustScore]
    
    # Detected issues
    degradation_patterns: List[DegradationPattern]
    warnings: List[str]
    errors: List[str]
    
    # Validation result
    is_valid: Optional[bool]
    validation_notes: str


class EquityCurveAnalyzer:
    """
    Analyzes equity curves for validation metrics.
    """
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.equity_history: List[Tuple[datetime, float]] = []
        self.daily_equity: Dict[str, float] = {}
        self.trade_times: List[datetime] = []
    
    def add_equity_point(self, timestamp: datetime, equity: float):
        """Add an equity snapshot."""
        self.equity_history.append((timestamp, equity))
        
        # Track daily close
        date_str = timestamp.strftime('%Y-%m-%d')
        self.daily_equity[date_str] = equity
    
    def add_trade(self, timestamp: datetime):
        """Record a trade execution time."""
        self.trade_times.append(timestamp)
    
    def compute_metrics(self) -> EquityCurveMetrics:
        """Compute comprehensive equity curve metrics."""
        if len(self.equity_history) < 2:
            return self._empty_metrics()
        
        equities = np.array([e[1] for e in self.equity_history])
        timestamps = [e[0] for e in self.equity_history]
        
        # Basic stats
        current_equity = equities[-1]
        peak_equity = np.max(equities)
        total_return_pct = (current_equity - self.initial_equity) / self.initial_equity
        
        # Daily returns
        daily_values = list(self.daily_equity.values())
        if len(daily_values) < 2:
            daily_returns = np.array([0])
        else:
            daily_values = np.array(daily_values)
            daily_returns = np.diff(daily_values) / daily_values[:-1]
        
        # Slope analysis (linear regression on cumulative returns)
        x = np.arange(len(equities))
        if len(x) > 1:
            slope, intercept = np.polyfit(x, equities, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((equities - y_pred) ** 2)
            ss_tot = np.sum((equities - np.mean(equities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0
            r_squared = 0
        
        # Annualize daily slope
        days_elapsed = max(1, (timestamps[-1] - timestamps[0]).days)
        daily_slope_pct = (slope / self.initial_equity) if self.initial_equity > 0 else 0
        
        # Consistency (% of positive days)
        positive_days = np.sum(daily_returns > 0)
        total_days = len(daily_returns)
        slope_consistency = positive_days / total_days if total_days > 0 else 0
        
        # Volatility
        daily_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        negative_returns = daily_returns[daily_returns < 0]
        downside_volatility = np.std(negative_returns) if len(negative_returns) > 1 else 0
        volatility_ratio = downside_volatility / daily_volatility if daily_volatility > 0 else 0
        
        # Drawdown analysis
        running_max = np.maximum.accumulate(equities)
        drawdowns = (running_max - equities) / running_max
        
        current_dd = drawdowns[-1]
        max_dd = np.max(drawdowns)
        avg_dd = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0
        
        # Drawdown duration
        in_drawdown = drawdowns > 0.001  # > 0.1% considered in drawdown
        dd_duration = self._count_current_streak(in_drawdown)
        
        # Max drawdown recovery time
        max_dd_recovery = self._compute_max_dd_recovery(equities, running_max)
        
        # Time underwater
        time_underwater = np.sum(in_drawdown) / len(in_drawdown) if len(in_drawdown) > 0 else 0
        
        # Recovery factor
        recovery_factor = total_return_pct / max_dd if max_dd > 0 else float('inf') if total_return_pct > 0 else 0
        
        return EquityCurveMetrics(
            total_return_pct=total_return_pct,
            current_equity=current_equity,
            peak_equity=peak_equity,
            slope_daily=daily_slope_pct,
            slope_r_squared=r_squared,
            slope_consistency=slope_consistency,
            daily_volatility=daily_volatility,
            downside_volatility=downside_volatility,
            volatility_ratio=volatility_ratio,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
            avg_drawdown_pct=avg_dd,
            drawdown_duration_days=dd_duration,
            max_drawdown_recovery_days=max_dd_recovery,
            time_underwater_pct=time_underwater,
            recovery_factor=recovery_factor
        )
    
    def _count_current_streak(self, condition: np.ndarray) -> int:
        """Count current streak of True values from the end."""
        count = 0
        for val in reversed(condition):
            if val:
                count += 1
            else:
                break
        return count
    
    def _compute_max_dd_recovery(self, equities: np.ndarray, running_max: np.ndarray) -> Optional[int]:
        """Compute days to recover from max drawdown."""
        drawdowns = (running_max - equities) / running_max
        max_dd_idx = np.argmax(drawdowns)
        max_val_before = running_max[max_dd_idx]
        
        # Find when we recovered to previous peak
        for i in range(max_dd_idx, len(equities)):
            if equities[i] >= max_val_before:
                return i - max_dd_idx
        
        return None  # Not yet recovered
    
    def _empty_metrics(self) -> EquityCurveMetrics:
        return EquityCurveMetrics(
            total_return_pct=0, current_equity=self.initial_equity, peak_equity=self.initial_equity,
            slope_daily=0, slope_r_squared=0, slope_consistency=0,
            daily_volatility=0, downside_volatility=0, volatility_ratio=0,
            current_drawdown_pct=0, max_drawdown_pct=0, avg_drawdown_pct=0,
            drawdown_duration_days=0, max_drawdown_recovery_days=None,
            time_underwater_pct=0, recovery_factor=0
        )


class DegradationDetector:
    """
    Detects equity curve degradation patterns.
    """
    
    def __init__(self):
        self.detected_patterns: List[DegradationPattern] = []
    
    def detect_patterns(
        self,
        equity_metrics: EquityCurveMetrics,
        trades_per_day: float,
        expected_trades_per_day: float,
        avg_trade_pnl: float,
        trade_count: int
    ) -> List[DegradationPattern]:
        """
        Detect degradation patterns in equity curve.
        
        Returns list of detected patterns.
        """
        patterns = []
        
        # CHOP: High volatility, near-zero return, many trades
        if (equity_metrics.daily_volatility > 0.02 and 
            abs(equity_metrics.total_return_pct) < 0.02 and
            trades_per_day > expected_trades_per_day * 0.5):
            patterns.append(DegradationPattern.CHOP)
        
        # OVERTRADE: Way too many trades, negative expectancy
        if trades_per_day > expected_trades_per_day * 2.0:
            if avg_trade_pnl < 0:
                patterns.append(DegradationPattern.OVERTRADE)
        
        # BLEED: Consistent negative slope, low volatility
        if (equity_metrics.slope_daily < -0.001 and
            equity_metrics.slope_r_squared > 0.6 and
            equity_metrics.daily_volatility < 0.02):
            patterns.append(DegradationPattern.BLEED)
        
        # SPIKE_LOSS: Max DD happened quickly
        if (equity_metrics.max_drawdown_pct > 0.05 and
            equity_metrics.drawdown_duration_days < 2):
            patterns.append(DegradationPattern.SPIKE_LOSS)
        
        # STAGNATION: Very flat equity, low trade count
        if (abs(equity_metrics.total_return_pct) < 0.005 and
            equity_metrics.slope_r_squared > 0.8 and
            trades_per_day < expected_trades_per_day * 0.3):
            patterns.append(DegradationPattern.STAGNATION)
        
        if not patterns:
            patterns.append(DegradationPattern.NONE)
        
        self.detected_patterns = patterns
        return patterns


class ExpectancyTracker:
    """
    Tracks live expectancy vs backtest expectancy.
    """
    
    def __init__(
        self,
        backtest_expectancy: float,
        backtest_win_rate: float,
        backtest_profit_factor: float,
        backtest_confidence_accuracy: Dict[str, float]
    ):
        """
        Initialize with backtest baseline metrics.
        """
        self.backtest_expectancy = backtest_expectancy
        self.backtest_win_rate = backtest_win_rate
        self.backtest_profit_factor = backtest_profit_factor
        self.backtest_confidence_accuracy = backtest_confidence_accuracy
        
        # Live tracking
        self.live_trades: List[Dict] = []
    
    def add_trade(
        self,
        pnl: float,
        confidence: float,
        was_win: bool
    ):
        """Record a live trade."""
        self.live_trades.append({
            'pnl': pnl,
            'confidence': confidence,
            'was_win': was_win,
            'timestamp': datetime.now()
        })
    
    def compute_drift(self) -> ExpectancyDrift:
        """Compute expectancy drift from backtest."""
        if len(self.live_trades) == 0:
            return ExpectancyDrift(
                backtest_expectancy=self.backtest_expectancy,
                live_expectancy=0,
                drift_pct=0,
                drift_significance="insufficient_data",
                backtest_win_rate=self.backtest_win_rate,
                live_win_rate=0,
                win_rate_drift_pct=0,
                backtest_profit_factor=self.backtest_profit_factor,
                live_profit_factor=0,
                profit_factor_drift_pct=0
            )
        
        # Live expectancy
        pnls = [t['pnl'] for t in self.live_trades]
        live_expectancy = np.mean(pnls)
        
        # Live win rate
        wins = sum(1 for t in self.live_trades if t['was_win'])
        live_win_rate = wins / len(self.live_trades)
        
        # Live profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        live_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Drift calculations
        if self.backtest_expectancy != 0:
            exp_drift = (live_expectancy - self.backtest_expectancy) / abs(self.backtest_expectancy)
        else:
            exp_drift = live_expectancy  # Raw value if backtest was 0
        
        if self.backtest_win_rate != 0:
            wr_drift = (live_win_rate - self.backtest_win_rate) / self.backtest_win_rate
        else:
            wr_drift = 0
        
        if self.backtest_profit_factor != 0 and self.backtest_profit_factor != float('inf'):
            pf_drift = (live_profit_factor - self.backtest_profit_factor) / self.backtest_profit_factor
        else:
            pf_drift = 0
        
        # Confidence bucket drift
        conf_drift = self._compute_confidence_drift()
        
        # Significance assessment
        if len(self.live_trades) < 30:
            significance = "insufficient_data"
        elif abs(exp_drift) < 0.15:
            significance = "acceptable"
        elif abs(exp_drift) < 0.30:
            significance = "warning"
        else:
            significance = "critical"
        
        return ExpectancyDrift(
            backtest_expectancy=self.backtest_expectancy,
            live_expectancy=live_expectancy,
            drift_pct=exp_drift,
            drift_significance=significance,
            backtest_win_rate=self.backtest_win_rate,
            live_win_rate=live_win_rate,
            win_rate_drift_pct=wr_drift,
            backtest_profit_factor=self.backtest_profit_factor,
            live_profit_factor=live_profit_factor,
            profit_factor_drift_pct=pf_drift,
            confidence_drift=conf_drift
        )
    
    def _compute_confidence_drift(self) -> Dict[str, float]:
        """Compute win rate drift per confidence bucket."""
        buckets = {
            '0.60-0.65': {'trades': [], 'wins': 0},
            '0.65-0.70': {'trades': [], 'wins': 0},
            '0.70-0.75': {'trades': [], 'wins': 0},
            '0.75-0.80': {'trades': [], 'wins': 0},
            '0.80-1.00': {'trades': [], 'wins': 0}
        }
        
        bucket_ranges = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.00)]
        
        for trade in self.live_trades:
            conf = trade['confidence']
            for low, high in bucket_ranges:
                if low <= conf < high:
                    key = f"{low:.2f}-{high:.2f}"
                    buckets[key]['trades'].append(trade)
                    if trade['was_win']:
                        buckets[key]['wins'] += 1
                    break
        
        drift = {}
        for key, data in buckets.items():
            if len(data['trades']) >= 5:  # Minimum 5 trades per bucket
                live_wr = data['wins'] / len(data['trades'])
                backtest_wr = self.backtest_confidence_accuracy.get(key, 0.5)
                if backtest_wr > 0:
                    drift[key] = (live_wr - backtest_wr) / backtest_wr
                else:
                    drift[key] = 0
        
        return drift


class CapitalTrustScorer:
    """
    Computes Capital Trust Score (CTS).
    
    CTS measures system STABILITY, not raw returns.
    A system can have negative returns but high trust if it behaves predictably.
    """
    
    def __init__(self):
        self.weights = {
            'stability': 0.25,
            'consistency': 0.20,
            'risk': 0.25,
            'recovery': 0.15,
            'pattern': 0.15
        }
    
    def compute_score(
        self,
        equity_metrics: EquityCurveMetrics,
        expectancy_drift: ExpectancyDrift,
        degradation_patterns: List[DegradationPattern],
        days_elapsed: int
    ) -> CapitalTrustScore:
        """
        Compute Capital Trust Score.
        
        Returns score 0-100 based on stability metrics.
        """
        # 1. Stability Score (equity curve smoothness)
        stability = self._compute_stability_score(equity_metrics)
        
        # 2. Consistency Score (live vs backtest alignment)
        consistency = self._compute_consistency_score(expectancy_drift)
        
        # 3. Risk Score (drawdown management)
        risk = self._compute_risk_score(equity_metrics)
        
        # 4. Recovery Score (ability to recover from losses)
        recovery = self._compute_recovery_score(equity_metrics)
        
        # 5. Pattern Score (absence of degradation)
        pattern = self._compute_pattern_score(degradation_patterns)
        
        # Weighted total
        total = (
            stability * self.weights['stability'] +
            consistency * self.weights['consistency'] +
            risk * self.weights['risk'] +
            recovery * self.weights['recovery'] +
            pattern * self.weights['pattern']
        )
        
        # Determine trust level
        if days_elapsed < 14:
            trust_level = TrustLevel.UNTRUSTED
            recommendation = f"Continue validation for {14 - days_elapsed} more days"
        elif total >= 80:
            trust_level = TrustLevel.VALIDATED
            recommendation = "✅ System validated. Ready for live deployment with small capital."
        elif total >= 60:
            trust_level = TrustLevel.HIGH
            recommendation = "⚠️ Good stability. Consider extended validation or reduced position sizes."
        elif total >= 40:
            trust_level = TrustLevel.MEDIUM
            recommendation = "⚠️ Moderate stability. Review detected issues before live deployment."
        else:
            trust_level = TrustLevel.LOW
            recommendation = "❌ Low trust. Do not deploy to live trading. Investigate issues."
        
        return CapitalTrustScore(
            total_score=total,
            stability_score=stability,
            consistency_score=consistency,
            risk_score=risk,
            recovery_score=recovery,
            pattern_score=pattern,
            weights=self.weights,
            trust_level=trust_level,
            recommendation=recommendation
        )
    
    def _compute_stability_score(self, metrics: EquityCurveMetrics) -> float:
        """Score based on equity curve smoothness."""
        score = 100
        
        # Penalize high volatility
        if metrics.daily_volatility > 0.05:
            score -= 40
        elif metrics.daily_volatility > 0.03:
            score -= 20
        elif metrics.daily_volatility > 0.02:
            score -= 10
        
        # Reward high R-squared (linear equity curve)
        score += metrics.slope_r_squared * 20
        
        # Reward consistency (positive day ratio)
        score += (metrics.slope_consistency - 0.5) * 40  # 0.5 is neutral
        
        return max(0, min(100, score))
    
    def _compute_consistency_score(self, drift: ExpectancyDrift) -> float:
        """Score based on alignment with backtest expectations."""
        score = 100
        
        # Penalize expectancy drift
        if drift.drift_significance == "critical":
            score -= 50
        elif drift.drift_significance == "warning":
            score -= 25
        elif drift.drift_significance == "insufficient_data":
            score = 50  # Neutral
        
        # Penalize win rate drift
        if abs(drift.win_rate_drift_pct) > 0.20:
            score -= 20
        elif abs(drift.win_rate_drift_pct) > 0.10:
            score -= 10
        
        # Penalize profit factor drift
        if abs(drift.profit_factor_drift_pct) > 0.30:
            score -= 20
        elif abs(drift.profit_factor_drift_pct) > 0.15:
            score -= 10
        
        return max(0, min(100, score))
    
    def _compute_risk_score(self, metrics: EquityCurveMetrics) -> float:
        """Score based on drawdown management."""
        score = 100
        
        # Penalize max drawdown
        if metrics.max_drawdown_pct > 0.10:
            score -= 50
        elif metrics.max_drawdown_pct > 0.05:
            score -= 25
        elif metrics.max_drawdown_pct > 0.03:
            score -= 10
        
        # Penalize current drawdown
        if metrics.current_drawdown_pct > 0.05:
            score -= 20
        elif metrics.current_drawdown_pct > 0.02:
            score -= 10
        
        # Penalize time underwater
        if metrics.time_underwater_pct > 0.50:
            score -= 20
        elif metrics.time_underwater_pct > 0.30:
            score -= 10
        
        return max(0, min(100, score))
    
    def _compute_recovery_score(self, metrics: EquityCurveMetrics) -> float:
        """Score based on recovery ability."""
        score = 100
        
        # Penalize long drawdown duration
        if metrics.drawdown_duration_days > 7:
            score -= 30
        elif metrics.drawdown_duration_days > 3:
            score -= 15
        
        # Penalize if not recovered from max DD
        if metrics.max_drawdown_recovery_days is None:
            if metrics.max_drawdown_pct > 0.03:
                score -= 30
        else:
            # Reward quick recovery
            if metrics.max_drawdown_recovery_days <= 2:
                score += 10
            elif metrics.max_drawdown_recovery_days > 5:
                score -= 15
        
        # Reward good recovery factor
        if metrics.recovery_factor > 2.0:
            score += 15
        elif metrics.recovery_factor < 0.5:
            score -= 20
        
        return max(0, min(100, score))
    
    def _compute_pattern_score(self, patterns: List[DegradationPattern]) -> float:
        """Score based on absence of degradation patterns."""
        if DegradationPattern.NONE in patterns and len(patterns) == 1:
            return 100
        
        score = 100
        
        # Penalties for each pattern
        penalties = {
            DegradationPattern.CHOP: 25,
            DegradationPattern.OVERTRADE: 35,
            DegradationPattern.BLEED: 40,
            DegradationPattern.SPIKE_LOSS: 30,
            DegradationPattern.STAGNATION: 20
        }
        
        for pattern in patterns:
            if pattern in penalties:
                score -= penalties[pattern]
        
        return max(0, min(100, score))


class CapitalValidationMode:
    """
    Main Capital Validation Mode controller.
    
    Manages the 14-day validation process:
    1. Locks model at start
    2. Tracks equity and trades
    3. Computes daily metrics
    4. Detects issues early
    5. Provides go/no-go recommendation
    """
    
    VALIDATION_DAYS = 14
    
    def __init__(
        self,
        model_version: str,
        initial_equity: float = 10000.0,
        backtest_metrics: Optional[Dict] = None
    ):
        """
        Initialize validation mode.
        
        Args:
            model_version: Version of model being validated
            initial_equity: Starting capital
            backtest_metrics: Results from backtesting (for drift comparison)
        """
        self.model_version = model_version
        self.initial_equity = initial_equity
        
        # Lock model
        self.model_locked_at = datetime.now()
        self.learning_disabled = True
        
        # Initialize trackers
        self.equity_analyzer = EquityCurveAnalyzer(initial_equity)
        self.degradation_detector = DegradationDetector()
        self.trust_scorer = CapitalTrustScorer()
        
        # Setup expectancy tracker with backtest baseline
        if backtest_metrics:
            self.expectancy_tracker = ExpectancyTracker(
                backtest_expectancy=backtest_metrics.get('expectancy', 0),
                backtest_win_rate=backtest_metrics.get('win_rate', 0.5),
                backtest_profit_factor=backtest_metrics.get('profit_factor', 1.0),
                backtest_confidence_accuracy=backtest_metrics.get('confidence_accuracy', {})
            )
        else:
            self.expectancy_tracker = ExpectancyTracker(0, 0.5, 1.0, {})
        
        # State
        self.start_date = datetime.now()
        self.trade_count = 0
        self.expected_trades_per_day = backtest_metrics.get('trades_per_day', 2.0) if backtest_metrics else 2.0
        
        # Daily summaries
        self.daily_summaries: List[Dict] = []
        
        logger.info(f"Capital Validation Mode started for model {model_version}")
        logger.info(f"Model locked at {self.model_locked_at}. Learning disabled.")
    
    def record_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """Record current equity snapshot."""
        ts = timestamp or datetime.now()
        self.equity_analyzer.add_equity_point(ts, equity)
    
    def record_trade(
        self,
        pnl: float,
        confidence: float,
        was_win: bool,
        timestamp: Optional[datetime] = None
    ):
        """Record a completed trade."""
        ts = timestamp or datetime.now()
        self.trade_count += 1
        self.equity_analyzer.add_trade(ts)
        self.expectancy_tracker.add_trade(pnl, confidence, was_win)
    
    def get_state(self) -> ValidationState:
        """Get current validation state with all metrics."""
        now = datetime.now()
        days_elapsed = (now - self.start_date).days
        
        # Determine phase
        if days_elapsed < 3:
            phase = ValidationPhase.WARMUP
        elif days_elapsed < 10:
            phase = ValidationPhase.OBSERVATION
        elif days_elapsed < 14:
            phase = ValidationPhase.CONFIRMATION
        else:
            phase = ValidationPhase.COMPLETE
        
        # Compute metrics
        equity_metrics = self.equity_analyzer.compute_metrics()
        expectancy_drift = self.expectancy_tracker.compute_drift()
        
        # Compute trades per day
        trades_per_day = self.trade_count / max(1, days_elapsed)
        avg_pnl = self.expectancy_tracker.live_trades and np.mean([t['pnl'] for t in self.expectancy_tracker.live_trades]) or 0
        
        # Detect degradation
        patterns = self.degradation_detector.detect_patterns(
            equity_metrics,
            trades_per_day,
            self.expected_trades_per_day,
            avg_pnl,
            self.trade_count
        )
        
        # Compute trust score
        trust_score = self.trust_scorer.compute_score(
            equity_metrics,
            expectancy_drift,
            patterns,
            days_elapsed
        )
        
        # Generate warnings
        warnings = []
        errors = []
        
        if expectancy_drift.drift_significance == "warning":
            warnings.append(f"Expectancy drift detected: {expectancy_drift.drift_pct:.1%}")
        if expectancy_drift.drift_significance == "critical":
            errors.append(f"Critical expectancy drift: {expectancy_drift.drift_pct:.1%}")
        
        if equity_metrics.max_drawdown_pct > 0.05:
            warnings.append(f"Max drawdown exceeded 5%: {equity_metrics.max_drawdown_pct:.1%}")
        
        for pattern in patterns:
            if pattern != DegradationPattern.NONE:
                warnings.append(f"Degradation pattern detected: {pattern.value}")
        
        # Determine if valid
        is_valid = None
        if phase == ValidationPhase.COMPLETE:
            is_valid = trust_score.trust_level in [TrustLevel.HIGH, TrustLevel.VALIDATED]
        
        return ValidationState(
            phase=phase,
            start_date=self.start_date,
            days_elapsed=days_elapsed,
            model_version_locked=self.model_version,
            model_locked_at=self.model_locked_at,
            learning_disabled=self.learning_disabled,
            equity_metrics=equity_metrics,
            expectancy_drift=expectancy_drift,
            trust_score=trust_score,
            degradation_patterns=patterns,
            warnings=warnings,
            errors=errors,
            is_valid=is_valid,
            validation_notes=trust_score.recommendation
        )
    
    def generate_daily_report(self) -> str:
        """Generate daily validation progress report."""
        state = self.get_state()
        
        report = []
        report.append("=" * 60)
        report.append("AURIX CAPITAL VALIDATION - DAILY REPORT")
        report.append("=" * 60)
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Phase: {state.phase.value.upper()}")
        report.append(f"Days Elapsed: {state.days_elapsed} / {self.VALIDATION_DAYS}")
        report.append(f"Model Version: {state.model_version_locked} (LOCKED)")
        
        report.append("\n--- CAPITAL TRUST SCORE ---")
        cts = state.trust_score
        report.append(f"Total CTS: {cts.total_score:.1f} / 100")
        report.append(f"  Stability:   {cts.stability_score:.1f}")
        report.append(f"  Consistency: {cts.consistency_score:.1f}")
        report.append(f"  Risk Mgmt:   {cts.risk_score:.1f}")
        report.append(f"  Recovery:    {cts.recovery_score:.1f}")
        report.append(f"  Pattern:     {cts.pattern_score:.1f}")
        report.append(f"Trust Level: {cts.trust_level.value.upper()}")
        
        report.append("\n--- EQUITY METRICS ---")
        em = state.equity_metrics
        report.append(f"Current Equity: ${em.current_equity:,.2f}")
        report.append(f"Total Return: {em.total_return_pct:.2%}")
        report.append(f"Max Drawdown: {em.max_drawdown_pct:.2%}")
        report.append(f"Current Drawdown: {em.current_drawdown_pct:.2%}")
        report.append(f"Daily Volatility: {em.daily_volatility:.2%}")
        report.append(f"Time Underwater: {em.time_underwater_pct:.1%}")
        
        report.append("\n--- EXPECTANCY DRIFT ---")
        ed = state.expectancy_drift
        report.append(f"Backtest Expectancy: ${ed.backtest_expectancy:.2f}")
        report.append(f"Live Expectancy: ${ed.live_expectancy:.2f}")
        report.append(f"Drift: {ed.drift_pct:.1%} ({ed.drift_significance})")
        report.append(f"Win Rate: {ed.live_win_rate:.1%} (backtest: {ed.backtest_win_rate:.1%})")
        
        report.append("\n--- DEGRADATION PATTERNS ---")
        for pattern in state.degradation_patterns:
            status = "✅ None detected" if pattern == DegradationPattern.NONE else f"⚠️ {pattern.value}"
            report.append(f"  {status}")
        
        if state.warnings:
            report.append("\n--- WARNINGS ---")
            for w in state.warnings:
                report.append(f"  ⚠️ {w}")
        
        if state.errors:
            report.append("\n--- ERRORS ---")
            for e in state.errors:
                report.append(f"  ❌ {e}")
        
        report.append("\n--- RECOMMENDATION ---")
        report.append(cts.recommendation)
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_state(self, filepath: str):
        """Save validation state to JSON file."""
        state = self.get_state()
        
        data = {
            'phase': state.phase.value,
            'start_date': state.start_date.isoformat(),
            'days_elapsed': state.days_elapsed,
            'model_version': state.model_version_locked,
            'model_locked_at': state.model_locked_at.isoformat(),
            'learning_disabled': state.learning_disabled,
            'equity_metrics': {
                'current_equity': state.equity_metrics.current_equity,
                'total_return_pct': state.equity_metrics.total_return_pct,
                'max_drawdown_pct': state.equity_metrics.max_drawdown_pct,
                'daily_volatility': state.equity_metrics.daily_volatility,
                'slope_daily': state.equity_metrics.slope_daily,
                'slope_r_squared': state.equity_metrics.slope_r_squared
            },
            'expectancy_drift': {
                'live_expectancy': state.expectancy_drift.live_expectancy,
                'drift_pct': state.expectancy_drift.drift_pct,
                'drift_significance': state.expectancy_drift.drift_significance,
                'live_win_rate': state.expectancy_drift.live_win_rate
            },
            'trust_score': {
                'total': state.trust_score.total_score,
                'stability': state.trust_score.stability_score,
                'consistency': state.trust_score.consistency_score,
                'risk': state.trust_score.risk_score,
                'recovery': state.trust_score.recovery_score,
                'pattern': state.trust_score.pattern_score,
                'trust_level': state.trust_score.trust_level.value,
                'recommendation': state.trust_score.recommendation
            },
            'degradation_patterns': [p.value for p in state.degradation_patterns],
            'warnings': state.warnings,
            'errors': state.errors,
            'is_valid': state.is_valid,
            'trade_count': self.trade_count,
            'equity_history_length': len(self.equity_analyzer.equity_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Validation state saved to {filepath}")
    
    def should_halt(self) -> Tuple[bool, str]:
        """
        Check if trading should be halted due to validation failures.
        
        Returns:
            Tuple of (should_halt, reason)
        """
        state = self.get_state()
        
        # Critical drift = halt
        if state.expectancy_drift.drift_significance == "critical":
            return True, "Critical expectancy drift detected"
        
        # Max drawdown exceeded
        if state.equity_metrics.max_drawdown_pct > 0.10:
            return True, "Maximum drawdown exceeded 10%"
        
        # Multiple degradation patterns
        bad_patterns = [p for p in state.degradation_patterns if p != DegradationPattern.NONE]
        if len(bad_patterns) >= 2:
            return True, f"Multiple degradation patterns: {', '.join(p.value for p in bad_patterns)}"
        
        # CTS too low after warmup
        if state.days_elapsed >= 5 and state.trust_score.total_score < 25:
            return True, f"Capital Trust Score critically low: {state.trust_score.total_score:.1f}"
        
        return False, ""
