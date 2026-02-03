"""
AURIX Capital Growth Orchestrator

State machine for dynamic risk management:
- Accumulation: Steady conservative growth
- Expansion: Capitalize on momentum
- Defense: Reduce exposure on warnings
- Preservation: Protect capital at all costs

Includes Capital Fatigue Index (CFI) for detecting
diminishing edge and preventing over-capitalization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GrowthState(Enum):
    """Capital growth states."""
    ACCUMULATION = "accumulation"  # Default steady growth
    EXPANSION = "expansion"        # Capitalize on momentum
    DEFENSE = "defense"            # Reduce exposure on warning
    PRESERVATION = "preservation"  # Protect capital at all costs


@dataclass
class GrowthParameters:
    """Trading parameters for each growth state."""
    risk_per_trade: float      # % of equity per trade (0.01 = 1%)
    aggression_factor: float   # Multiplier on position sizing
    exposure_ceiling: float    # Max % of capital deployed
    trade_frequency_mult: float = 1.0  # Trade frequency multiplier


@dataclass
class FatigueComponents:
    """Components of Capital Fatigue Index."""
    return_per_risk_trend: float  # Trend of return/risk ratio
    profit_velocity_ratio: float  # Profit velocity vs exposure growth
    trade_efficiency_decay: float # Decay in avg trade efficiency
    grinding_detected: bool       # True if in grinding phase


@dataclass
class CapitalFatigueIndex:
    """Capital Fatigue Index with component breakdown."""
    value: float  # 0.0 (no fatigue) to 1.0 (severe fatigue)
    components: FatigueComponents
    is_fatigued: bool
    grinding_phase: bool
    recommendation: str


@dataclass
class GrowthStateInfo:
    """Full state information with transition details."""
    state: GrowthState
    parameters: GrowthParameters
    state_duration_hours: float
    transition_reason: str
    next_likely_state: Optional[str]
    
    # Metrics
    equity_slope: float
    drawdown_velocity: float
    fatigue_index: CapitalFatigueIndex
    
    def to_log_string(self) -> str:
        return (
            f"GrowthState={self.state.value} | "
            f"Risk={self.parameters.risk_per_trade:.1%} "
            f"Agg={self.parameters.aggression_factor:.1f}x "
            f"Cap={self.parameters.exposure_ceiling:.0%} | "
            f"Fatigue={self.fatigue_index.value:.2f}"
        )


# Default parameters for each state
STATE_PARAMETERS = {
    GrowthState.ACCUMULATION: GrowthParameters(
        risk_per_trade=0.01,      # 1%
        aggression_factor=0.8,
        exposure_ceiling=0.30,    # 30%
        trade_frequency_mult=1.0
    ),
    GrowthState.EXPANSION: GrowthParameters(
        risk_per_trade=0.015,     # 1.5%
        aggression_factor=1.2,
        exposure_ceiling=0.50,    # 50%
        trade_frequency_mult=1.2
    ),
    GrowthState.DEFENSE: GrowthParameters(
        risk_per_trade=0.005,     # 0.5%
        aggression_factor=0.5,
        exposure_ceiling=0.20,    # 20%
        trade_frequency_mult=0.7
    ),
    GrowthState.PRESERVATION: GrowthParameters(
        risk_per_trade=0.0025,    # 0.25%
        aggression_factor=0.3,
        exposure_ceiling=0.10,    # 10%
        trade_frequency_mult=0.5
    ),
}


class GrowthOrchestrator:
    """
    Capital Growth Orchestrator
    
    Dynamically adjusts risk/aggression/exposure based on:
    1. Equity curve slope
    2. Drawdown velocity
    3. Reality Score trend
    4. Capital Fatigue Index
    
    Goal: Maximize long-term equity curve convexity
    while minimizing psychological & capital fatigue.
    """
    
    # Transition thresholds
    EQUITY_SLOPE_EXPANSION = 0.5    # Enter expansion
    EQUITY_SLOPE_COOLDOWN = 0.3     # Exit expansion
    DRAWDOWN_WARNING = 0.03         # 3% DD triggers defense
    DRAWDOWN_CRITICAL = 0.06        # 6% DD triggers preservation
    DRAWDOWN_RECOVERY = 0.02        # 2% DD allows accumulation
    DRAWDOWN_PARTIAL_RECOVERY = 0.04  # 4% DD allows defense
    REALITY_SCORE_MIN = 0.5         # Min score for expansion
    FATIGUE_THRESHOLD = 0.6         # CFI threshold for state change
    
    # Timing
    MIN_STATE_DURATION_HOURS = 4.0
    
    def __init__(self, history_size: int = 100):
        """
        Initialize orchestrator.
        
        Args:
            history_size: Size of metric histories
        """
        self.history_size = history_size
        
        # Current state
        self._state = GrowthState.ACCUMULATION
        self._state_start = datetime.now()
        self._transition_reason = "Initial state"
        
        # Metric histories
        self._equity_history: List[float] = []
        self._drawdown_history: List[float] = []
        self._trade_returns: List[float] = []  # Individual trade returns
        self._trade_risks: List[float] = []    # Risk per trade taken
        self._trade_efficiencies: List[float] = []  # Return / Risk ratio
        
        # State history
        self._state_history: List[Tuple[datetime, GrowthState, str]] = []
        
        # Consecutive tracking
        self._consecutive_wins = 0
        self._consecutive_losses = 0
    
    @property
    def state(self) -> GrowthState:
        return self._state
    
    @property
    def parameters(self) -> GrowthParameters:
        return STATE_PARAMETERS[self._state]
    
    def update(
        self,
        current_equity: float,
        current_drawdown_pct: float,
        reality_score: float,
        reality_trend: str,
        consecutive_losses: int,
        consecutive_wins: int = 0
    ) -> GrowthStateInfo:
        """
        Update state machine based on current metrics.
        
        Args:
            current_equity: Current account equity
            current_drawdown_pct: Current drawdown as decimal (0.05 = 5%)
            reality_score: Reality Score (0-1)
            reality_trend: "improving", "stable", "degrading"
            consecutive_losses: Current consecutive loss streak
            consecutive_wins: Current consecutive win streak
            
        Returns:
            GrowthStateInfo with current state and parameters
        """
        # Update histories
        self._equity_history.append(current_equity)
        self._drawdown_history.append(current_drawdown_pct)
        
        if len(self._equity_history) > self.history_size:
            self._equity_history = self._equity_history[-self.history_size:]
            self._drawdown_history = self._drawdown_history[-self.history_size:]
        
        self._consecutive_wins = consecutive_wins
        self._consecutive_losses = consecutive_losses
        
        # Calculate metrics
        equity_slope = self._calculate_equity_slope()
        drawdown_velocity = self._calculate_drawdown_velocity()
        fatigue_index = self._calculate_fatigue_index()
        
        # Check for state transition
        new_state, reason = self._evaluate_transition(
            equity_slope=equity_slope,
            drawdown_pct=current_drawdown_pct,
            drawdown_velocity=drawdown_velocity,
            reality_score=reality_score,
            reality_trend=reality_trend,
            consecutive_losses=consecutive_losses,
            fatigue_index=fatigue_index
        )
        
        if new_state != self._state:
            self._transition_to(new_state, reason)
        
        # Calculate state duration
        state_duration = (datetime.now() - self._state_start).total_seconds() / 3600
        
        # Predict next state
        next_likely = self._predict_next_state(
            equity_slope, current_drawdown_pct, fatigue_index
        )
        
        state_info = GrowthStateInfo(
            state=self._state,
            parameters=self.parameters,
            state_duration_hours=state_duration,
            transition_reason=self._transition_reason,
            next_likely_state=next_likely,
            equity_slope=equity_slope,
            drawdown_velocity=drawdown_velocity,
            fatigue_index=fatigue_index
        )
        
        logger.info(f"[GROWTH] {state_info.to_log_string()}")
        
        return state_info
    
    def record_trade(self, pnl: float, risk_taken: float):
        """
        Record a completed trade for fatigue calculation.
        
        Args:
            pnl: Profit/loss of trade
            risk_taken: Risk amount taken on trade
        """
        self._trade_returns.append(pnl)
        self._trade_risks.append(risk_taken)
        
        efficiency = pnl / risk_taken if risk_taken > 0 else 0
        self._trade_efficiencies.append(efficiency)
        
        # Trim histories
        if len(self._trade_returns) > self.history_size:
            self._trade_returns = self._trade_returns[-self.history_size:]
            self._trade_risks = self._trade_risks[-self.history_size:]
            self._trade_efficiencies = self._trade_efficiencies[-self.history_size:]
    
    def _calculate_equity_slope(self, window: int = 20) -> float:
        """
        Calculate normalized slope of equity curve.
        
        Returns:
            Slope from -1 (steep decline) to +1 (steep rise)
        """
        if len(self._equity_history) < window:
            return 0.0
        
        recent = self._equity_history[-window:]
        x = np.arange(len(recent))
        
        try:
            slope = np.polyfit(x, recent, 1)[0]
            avg_equity = np.mean(recent)
            # Normalize using tanh for bounded output
            normalized = np.tanh(slope / avg_equity * 100)
            return float(normalized)
        except Exception:
            return 0.0
    
    def _calculate_drawdown_velocity(self, window: int = 5) -> float:
        """
        Calculate rate of drawdown change.
        
        Returns:
            Negative = worsening, Positive = recovering
        """
        if len(self._drawdown_history) < window:
            return 0.0
        
        recent = self._drawdown_history[-window:]
        # Negative value means drawdown is increasing (bad)
        velocity = -(recent[-1] - recent[0]) / window
        return float(velocity)
    
    def _calculate_fatigue_index(self) -> CapitalFatigueIndex:
        """
        Calculate Capital Fatigue Index (CFI).
        
        Combines:
        1. Return per unit risk trend
        2. Profit velocity vs exposure growth
        3. Average trade efficiency decay
        """
        # Need enough data
        if len(self._trade_efficiencies) < 10:
            return CapitalFatigueIndex(
                value=0.0,
                components=FatigueComponents(0, 0, 0, False),
                is_fatigued=False,
                grinding_phase=False,
                recommendation="Insufficient data"
            )
        
        # 1. Return per risk trend (compare recent vs older)
        half = len(self._trade_efficiencies) // 2
        old_rpr = np.mean(self._trade_efficiencies[:half]) if half > 0 else 0
        new_rpr = np.mean(self._trade_efficiencies[half:])
        
        if old_rpr > 0:
            rpr_trend = (old_rpr - new_rpr) / old_rpr  # Positive = declining
        else:
            rpr_trend = 0.0
        rpr_trend = np.clip(rpr_trend, -1, 1)
        
        # 2. Profit velocity vs exposure growth
        if len(self._equity_history) >= 10:
            old_equity = self._equity_history[-10]
            new_equity = self._equity_history[-1]
            equity_growth = (new_equity - old_equity) / old_equity if old_equity > 0 else 0
            
            old_risk = np.mean(self._trade_risks[:half]) if half > 0 else 0
            new_risk = np.mean(self._trade_risks[half:])
            risk_growth = (new_risk - old_risk) / old_risk if old_risk > 0 else 0
            
            # Fatigue if risk growing faster than returns
            pv_ratio = risk_growth - equity_growth if risk_growth > equity_growth else 0
        else:
            pv_ratio = 0.0
        pv_ratio = np.clip(pv_ratio, 0, 1)
        
        # 3. Trade efficiency decay (moving average comparison)
        window = min(5, len(self._trade_efficiencies) // 2)
        if window >= 2:
            recent_5 = np.mean(self._trade_efficiencies[-window:])
            older_5 = np.mean(self._trade_efficiencies[-window*2:-window])
            
            if older_5 > 0:
                efficiency_decay = (older_5 - recent_5) / older_5
            else:
                efficiency_decay = 0.0
        else:
            efficiency_decay = 0.0
        efficiency_decay = np.clip(efficiency_decay, 0, 1)
        
        # Detect grinding phase
        # Equity still rising but marginal return decreasing
        equity_rising = self._calculate_equity_slope() > 0.1
        marginal_declining = rpr_trend > 0.2 or efficiency_decay > 0.2
        grinding = equity_rising and marginal_declining
        
        # Composite CFI (weighted average)
        cfi = (
            rpr_trend * 0.40 +
            pv_ratio * 0.30 +
            efficiency_decay * 0.30
        )
        cfi = np.clip(cfi, 0, 1)
        
        # Determine fatigue level
        is_fatigued = cfi > self.FATIGUE_THRESHOLD
        
        # Recommendation
        if grinding:
            recommendation = "Grinding detected - reduce expansion, consolidate gains"
        elif is_fatigued:
            recommendation = "Capital fatigue - reduce risk, allow recovery"
        elif cfi > 0.4:
            recommendation = "Mild fatigue - monitor closely"
        else:
            recommendation = "Healthy capital efficiency"
        
        return CapitalFatigueIndex(
            value=float(cfi),
            components=FatigueComponents(
                return_per_risk_trend=float(rpr_trend),
                profit_velocity_ratio=float(pv_ratio),
                trade_efficiency_decay=float(efficiency_decay),
                grinding_detected=grinding
            ),
            is_fatigued=is_fatigued,
            grinding_phase=grinding,
            recommendation=recommendation
        )
    
    def _evaluate_transition(
        self,
        equity_slope: float,
        drawdown_pct: float,
        drawdown_velocity: float,
        reality_score: float,
        reality_trend: str,
        consecutive_losses: int,
        fatigue_index: CapitalFatigueIndex
    ) -> Tuple[GrowthState, str]:
        """Evaluate if state transition is needed."""
        
        current = self._state
        state_hours = (datetime.now() - self._state_start).total_seconds() / 3600
        
        # Enforce minimum state duration (except for critical transitions)
        min_duration_met = state_hours >= self.MIN_STATE_DURATION_HOURS
        
        # === CRITICAL TRANSITIONS (bypass min duration) ===
        
        # Any state → Preservation: Critical drawdown
        if drawdown_pct >= self.DRAWDOWN_CRITICAL:
            return GrowthState.PRESERVATION, f"Critical drawdown: {drawdown_pct:.1%}"
        
        if consecutive_losses >= 4:
            return GrowthState.PRESERVATION, f"Loss streak: {consecutive_losses} consecutive"
        
        # === NORMAL TRANSITIONS ===
        
        if not min_duration_met and current not in [GrowthState.PRESERVATION]:
            return current, self._transition_reason
        
        # Expansion → Defense: Warning signs
        if current == GrowthState.EXPANSION:
            if drawdown_pct >= self.DRAWDOWN_WARNING:
                return GrowthState.DEFENSE, f"Drawdown warning: {drawdown_pct:.1%}"
            if reality_trend == "degrading":
                return GrowthState.DEFENSE, "Reality Score degrading"
            if fatigue_index.grinding_phase:
                return GrowthState.ACCUMULATION, "Grinding phase detected"
            if fatigue_index.is_fatigued:
                return GrowthState.ACCUMULATION, f"Capital fatigue: {fatigue_index.value:.2f}"
            if equity_slope < self.EQUITY_SLOPE_COOLDOWN:
                return GrowthState.ACCUMULATION, f"Momentum cooling: slope={equity_slope:.2f}"
        
        # Defense → Preservation: Worsening
        if current == GrowthState.DEFENSE:
            if consecutive_losses >= 3:
                return GrowthState.PRESERVATION, f"Consecutive losses: {consecutive_losses}"
            if reality_score < self.REALITY_SCORE_MIN:
                return GrowthState.PRESERVATION, f"Low Reality Score: {reality_score:.2f}"
        
        # Defense → Accumulation: Recovery
        if current == GrowthState.DEFENSE:
            if drawdown_pct < self.DRAWDOWN_RECOVERY and reality_score >= 0.6:
                return GrowthState.ACCUMULATION, "Drawdown recovered"
        
        # Preservation → Defense: Partial recovery
        if current == GrowthState.PRESERVATION:
            if drawdown_pct < self.DRAWDOWN_PARTIAL_RECOVERY and state_hours >= 6:
                return GrowthState.DEFENSE, "Partial recovery - cautious resume"
        
        # Accumulation → Expansion: Strong momentum
        if current == GrowthState.ACCUMULATION:
            if (equity_slope > self.EQUITY_SLOPE_EXPANSION and 
                reality_score > 0.7 and 
                self._consecutive_wins >= 3 and
                not fatigue_index.is_fatigued):
                return GrowthState.EXPANSION, "Strong momentum detected"
        
        return current, self._transition_reason
    
    def _transition_to(self, new_state: GrowthState, reason: str):
        """Execute state transition."""
        old_state = self._state
        
        logger.warning(f"[GROWTH] State transition: {old_state.value} → {new_state.value}")
        logger.warning(f"[GROWTH] Reason: {reason}")
        
        self._state_history.append((datetime.now(), old_state, reason))
        
        self._state = new_state
        self._state_start = datetime.now()
        self._transition_reason = reason
        
        # Log parameter change
        old_params = STATE_PARAMETERS[old_state]
        new_params = STATE_PARAMETERS[new_state]
        
        logger.info(f"[GROWTH] Risk: {old_params.risk_per_trade:.2%} → {new_params.risk_per_trade:.2%}")
        logger.info(f"[GROWTH] Aggression: {old_params.aggression_factor:.1f}x → {new_params.aggression_factor:.1f}x")
        logger.info(f"[GROWTH] Cap: {old_params.exposure_ceiling:.0%} → {new_params.exposure_ceiling:.0%}")
    
    def _predict_next_state(
        self,
        equity_slope: float,
        drawdown_pct: float,
        fatigue: CapitalFatigueIndex
    ) -> Optional[str]:
        """Predict likely next state based on trends."""
        if self._state == GrowthState.ACCUMULATION:
            if equity_slope > 0.4:
                return "expansion (momentum building)"
            if drawdown_pct > 0.025:
                return "defense (drawdown increasing)"
        
        elif self._state == GrowthState.EXPANSION:
            if fatigue.value > 0.5:
                return "accumulation (fatigue building)"
            if drawdown_pct > 0.025:
                return "defense (drawdown risk)"
        
        elif self._state == GrowthState.DEFENSE:
            if drawdown_pct < 0.015:
                return "accumulation (recovering)"
            if drawdown_pct > 0.05:
                return "preservation (deteriorating)"
        
        elif self._state == GrowthState.PRESERVATION:
            if drawdown_pct < 0.035:
                return "defense (improving)"
        
        return None
    
    def force_state(self, state: GrowthState, reason: str = "Manual override"):
        """Force transition to specific state."""
        logger.warning(f"[GROWTH] FORCED STATE: {state.value} - {reason}")
        self._transition_to(state, reason)
    
    def get_state_history(self, limit: int = 20) -> List[Tuple[datetime, GrowthState, str]]:
        """Get recent state transition history."""
        return self._state_history[-limit:]
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        state_hours = (datetime.now() - self._state_start).total_seconds() / 3600
        fatigue = self._calculate_fatigue_index()
        
        lines = [
            "=== Capital Growth Orchestrator ===",
            f"State: {self._state.value.upper()}",
            f"Duration: {state_hours:.1f} hours",
            f"Transition Reason: {self._transition_reason}",
            "",
            f"Risk/Trade: {self.parameters.risk_per_trade:.2%}",
            f"Aggression: {self.parameters.aggression_factor:.1f}x",
            f"Exposure Cap: {self.parameters.exposure_ceiling:.0%}",
            "",
            f"Equity Slope: {self._calculate_equity_slope():.2f}",
            f"Capital Fatigue: {fatigue.value:.2f} ({'FATIGUED' if fatigue.is_fatigued else 'healthy'})",
            f"Grinding Phase: {'YES' if fatigue.grinding_phase else 'No'}",
        ]
        
        return "\n".join(lines)
