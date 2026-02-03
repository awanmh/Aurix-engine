"""
AURIX Capital Efficiency Scorer

Measures return per unit of risk per unit of time.
CES = (Net Return × Win Rate) / (Max Drawdown × Avg Hold Time)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EfficiencyTrend(Enum):
    """Trend direction for efficiency score."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class EfficiencyScore:
    """Capital efficiency score with components."""
    composite: float  # Overall CES (0-1 normalized)
    return_score: float  # Return component
    risk_score: float  # Risk component (inverse of drawdown)
    time_score: float  # Time component (faster = better)
    trend: EfficiencyTrend
    percentile: float  # Where this score ranks historically
    position_multiplier: float  # Suggested position size multiplier
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'composite': self.composite,
            'return_score': self.return_score,
            'risk_score': self.risk_score,
            'time_score': self.time_score,
            'trend': self.trend.value,
            'percentile': self.percentile,
            'position_multiplier': self.position_multiplier,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TradeRecord:
    """Simplified trade record for efficiency calculation."""
    entry_time: datetime
    exit_time: datetime
    net_pnl: float
    max_drawdown: float  # Max adverse excursion during trade
    capital_at_risk: float
    symbol: str
    direction: str


class CapitalEfficiencyScorer:
    """
    Calculates Capital Efficiency Score (CES) to optimize capital deployment.
    
    CES Formula:
        CES = (Return Score × Win Rate) / (Risk Score × Time Score)
        
    Where:
    - Return Score = Net Return / Capital at Risk
    - Risk Score = Max Drawdown during trades (lower = better)
    - Time Score = Average holding time (shorter = better for same return)
    """
    
    def __init__(
        self,
        window_days: int = 30,
        return_weight: float = 0.35,
        risk_weight: float = 0.35,
        time_weight: float = 0.30,
        min_trades_for_score: int = 10,
        target_hold_time_minutes: int = 60
    ):
        """
        Initialize scorer.
        
        Args:
            window_days: Rolling window for score calculation
            return_weight: Weight for return component
            risk_weight: Weight for risk component
            time_weight: Weight for time component
            min_trades_for_score: Minimum trades needed for valid score
            target_hold_time_minutes: Target hold time for normalization
        """
        self.window_days = window_days
        self.weights = {
            'return': return_weight,
            'risk': risk_weight,
            'time': time_weight
        }
        self.min_trades = min_trades_for_score
        self.target_hold_time = target_hold_time_minutes
        
        # Historical scores for percentile calculation
        self.score_history: List[float] = []
        self.trade_history: List[TradeRecord] = []
    
    def add_trade(self, trade: TradeRecord):
        """Add a completed trade to history."""
        self.trade_history.append(trade)
        
        # Keep only recent trades
        cutoff = datetime.now() - timedelta(days=self.window_days * 2)
        self.trade_history = [t for t in self.trade_history if t.exit_time > cutoff]
    
    def calculate_score(
        self,
        symbol: Optional[str] = None,
        window_days: Optional[int] = None
    ) -> Optional[EfficiencyScore]:
        """
        Calculate current capital efficiency score.
        
        Args:
            symbol: Optional symbol filter
            window_days: Optional override for window
            
        Returns:
            EfficiencyScore or None if insufficient data
        """
        window = window_days or self.window_days
        cutoff = datetime.now() - timedelta(days=window)
        
        # Filter trades
        trades = [t for t in self.trade_history if t.exit_time > cutoff]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if len(trades) < self.min_trades:
            logger.debug(f"Insufficient trades for CES: {len(trades)}/{self.min_trades}")
            return None
        
        # Calculate components
        return_score = self._calculate_return_score(trades)
        risk_score = self._calculate_risk_score(trades)
        time_score = self._calculate_time_score(trades)
        win_rate = sum(1 for t in trades if t.net_pnl > 0) / len(trades)
        
        # Composite score
        # Higher return, lower risk, faster time = higher score
        composite = (
            (return_score * win_rate) /
            (risk_score * time_score + 0.001)  # Avoid division by zero
        )
        
        # Normalize to 0-1 range using sigmoid
        composite_normalized = 1 / (1 + np.exp(-composite + 1))
        
        # Calculate trend
        trend = self._calculate_trend(composite_normalized)
        
        # Calculate percentile
        self.score_history.append(composite_normalized)
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-1000:]
        
        percentile = (
            sum(1 for s in self.score_history if s <= composite_normalized) /
            len(self.score_history)
        )
        
        # Position multiplier based on score
        position_multiplier = self._calculate_position_multiplier(
            composite_normalized, percentile
        )
        
        return EfficiencyScore(
            composite=composite_normalized,
            return_score=return_score,
            risk_score=risk_score,
            time_score=time_score,
            trend=trend,
            percentile=percentile,
            position_multiplier=position_multiplier
        )
    
    def _calculate_return_score(self, trades: List[TradeRecord]) -> float:
        """Calculate return score component."""
        if not trades:
            return 0.0
        
        total_return = sum(t.net_pnl for t in trades)
        total_capital = sum(t.capital_at_risk for t in trades)
        
        if total_capital == 0:
            return 0.0
        
        # Return on capital at risk
        return_pct = total_return / total_capital
        
        # Normalize (target 5% per window = 1.0 score)
        target_return = 0.05
        return min(2.0, return_pct / target_return)
    
    def _calculate_risk_score(self, trades: List[TradeRecord]) -> float:
        """Calculate risk score component (lower = better)."""
        if not trades:
            return 1.0
        
        # Average max drawdown as percentage of capital
        avg_drawdown = np.mean([
            t.max_drawdown / t.capital_at_risk 
            for t in trades 
            if t.capital_at_risk > 0
        ])
        
        # Normalize (target 1% max drawdown = 1.0 score)
        target_drawdown = 0.01
        return max(0.1, avg_drawdown / target_drawdown)
    
    def _calculate_time_score(self, trades: List[TradeRecord]) -> float:
        """Calculate time score component (shorter = better for same return)."""
        if not trades:
            return 1.0
        
        # Average hold time in minutes
        hold_times = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in trades
        ]
        avg_hold_time = np.mean(hold_times)
        
        # Normalize against target
        return max(0.1, avg_hold_time / self.target_hold_time)
    
    def _calculate_trend(self, current_score: float) -> EfficiencyTrend:
        """Determine trend direction from recent scores."""
        if len(self.score_history) < 5:
            return EfficiencyTrend.STABLE
        
        recent = self.score_history[-5:]
        older = self.score_history[-10:-5] if len(self.score_history) >= 10 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return EfficiencyTrend.IMPROVING
        elif diff < -0.05:
            return EfficiencyTrend.DEGRADING
        else:
            return EfficiencyTrend.STABLE
    
    def _calculate_position_multiplier(
        self,
        score: float,
        percentile: float
    ) -> float:
        """
        Calculate position size multiplier based on efficiency.
        
        Higher efficiency = larger positions allowed
        """
        # Base multiplier from score
        base = 0.5 + (score * 0.5)  # Range: 0.5 - 1.0
        
        # Bonus for high percentile performance
        if percentile > 0.8:
            base *= 1.1
        elif percentile < 0.3:
            base *= 0.85
        
        return min(1.5, max(0.5, base))
    
    def get_efficiency_summary(self) -> Dict:
        """Get summary of efficiency metrics."""
        if len(self.trade_history) < self.min_trades:
            return {'status': 'insufficient_data', 'trade_count': len(self.trade_history)}
        
        score = self.calculate_score()
        if not score:
            return {'status': 'calculation_failed'}
        
        return {
            'status': 'ok',
            'trade_count': len(self.trade_history),
            'current_score': score.to_dict(),
            'score_7d': self._get_score_for_window(7),
            'score_30d': self._get_score_for_window(30)
        }
    
    def _get_score_for_window(self, days: int) -> Optional[Dict]:
        """Get score for specific window."""
        score = self.calculate_score(window_days=days)
        return score.to_dict() if score else None
    
    def should_trade(self, min_score: float = 0.4) -> Tuple[bool, str]:
        """
        Check if current efficiency warrants trading.
        
        Args:
            min_score: Minimum CES required to trade
            
        Returns:
            Tuple of (should_trade, reason)
        """
        score = self.calculate_score()
        
        if score is None:
            return True, "Insufficient data, allowing trades"
        
        if score.composite < min_score:
            return False, f"CES too low: {score.composite:.2f} < {min_score}"
        
        if score.trend == EfficiencyTrend.DEGRADING and score.percentile < 0.3:
            return False, f"Degrading efficiency in bottom 30%"
        
        return True, f"CES OK: {score.composite:.2f}"
