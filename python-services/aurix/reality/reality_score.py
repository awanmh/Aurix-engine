"""
AURIX Reality Score

Unified Reality Score (0-1) combining multiple health dimensions.
Provides attribution and explainability for score changes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScoreTrend(Enum):
    """Trend direction for Reality Score."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"


class TradingRecommendation(Enum):
    """Trading recommendation based on Reality Score."""
    FULL = "full_trading"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    HALT = "halt"


@dataclass
class ScoreComponent:
    """Individual component of Reality Score with attribution."""
    name: str
    value: float  # 0.0 to 1.0
    weight: float  # Component weight
    weighted_contribution: float  # value * weight
    trend: str  # "up", "down", "stable"
    explanation: str  # Human-readable explanation
    is_primary_cause: bool = False  # True if main contributor to score drop


@dataclass
class ScoreAttribution:
    """Attribution explaining why score changed."""
    top_contributors: List[ScoreComponent]  # Ranked by impact
    primary_cause: Optional[str]  # Main reason for score change
    explanation: str  # Full explanation text
    recommendations: List[str]  # Action recommendations
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RealityScore:
    """Unified Reality Score with full explainability."""
    value: float  # 0.0 (critical) to 1.0 (healthy)
    components: Dict[str, ScoreComponent]
    trend: ScoreTrend
    recommendation: TradingRecommendation
    attribution: ScoreAttribution
    
    # Multipliers for degraded trading
    position_multiplier: float = 1.0
    frequency_multiplier: float = 1.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_log_string(self) -> str:
        """Get concise log string."""
        return (
            f"RealityScore={self.value:.2f} ({self.recommendation.value}) | "
            f"Pos={self.position_multiplier:.0%} Freq={self.frequency_multiplier:.0%} | "
            f"Primary: {self.attribution.primary_cause or 'None'}"
        )


class RealityScorer:
    """
    Reality Score Calculator
    
    Combines 5 health dimensions into unified score (0-1):
    1. Data Quality (20%) - From DataGuard
    2. Slippage Deviation (15%) - Actual vs expected slippage
    3. Stress Failure Rate (20%) - Trades failed under stress
    4. Overfit Penalty (25%) - Train/forward divergence
    5. Confidence Health (20%) - Rolling avg confidence
    
    Provides full attribution and explainability.
    """
    
    # Component weights (must sum to 1.0)
    WEIGHTS = {
        'data_quality': 0.20,
        'slippage_deviation': 0.15,
        'stress_failure_rate': 0.20,
        'overfit_penalty': 0.25,
        'confidence_health': 0.20
    }
    
    # Thresholds
    SOFT_THRESHOLD_POSITION = 0.7
    SOFT_THRESHOLD_FREQUENCY = 0.6
    HARD_THRESHOLD = 0.4
    
    def __init__(self, history_size: int = 50):
        """
        Initialize scorer.
        
        Args:
            history_size: Number of historical scores to keep
        """
        self.history_size = history_size
        self._score_history: List[RealityScore] = []
        self._component_history: Dict[str, List[float]] = {k: [] for k in self.WEIGHTS}
    
    def calculate_score(
        self,
        data_quality: float,
        slippage_deviation: float,
        stress_failure_rate: float,
        overfit_penalty: float,
        confidence_health: float
    ) -> RealityScore:
        """
        Calculate unified Reality Score.
        
        All inputs should be 0.0-1.0 where 1.0 = healthy.
        
        Args:
            data_quality: Data completeness/quality score
            slippage_deviation: 1.0 - (actual_slippage / expected_slippage) capped at 0
            stress_failure_rate: 1.0 - failure_rate
            overfit_penalty: 1.0 - divergence_penalty
            confidence_health: Rolling avg confidence normalized
            
        Returns:
            RealityScore with full attribution
        """
        # Build components
        raw_values = {
            'data_quality': np.clip(data_quality, 0, 1),
            'slippage_deviation': np.clip(slippage_deviation, 0, 1),
            'stress_failure_rate': np.clip(stress_failure_rate, 0, 1),
            'overfit_penalty': np.clip(overfit_penalty, 0, 1),
            'confidence_health': np.clip(confidence_health, 0, 1)
        }
        
        # Calculate weighted score
        weighted_sum = sum(
            raw_values[k] * self.WEIGHTS[k] 
            for k in self.WEIGHTS
        )
        
        # Build component objects with trends
        components = {}
        for name, value in raw_values.items():
            trend = self._calculate_component_trend(name, value)
            explanation = self._generate_component_explanation(name, value)
            
            components[name] = ScoreComponent(
                name=name,
                value=value,
                weight=self.WEIGHTS[name],
                weighted_contribution=value * self.WEIGHTS[name],
                trend=trend,
                explanation=explanation
            )
            
            # Update history
            self._component_history[name].append(value)
            if len(self._component_history[name]) > self.history_size:
                self._component_history[name] = self._component_history[name][-self.history_size:]
        
        # Calculate overall trend
        overall_trend = self._calculate_overall_trend(weighted_sum)
        
        # Get recommendation based on score
        recommendation = self._get_recommendation(weighted_sum)
        
        # Calculate multipliers
        position_mult, freq_mult = self._calculate_multipliers(weighted_sum)
        
        # Generate attribution
        attribution = self._generate_attribution(components, weighted_sum)
        
        # Build final score
        score = RealityScore(
            value=weighted_sum,
            components=components,
            trend=overall_trend,
            recommendation=recommendation,
            attribution=attribution,
            position_multiplier=position_mult,
            frequency_multiplier=freq_mult
        )
        
        # Store in history
        self._score_history.append(score)
        if len(self._score_history) > self.history_size:
            self._score_history = self._score_history[-self.history_size:]
        
        # Log
        self._log_score(score)
        
        return score
    
    def _calculate_component_trend(self, name: str, current: float) -> str:
        """Calculate trend for single component."""
        history = self._component_history.get(name, [])
        
        if len(history) < 3:
            return "stable"
        
        recent_avg = np.mean(history[-3:])
        
        if current > recent_avg + 0.05:
            return "up"
        elif current < recent_avg - 0.05:
            return "down"
        return "stable"
    
    def _calculate_overall_trend(self, current_score: float) -> ScoreTrend:
        """Calculate overall score trend."""
        if len(self._score_history) < 3:
            return ScoreTrend.STABLE
        
        recent = [s.value for s in self._score_history[-5:]]
        avg = np.mean(recent)
        
        if current_score < self.HARD_THRESHOLD:
            return ScoreTrend.CRITICAL
        elif current_score > avg + 0.05:
            return ScoreTrend.IMPROVING
        elif current_score < avg - 0.05:
            return ScoreTrend.DEGRADING
        return ScoreTrend.STABLE
    
    def _get_recommendation(self, score: float) -> TradingRecommendation:
        """Get trading recommendation based on score."""
        if score >= 0.8:
            return TradingRecommendation.FULL
        elif score >= 0.6:
            return TradingRecommendation.REDUCED
        elif score >= self.HARD_THRESHOLD:
            return TradingRecommendation.MINIMAL
        return TradingRecommendation.HALT
    
    def _calculate_multipliers(self, score: float) -> Tuple[float, float]:
        """Calculate position and frequency multipliers."""
        if score >= 0.8:
            return 1.0, 1.0
        elif score >= self.SOFT_THRESHOLD_POSITION:
            # Linear degradation 0.7-0.8 -> 80%-100%
            pos_mult = 0.8 + (score - 0.7) * 2.0
            return pos_mult, 1.0
        elif score >= self.SOFT_THRESHOLD_FREQUENCY:
            # 0.6-0.7 -> pos 60-80%, freq 80-100%
            pos_mult = 0.6 + (score - 0.6) * 2.0
            freq_mult = 0.8 + (score - 0.6) * 2.0
            return pos_mult, freq_mult
        elif score >= self.HARD_THRESHOLD:
            # 0.4-0.6 -> pos 30-60%, freq 50-80%
            pos_mult = 0.3 + (score - 0.4) * 1.5
            freq_mult = 0.5 + (score - 0.4) * 1.5
            return pos_mult, freq_mult
        # Below hard threshold
        return 0.0, 0.0
    
    def _generate_component_explanation(self, name: str, value: float) -> str:
        """Generate human-readable explanation for component."""
        explanations = {
            'data_quality': {
                'high': "Data quality is excellent with minimal gaps",
                'medium': "Data has some gaps or quality issues",
                'low': "Significant data quality problems detected"
            },
            'slippage_deviation': {
                'high': "Slippage within expected range",
                'medium': "Slippage slightly higher than expected",
                'low': "Execution costs significantly higher than modeled"
            },
            'stress_failure_rate': {
                'high': "System handling stress conditions well",
                'medium': "Some trades failing under stress",
                'low': "High failure rate under stress conditions"
            },
            'overfit_penalty': {
                'high': "Model generalizing well to new data",
                'medium': "Some train/forward performance gap",
                'low': "Significant overfitting detected"
            },
            'confidence_health': {
                'high': "Model confidence levels healthy",
                'medium': "Confidence levels declining",
                'low': "Confidence collapse detected"
            }
        }
        
        if value >= 0.7:
            level = 'high'
        elif value >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return explanations.get(name, {}).get(level, f"{name}: {value:.1%}")
    
    def _generate_attribution(
        self,
        components: Dict[str, ScoreComponent],
        score: float
    ) -> ScoreAttribution:
        """Generate full attribution with ranked contributors."""
        # Rank components by impact (lower value = more impact on score drop)
        ranked = sorted(
            components.values(),
            key=lambda c: c.value
        )
        
        # Top 3 contributors to low score
        top_contributors = ranked[:3]
        
        # Mark primary cause
        if ranked[0].value < 0.5:
            ranked[0].is_primary_cause = True
            primary_cause = self._format_component_name(ranked[0].name)
        else:
            primary_cause = None
        
        # Generate explanation
        if score >= 0.8:
            explanation = "System operating normally. All health metrics within acceptable range."
        elif score >= 0.6:
            weak_components = [c.name for c in ranked if c.value < 0.7][:2]
            explanation = f"Slight degradation detected in: {', '.join(self._format_component_name(c) for c in weak_components)}."
        elif score >= self.HARD_THRESHOLD:
            weak_components = [c.name for c in ranked if c.value < 0.5][:3]
            explanation = f"Significant issues in: {', '.join(self._format_component_name(c) for c in weak_components)}. Trading restricted."
        else:
            explanation = f"CRITICAL: Primary issue is {primary_cause}. Trading halted."
        
        # Generate recommendations
        recommendations = self._generate_recommendations(ranked, score)
        
        return ScoreAttribution(
            top_contributors=top_contributors,
            primary_cause=primary_cause,
            explanation=explanation,
            recommendations=recommendations
        )
    
    def _format_component_name(self, name: str) -> str:
        """Format component name for display."""
        return name.replace('_', ' ').title()
    
    def _generate_recommendations(
        self,
        ranked_components: List[ScoreComponent],
        score: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for comp in ranked_components[:3]:
            if comp.value < 0.5:
                if comp.name == 'data_quality':
                    recommendations.append("Review data source for gaps or quality issues")
                elif comp.name == 'slippage_deviation':
                    recommendations.append("Adjust slippage model parameters or reduce order sizes")
                elif comp.name == 'stress_failure_rate':
                    recommendations.append("Reduce stress intensity or improve error handling")
                elif comp.name == 'overfit_penalty':
                    recommendations.append("Consider model retrain with more recent data")
                elif comp.name == 'confidence_health':
                    recommendations.append("Review prediction calibration or wait for market stabilization")
        
        if score < self.HARD_THRESHOLD:
            recommendations.insert(0, "CRITICAL: Wait for automatic recovery protocol")
        
        return recommendations[:5]  # Max 5 recommendations
    
    def _log_score(self, score: RealityScore):
        """Log score with appropriate level."""
        if score.value >= 0.8:
            logger.info(f"[REALITY] {score.to_log_string()}")
        elif score.value >= 0.6:
            logger.warning(f"[REALITY] {score.to_log_string()}")
        elif score.value >= self.HARD_THRESHOLD:
            logger.warning(f"[REALITY] {score.to_log_string()}")
        else:
            logger.critical(f"[REALITY] {score.to_log_string()}")
    
    def get_score_history(self, limit: int = 20) -> List[RealityScore]:
        """Get recent score history."""
        return self._score_history[-limit:]
    
    def get_component_history(self, component: str, limit: int = 20) -> List[float]:
        """Get history for single component."""
        return self._component_history.get(component, [])[-limit:]
    
    def generate_postmortem(self, since: Optional[datetime] = None) -> str:
        """
        Generate post-mortem analysis of score degradation.
        
        Args:
            since: Analyze scores since this time (default: last 24h)
            
        Returns:
            Markdown-formatted post-mortem report
        """
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        relevant_scores = [s for s in self._score_history if s.timestamp >= since]
        
        if not relevant_scores:
            return "No score data available for the specified period."
        
        # Find lowest point
        lowest = min(relevant_scores, key=lambda s: s.value)
        highest = max(relevant_scores, key=lambda s: s.value)
        current = relevant_scores[-1]
        
        lines = [
            "# Reality Score Post-Mortem",
            f"\n**Period**: {since.isoformat()} to {datetime.now().isoformat()}",
            f"\n**Scores Analyzed**: {len(relevant_scores)}",
            "",
            "## Summary",
            f"- **Lowest Score**: {lowest.value:.2f} at {lowest.timestamp}",
            f"- **Highest Score**: {highest.value:.2f}",
            f"- **Current Score**: {current.value:.2f}",
            "",
            "## Primary Issues",
        ]
        
        # Aggregate primary causes
        causes = {}
        for score in relevant_scores:
            if score.attribution.primary_cause:
                causes[score.attribution.primary_cause] = causes.get(score.attribution.primary_cause, 0) + 1
        
        for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cause}**: {count} occurrences")
        
        lines.extend([
            "",
            "## Recommendations",
        ])
        
        # Aggregate recommendations
        all_recommendations = set()
        for score in relevant_scores[-10:]:
            all_recommendations.update(score.attribution.recommendations)
        
        for rec in list(all_recommendations)[:5]:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def suggest_parameter_tuning(self) -> Dict[str, str]:
        """
        Suggest parameter adjustments based on score history.
        
        Returns:
            Dict of parameter -> suggested adjustment
        """
        suggestions = {}
        
        for name, history in self._component_history.items():
            if len(history) < 10:
                continue
            
            avg = np.mean(history[-10:])
            
            if avg < 0.5:
                if name == 'slippage_deviation':
                    suggestions['base_slippage_pct'] = "Increase by 50% to match reality"
                elif name == 'stress_failure_rate':
                    suggestions['stress_intensity'] = "Reduce from 0.5 to 0.3"
                elif name == 'overfit_penalty':
                    suggestions['min_retrain_cooldown_days'] = "Reduce from 7 to 5 days"
                elif name == 'confidence_health':
                    suggestions['base_confidence_threshold'] = "Increase minimum threshold"
        
        return suggestions
