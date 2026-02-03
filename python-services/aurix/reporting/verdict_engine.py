"""
Verdict Engine

Determines overall system health: HEALTHY, WARNING, or CRITICAL.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .config import ReporterThresholds


class Verdict(Enum):
    """Health verdict levels."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class VerdictResult:
    """Result of health verdict evaluation."""
    verdict: Verdict
    reasons: List[str]
    recommendations: List[str]
    
    @property
    def emoji(self) -> str:
        """Get emoji for verdict."""
        return {
            Verdict.HEALTHY: "✅",
            Verdict.WARNING: "⚠️",
            Verdict.CRITICAL: "❌"
        }[self.verdict]
    
    @property
    def message(self) -> str:
        """Get summary message for verdict."""
        return {
            Verdict.HEALTHY: "All constraints satisfied. Continue validation.",
            Verdict.WARNING: "Soft degradation detected. Monitor closely.",
            Verdict.CRITICAL: "Validation should be paused. Manual review required."
        }[self.verdict]


class VerdictEngine:
    """
    Evaluates metrics and determines system health verdict.
    
    Rules:
    - CRITICAL: Any critical condition triggers immediate pause
    - WARNING: Soft degradations that need monitoring
    - HEALTHY: All constraints satisfied
    """
    
    def __init__(self, thresholds: ReporterThresholds = None):
        self.thresholds = thresholds or ReporterThresholds()
    
    def evaluate(self, metrics) -> VerdictResult:
        """
        Evaluate metrics and return verdict.
        
        Args:
            metrics: DailyMetrics object
            
        Returns:
            VerdictResult with verdict, reasons, and recommendations
        """
        critical_reasons = []
        warning_reasons = []
        recommendations = []
        
        # Check for critical conditions
        critical_reasons.extend(self._check_critical(metrics))
        
        # Check for warning conditions
        warning_reasons.extend(self._check_warnings(metrics))
        
        # Determine verdict
        if critical_reasons:
            verdict = Verdict.CRITICAL
            recommendations = self._get_critical_recommendations(critical_reasons)
        elif warning_reasons:
            verdict = Verdict.WARNING
            recommendations = self._get_warning_recommendations(warning_reasons)
        else:
            verdict = Verdict.HEALTHY
            recommendations = ["Continue monitoring. All systems nominal."]
        
        all_reasons = critical_reasons + warning_reasons
        
        return VerdictResult(
            verdict=verdict,
            reasons=all_reasons if all_reasons else ["No issues detected"],
            recommendations=recommendations
        )
    
    def _check_critical(self, metrics) -> List[str]:
        """Check for critical conditions that require immediate action."""
        reasons = []
        
        # Kill switch triggered
        if metrics.risk.kill_switch_triggered:
            reasons.append(
                f"Kill Switch ACTIVATED: {metrics.risk.kill_switch_reason or 'Unknown reason'}"
            )
        
        # Critical drawdown
        if metrics.risk.max_drawdown_24h >= self.thresholds.max_drawdown_critical:
            reasons.append(
                f"Drawdown CRITICAL: {metrics.risk.max_drawdown_24h:.1f}% "
                f"(threshold: {self.thresholds.max_drawdown_critical}%)"
            )
        
        # Reality score floor breach
        if metrics.reality.min_score < self.thresholds.reality_score_min_floor:
            reasons.append(
                f"Reality Score FLOOR BREACH: {metrics.reality.min_score:.2f} "
                f"(min: {self.thresholds.reality_score_min_floor})"
            )
        
        # CFI critical
        if metrics.risk.cfi_max >= self.thresholds.cfi_critical:
            reasons.append(
                f"Capital Fatigue CRITICAL: {metrics.risk.cfi_max:.2f} "
                f"(threshold: {self.thresholds.cfi_critical})"
            )
        
        # System dead
        if metrics.liveness.status == "DEAD":
            reasons.append("System DEAD: Redis or data pipeline failure")
        
        return reasons
    
    def _check_warnings(self, metrics) -> List[str]:
        """Check for warning conditions that need monitoring."""
        reasons = []
        
        # Reality score below average threshold
        if metrics.reality.avg_score < self.thresholds.reality_score_min_avg:
            reasons.append(
                f"Reality Score LOW: {metrics.reality.avg_score:.2f} "
                f"(threshold: {self.thresholds.reality_score_min_avg})"
            )
        
        # Drawdown warning
        if (metrics.risk.max_drawdown_24h >= self.thresholds.max_drawdown_warning and
            metrics.risk.max_drawdown_24h < self.thresholds.max_drawdown_critical):
            reasons.append(
                f"Drawdown WARNING: {metrics.risk.max_drawdown_24h:.1f}% "
                f"(warning: {self.thresholds.max_drawdown_warning}%)"
            )
        
        # CFI warning
        if (metrics.risk.cfi_avg >= self.thresholds.cfi_warning and
            metrics.risk.cfi_max < self.thresholds.cfi_critical):
            reasons.append(
                f"Capital Fatigue HIGH: {metrics.risk.cfi_avg:.2f} "
                f"(threshold: {self.thresholds.cfi_warning})"
            )
        
        # Preservation state exceeded
        if metrics.growth_state.preservation_exceeded:
            reasons.append(
                f"Preservation state EXCEEDED: {metrics.growth_state.preservation_pct:.1f}% "
                f"(max: 30%)"
            )
        
        # Grinding detected
        if metrics.risk.grinding_detected:
            reasons.append("Grinding phase DETECTED: Low return per risk")
        
        # System degraded
        if metrics.liveness.status == "DEGRADED":
            reasons.append(
                f"System DEGRADED: Process rate {metrics.liveness.process_rate:.1%}"
            )
        
        # Low win rate
        if (metrics.trading.trades_executed >= 5 and 
            metrics.trading.win_rate < self.thresholds.min_win_rate):
            reasons.append(
                f"Win rate LOW: {metrics.trading.win_rate:.1%} "
                f"(threshold: {self.thresholds.min_win_rate:.1%})"
            )
        
        # Negative PnL trend (if previous data available)
        if metrics.trend.has_previous:
            if metrics.trend.reality_score_delta < -0.1:
                reasons.append(
                    f"Reality Score DECLINING: {metrics.trend.reality_score_delta:+.2f} vs previous"
                )
        
        return reasons
    
    def _get_critical_recommendations(self, reasons: List[str]) -> List[str]:
        """Get recommendations for critical conditions."""
        recommendations = [
            "PAUSE validation immediately",
            "Review kill switch trigger conditions",
            "Check for market regime changes",
            "Verify data integrity"
        ]
        
        if any("Drawdown" in r for r in reasons):
            recommendations.append("Consider reducing position sizes")
        
        if any("Reality Score" in r for r in reasons):
            recommendations.append("Review model performance metrics")
        
        return recommendations[:4]  # Limit to 4
    
    def _get_warning_recommendations(self, reasons: List[str]) -> List[str]:
        """Get recommendations for warning conditions."""
        recommendations = ["Continue monitoring closely"]
        
        if any("Preservation" in r for r in reasons):
            recommendations.append("Consider waiting for market conditions to improve")
        
        if any("Grinding" in r for r in reasons):
            recommendations.append("Capital efficiency may be declining")
        
        if any("Win rate" in r for r in reasons):
            recommendations.append("Review trade selection criteria")
        
        if any("DECLINING" in r for r in reasons):
            recommendations.append("Trend deteriorating - watch for critical threshold")
        
        return recommendations[:3]  # Limit to 3
