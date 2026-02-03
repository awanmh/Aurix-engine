"""
Health Reporter Configuration

Configurable thresholds for daily health reports.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ReporterThresholds:
    """Configurable thresholds for health verdict."""
    
    # Reality Score
    reality_score_min_avg: float = 0.7
    reality_score_min_floor: float = 0.5
    
    # Growth State
    preservation_state_max_pct: float = 30.0
    
    # Drawdown
    max_drawdown_warning: float = 6.0
    max_drawdown_critical: float = 8.0
    
    # Capital Fatigue Index
    cfi_warning: float = 0.7
    cfi_critical: float = 0.85
    
    # Trading
    min_trades_per_day: int = 1
    min_win_rate: float = 0.45


@dataclass
class ReporterConfig:
    """Health Reporter configuration."""
    
    enabled: bool = True
    interval_hours: int = 24
    report_dir: str = "reports/daily"
    
    thresholds: ReporterThresholds = field(default_factory=ReporterThresholds)
    
    critical_event_triggers: List[str] = field(default_factory=lambda: [
        "kill_switch_activated",
        "preservation_exceeded",
        "drawdown_critical"
    ])
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReporterConfig':
        """Create config from dictionary."""
        thresholds_data = data.get('thresholds', {})
        thresholds = ReporterThresholds(**thresholds_data)
        
        return cls(
            enabled=data.get('enabled', True),
            interval_hours=data.get('interval_hours', 24),
            report_dir=data.get('report_dir', 'reports/daily'),
            thresholds=thresholds,
            critical_event_triggers=data.get('critical_event_triggers', [])
        )
