"""
AURIX Reality Validation & Anti-Overfitting Layer

Eliminates data leakage, overfitting illusions, and unrealistic backtest assumptions.
Forces AURIX to survive real-market conditions.

Modules:
- data_guard: Real data ingestion with lookahead prevention
- overfit_monitor: Train/forward divergence detection
- retrain_controller: Disciplined retraining logic
- stress_tester: Noise & stress injection
- slippage_model: Realistic execution costs
- kill_switch: Hard stop under dangerous conditions
- reality_score: Unified health score with attribution
- recovery_protocol: 3-phase gradual recovery
"""

from .config import RealityConfig
from .data_guard import DataGuard, DataGuardConfig
from .overfit_monitor import OverfitMonitor, OverfitState
from .retrain_controller import RetrainController, RetrainDecision
from .stress_tester import StressTester, StressConfig
from .slippage_model import SlippageModel, SlippageResult
from .kill_switch import KillSwitch, KillSwitchState
from .reality_score import RealityScorer, RealityScore, ScoreAttribution
from .recovery_protocol import RecoveryProtocol, RecoveryState, RecoveryPhase

__all__ = [
    "RealityConfig",
    "DataGuard",
    "DataGuardConfig",
    "OverfitMonitor",
    "OverfitState",
    "RetrainController",
    "RetrainDecision",
    "StressTester",
    "StressConfig",
    "SlippageModel",
    "SlippageResult",
    "KillSwitch",
    "KillSwitchState",
    "RealityScorer",
    "RealityScore",
    "ScoreAttribution",
    "RecoveryProtocol",
    "RecoveryState",
    "RecoveryPhase",
]

