"""
AURIX Capital Efficiency & Market Selection Layer

Maximizes capital growth speed without increasing risk through:
- Capital Efficiency Scoring (CES)
- Pair Ranking & Rotation
- Overtrading Detection
- Psychological Drift Proxy
- Capital Growth Orchestrator
"""

from .scorer import CapitalEfficiencyScorer, EfficiencyScore, TradeRecord
from .pair_manager import PairManager, PairRanking
from .overtrading import OvertradingDetector, OvertradingAlert
from .psych_drift import PsychDriftDetector, DriftState
from .gate import CapitalEfficiencyGate, GateResult
from .growth_orchestrator import (
    GrowthOrchestrator,
    GrowthState,
    GrowthParameters,
    CapitalFatigueIndex,
)

__all__ = [
    "CapitalEfficiencyScorer",
    "EfficiencyScore",
    "TradeRecord",
    "PairManager",
    "PairRanking",
    "OvertradingDetector",
    "OvertradingAlert",
    "PsychDriftDetector",
    "DriftState",
    "CapitalEfficiencyGate",
    "GateResult",
    "GrowthOrchestrator",
    "GrowthState",
    "GrowthParameters",
    "CapitalFatigueIndex",
]


