"""
AURIX Validation Package

Capital validation mode for testnet trials.
"""

from .capital_validator import (
    CapitalValidationMode,
    CapitalTrustScore,
    EquityCurveAnalyzer,
    EquityCurveMetrics,
    ExpectancyTracker,
    ExpectancyDrift,
    DegradationDetector,
    DegradationPattern,
    ValidationPhase,
    ValidationState,
    TrustLevel
)

__all__ = [
    "CapitalValidationMode",
    "CapitalTrustScore",
    "EquityCurveAnalyzer",
    "EquityCurveMetrics",
    "ExpectancyTracker",
    "ExpectancyDrift",
    "DegradationDetector",
    "DegradationPattern",
    "ValidationPhase",
    "ValidationState",
    "TrustLevel"
]
