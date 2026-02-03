"""
AURIX Testing Package

Contains testing utilities including:
- Failure simulation
- Integration tests
- Performance benchmarks
"""

from .failure_simulator import (
    FailureSimulator,
    FailureScenario,
    FailureType,
    RecoveryAction,
    SimulationResult,
    run_failure_tests
)

__all__ = [
    "FailureSimulator",
    "FailureScenario",
    "FailureType",
    "RecoveryAction",
    "SimulationResult",
    "run_failure_tests"
]
