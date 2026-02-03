"""
AURIX Recovery Protocol

3-phase gradual recovery from trading halt:
1. Cooldown - Mandatory wait period
2. Validation - Micro trades to confirm model health
3. Ramp-Up - Gradual return to full position size
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """Recovery phases after trading halt."""
    NORMAL = "normal"        # Full trading
    COOLDOWN = "cooldown"    # Mandatory wait
    VALIDATION = "validation"  # Micro trades
    RAMPUP = "rampup"        # Position ramp-up
    HALTED = "halted"        # Trading stopped


@dataclass
class ValidationTrade:
    """Record of validation trade during recovery."""
    timestamp: datetime
    symbol: str
    direction: str
    pnl: float
    is_win: bool


@dataclass
class RecoveryState:
    """Current recovery state."""
    phase: RecoveryPhase
    phase_start: datetime
    phase_progress: float  # 0.0 to 1.0
    position_multiplier: float
    frequency_multiplier: float
    
    # Cooldown info
    cooldown_hours_remaining: float = 0.0
    
    # Validation info
    validation_trades_done: int = 0
    validation_trades_required: int = 10
    validation_wins: int = 0
    validation_winrate: float = 0.0
    
    # Ramp-up info
    rampup_stage: int = 0
    rampup_stages_total: int = 4
    rampup_hours_remaining: float = 0.0
    
    # Reason for current state
    halt_reason: Optional[str] = None
    
    def to_log_string(self) -> str:
        """Get concise log string."""
        if self.phase == RecoveryPhase.NORMAL:
            return "Recovery: NORMAL (full trading)"
        elif self.phase == RecoveryPhase.COOLDOWN:
            return f"Recovery: COOLDOWN ({self.cooldown_hours_remaining:.1f}h remaining)"
        elif self.phase == RecoveryPhase.VALIDATION:
            return f"Recovery: VALIDATION ({self.validation_trades_done}/{self.validation_trades_required}, WR={self.validation_winrate:.0%})"
        elif self.phase == RecoveryPhase.RAMPUP:
            return f"Recovery: RAMPUP (stage {self.rampup_stage}/{self.rampup_stages_total}, pos={self.position_multiplier:.0%})"
        return f"Recovery: HALTED ({self.halt_reason})"


class RecoveryProtocol:
    """
    3-Phase Recovery Protocol
    
    Phase 1: COOLDOWN
    - Mandatory wait period after halt
    - No trading allowed
    - Allows market to stabilize
    
    Phase 2: VALIDATION
    - Micro trades with reduced position size
    - Must achieve minimum win rate
    - Failure restarts cooldown
    
    Phase 3: RAMP-UP
    - Gradual position size increase
    - 4 stages: 25% -> 50% -> 75% -> 100%
    - Each stage has minimum duration
    
    Phase 4: NORMAL
    - Full trading resumed
    """
    
    def __init__(
        self,
        cooldown_hours: float = 12.0,
        validation_trades: int = 10,
        validation_position_pct: float = 0.10,
        validation_min_winrate: float = 0.50,
        rampup_stages: List[float] = None,
        rampup_hours_per_stage: float = 6.0
    ):
        """
        Initialize recovery protocol.
        
        Args:
            cooldown_hours: Hours to wait in cooldown phase
            validation_trades: Number of validation trades required
            validation_position_pct: Position size during validation (0-1)
            validation_min_winrate: Minimum win rate to pass validation
            rampup_stages: Position multipliers for each ramp-up stage
            rampup_hours_per_stage: Hours per ramp-up stage
        """
        self.cooldown_hours = cooldown_hours
        self.validation_trades = validation_trades
        self.validation_position_pct = validation_position_pct
        self.validation_min_winrate = validation_min_winrate
        self.rampup_stages = rampup_stages or [0.25, 0.50, 0.75, 1.0]
        self.rampup_hours_per_stage = rampup_hours_per_stage
        
        # State
        self._phase = RecoveryPhase.NORMAL
        self._phase_start: Optional[datetime] = None
        self._halt_reason: Optional[str] = None
        
        # Validation tracking
        self._validation_trades: List[ValidationTrade] = []
        
        # Ramp-up tracking
        self._rampup_stage = 0
        self._rampup_stage_start: Optional[datetime] = None
        
        # Recovery history
        self._recovery_history: List[Dict] = []
    
    @property
    def phase(self) -> RecoveryPhase:
        """Get current phase."""
        return self._phase
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self._phase in [
            RecoveryPhase.NORMAL, 
            RecoveryPhase.VALIDATION, 
            RecoveryPhase.RAMPUP
        ]
    
    def trigger_halt(self, reason: str):
        """
        Trigger trading halt - starts recovery process.
        
        Args:
            reason: Reason for halt
        """
        logger.critical(f"[RECOVERY] HALT triggered: {reason}")
        
        self._phase = RecoveryPhase.HALTED
        self._phase_start = datetime.now()
        self._halt_reason = reason
        
        # Clear validation trades
        self._validation_trades = []
        
        # Record in history
        self._recovery_history.append({
            'type': 'halt',
            'reason': reason,
            'timestamp': datetime.now()
        })
        
        # Auto-start cooldown
        self._start_cooldown()
    
    def _start_cooldown(self):
        """Start cooldown phase."""
        logger.warning(f"[RECOVERY] Starting COOLDOWN phase ({self.cooldown_hours}h)")
        
        self._phase = RecoveryPhase.COOLDOWN
        self._phase_start = datetime.now()
    
    def update(self) -> RecoveryState:
        """
        Update recovery state based on time and conditions.
        
        Call this periodically to advance phases.
        
        Returns:
            Current RecoveryState
        """
        now = datetime.now()
        
        if self._phase == RecoveryPhase.COOLDOWN:
            self._update_cooldown(now)
        elif self._phase == RecoveryPhase.RAMPUP:
            self._update_rampup(now)
        
        return self.get_state()
    
    def _update_cooldown(self, now: datetime):
        """Check if cooldown is complete."""
        if self._phase_start is None:
            return
        
        elapsed = (now - self._phase_start).total_seconds() / 3600
        
        if elapsed >= self.cooldown_hours:
            logger.info("[RECOVERY] Cooldown complete, starting VALIDATION phase")
            self._phase = RecoveryPhase.VALIDATION
            self._phase_start = now
            self._validation_trades = []
    
    def _update_rampup(self, now: datetime):
        """Check if ramp-up stage is complete."""
        if self._rampup_stage_start is None:
            return
        
        elapsed = (now - self._rampup_stage_start).total_seconds() / 3600
        
        if elapsed >= self.rampup_hours_per_stage:
            self._rampup_stage += 1
            self._rampup_stage_start = now
            
            if self._rampup_stage >= len(self.rampup_stages):
                logger.info("[RECOVERY] Ramp-up complete, resuming NORMAL trading")
                self._phase = RecoveryPhase.NORMAL
                self._phase_start = now
                
                self._recovery_history.append({
                    'type': 'recovered',
                    'timestamp': now
                })
            else:
                new_mult = self.rampup_stages[self._rampup_stage]
                logger.info(f"[RECOVERY] Ramp-up stage {self._rampup_stage + 1}/{len(self.rampup_stages)}: position size = {new_mult:.0%}")
    
    def record_validation_trade(
        self,
        symbol: str,
        direction: str,
        pnl: float
    ):
        """
        Record a validation trade.
        
        Args:
            symbol: Trading pair
            direction: LONG or SHORT
            pnl: Profit/loss
        """
        if self._phase != RecoveryPhase.VALIDATION:
            return
        
        trade = ValidationTrade(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            pnl=pnl,
            is_win=pnl > 0
        )
        
        self._validation_trades.append(trade)
        
        logger.info(f"[RECOVERY] Validation trade: {'WIN' if trade.is_win else 'LOSS'} ({len(self._validation_trades)}/{self.validation_trades})")
        
        # Check if validation complete
        if len(self._validation_trades) >= self.validation_trades:
            self._evaluate_validation()
    
    def _evaluate_validation(self):
        """Evaluate validation trades and decide next phase."""
        wins = sum(1 for t in self._validation_trades if t.is_win)
        winrate = wins / len(self._validation_trades) if self._validation_trades else 0
        
        logger.info(f"[RECOVERY] Validation complete: {wins}/{len(self._validation_trades)} wins ({winrate:.0%})")
        
        if winrate >= self.validation_min_winrate:
            # Passed validation - start ramp-up
            logger.info("[RECOVERY] Validation PASSED, starting RAMPUP phase")
            self._phase = RecoveryPhase.RAMPUP
            self._phase_start = datetime.now()
            self._rampup_stage = 0
            self._rampup_stage_start = datetime.now()
            
            self._recovery_history.append({
                'type': 'validation_passed',
                'winrate': winrate,
                'timestamp': datetime.now()
            })
        else:
            # Failed validation - restart cooldown
            logger.warning(f"[RECOVERY] Validation FAILED ({winrate:.0%} < {self.validation_min_winrate:.0%}), restarting cooldown")
            self._start_cooldown()
            
            self._recovery_history.append({
                'type': 'validation_failed',
                'winrate': winrate,
                'timestamp': datetime.now()
            })
    
    def get_state(self) -> RecoveryState:
        """Get current recovery state."""
        now = datetime.now()
        
        # Calculate phase progress
        phase_progress = 0.0
        cooldown_remaining = 0.0
        rampup_remaining = 0.0
        
        if self._phase == RecoveryPhase.COOLDOWN and self._phase_start:
            elapsed = (now - self._phase_start).total_seconds() / 3600
            phase_progress = min(1.0, elapsed / self.cooldown_hours)
            cooldown_remaining = max(0, self.cooldown_hours - elapsed)
        
        elif self._phase == RecoveryPhase.VALIDATION:
            phase_progress = len(self._validation_trades) / self.validation_trades
        
        elif self._phase == RecoveryPhase.RAMPUP and self._rampup_stage_start:
            elapsed = (now - self._rampup_stage_start).total_seconds() / 3600
            stage_progress = min(1.0, elapsed / self.rampup_hours_per_stage)
            overall_progress = (self._rampup_stage + stage_progress) / len(self.rampup_stages)
            phase_progress = overall_progress
            rampup_remaining = (len(self.rampup_stages) - self._rampup_stage - stage_progress) * self.rampup_hours_per_stage
        
        # Calculate multipliers
        pos_mult, freq_mult = self._get_current_multipliers()
        
        # Validation stats
        validation_wins = sum(1 for t in self._validation_trades if t.is_win)
        validation_winrate = validation_wins / len(self._validation_trades) if self._validation_trades else 0
        
        return RecoveryState(
            phase=self._phase,
            phase_start=self._phase_start or now,
            phase_progress=phase_progress,
            position_multiplier=pos_mult,
            frequency_multiplier=freq_mult,
            cooldown_hours_remaining=cooldown_remaining,
            validation_trades_done=len(self._validation_trades),
            validation_trades_required=self.validation_trades,
            validation_wins=validation_wins,
            validation_winrate=validation_winrate,
            rampup_stage=self._rampup_stage,
            rampup_stages_total=len(self.rampup_stages),
            rampup_hours_remaining=rampup_remaining,
            halt_reason=self._halt_reason
        )
    
    def _get_current_multipliers(self) -> Tuple[float, float]:
        """Get current position and frequency multipliers."""
        if self._phase == RecoveryPhase.NORMAL:
            return 1.0, 1.0
        elif self._phase == RecoveryPhase.VALIDATION:
            return self.validation_position_pct, 1.0
        elif self._phase == RecoveryPhase.RAMPUP:
            pos_mult = self.rampup_stages[min(self._rampup_stage, len(self.rampup_stages) - 1)]
            return pos_mult, 1.0
        # HALTED or COOLDOWN
        return 0.0, 0.0
    
    def force_resume(self, reason: str = "Manual override"):
        """
        Force resume normal trading (USE WITH CAUTION).
        
        Args:
            reason: Reason for manual override
        """
        logger.warning(f"[RECOVERY] FORCED RESUME: {reason}")
        
        self._phase = RecoveryPhase.NORMAL
        self._phase_start = datetime.now()
        self._halt_reason = None
        
        self._recovery_history.append({
            'type': 'forced_resume',
            'reason': reason,
            'timestamp': datetime.now()
        })
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recovery history."""
        return self._recovery_history[-limit:]
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        state = self.get_state()
        
        lines = [
            "=== Recovery Protocol Status ===",
            f"Phase: {state.phase.value.upper()}",
            f"Progress: {state.phase_progress:.0%}",
            f"Position Multiplier: {state.position_multiplier:.0%}",
            f"Frequency Multiplier: {state.frequency_multiplier:.0%}",
        ]
        
        if state.phase == RecoveryPhase.COOLDOWN:
            lines.append(f"Cooldown Remaining: {state.cooldown_hours_remaining:.1f}h")
        elif state.phase == RecoveryPhase.VALIDATION:
            lines.append(f"Validation: {state.validation_trades_done}/{state.validation_trades_required}")
            lines.append(f"Current Win Rate: {state.validation_winrate:.0%}")
        elif state.phase == RecoveryPhase.RAMPUP:
            lines.append(f"Ramp-Up Stage: {state.rampup_stage + 1}/{state.rampup_stages_total}")
            lines.append(f"Time Remaining: {state.rampup_hours_remaining:.1f}h")
        
        if state.halt_reason:
            lines.append(f"Halt Reason: {state.halt_reason}")
        
        return "\n".join(lines)
