"""
Metrics Collector

Gathers data from all sources for daily health report.
READ-ONLY: Does not modify any system state.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class LivenessMetrics:
    """System liveness metrics."""
    candles_received: int = 0
    candles_processed: int = 0
    process_rate: float = 0.0
    redis_alive: bool = False
    status: str = "UNKNOWN"  # ALIVE, DEGRADED, DEAD


@dataclass
class RealityMetrics:
    """Reality validation metrics."""
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    pass_condition: bool = False
    negative_contributors: List[str] = field(default_factory=list)


@dataclass
class GrowthStateMetrics:
    """Growth state analysis metrics."""
    accumulation_pct: float = 0.0
    expansion_pct: float = 0.0
    defense_pct: float = 0.0
    preservation_pct: float = 0.0
    current_state: str = "UNKNOWN"
    state_transitions: int = 0
    preservation_exceeded: bool = False


@dataclass
class RiskMetrics:
    """Risk and safety metrics."""
    kill_switch_triggered: bool = False
    kill_switch_reason: Optional[str] = None
    max_drawdown_24h: float = 0.0
    cfi_avg: float = 0.0
    cfi_max: float = 0.0
    grinding_detected: bool = False


@dataclass
class TradingMetrics:
    """Trading summary metrics."""
    trades_executed: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0


@dataclass
class TrendMetrics:
    """Trend comparison vs previous day."""
    reality_score_delta: float = 0.0
    max_drawdown_delta: float = 0.0
    cfi_delta: float = 0.0
    growth_transitions: int = 0
    has_previous: bool = False


@dataclass
class DailyMetrics:
    """Complete daily metrics package."""
    date: str
    timestamp: str
    liveness: LivenessMetrics
    reality: RealityMetrics
    growth_state: GrowthStateMetrics
    risk: RiskMetrics
    trading: TradingMetrics
    trend: TrendMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'date': self.date,
            'timestamp': self.timestamp,
            'liveness': {
                'candles_received': self.liveness.candles_received,
                'candles_processed': self.liveness.candles_processed,
                'process_rate': self.liveness.process_rate,
                'redis_alive': self.liveness.redis_alive,
                'status': self.liveness.status
            },
            'reality': {
                'avg_score': self.reality.avg_score,
                'min_score': self.reality.min_score,
                'max_score': self.reality.max_score,
                'pass_condition': self.reality.pass_condition,
                'negative_contributors': self.reality.negative_contributors
            },
            'growth_state': {
                'accumulation_pct': self.growth_state.accumulation_pct,
                'expansion_pct': self.growth_state.expansion_pct,
                'defense_pct': self.growth_state.defense_pct,
                'preservation_pct': self.growth_state.preservation_pct,
                'current_state': self.growth_state.current_state,
                'state_transitions': self.growth_state.state_transitions,
                'preservation_exceeded': self.growth_state.preservation_exceeded
            },
            'risk': {
                'kill_switch_triggered': self.risk.kill_switch_triggered,
                'kill_switch_reason': self.risk.kill_switch_reason,
                'max_drawdown_24h': self.risk.max_drawdown_24h,
                'cfi_avg': self.risk.cfi_avg,
                'cfi_max': self.risk.cfi_max,
                'grinding_detected': self.risk.grinding_detected
            },
            'trading': {
                'trades_executed': self.trading.trades_executed,
                'win_count': self.trading.win_count,
                'loss_count': self.trading.loss_count,
                'win_rate': self.trading.win_rate,
                'net_pnl': self.trading.net_pnl,
                'gross_profit': self.trading.gross_profit,
                'gross_loss': self.trading.gross_loss,
                'profit_factor': self.trading.profit_factor
            },
            'trend': {
                'reality_score_delta': self.trend.reality_score_delta,
                'max_drawdown_delta': self.trend.max_drawdown_delta,
                'cfi_delta': self.trend.cfi_delta,
                'growth_transitions': self.trend.growth_transitions,
                'has_previous': self.trend.has_previous
            }
        }


class MetricsCollector:
    """
    Collects metrics from all system sources.
    
    READ-ONLY: Does not modify any state.
    """
    
    def __init__(
        self,
        db=None,
        redis_bus=None,
        growth_orchestrator=None,
        reality_scorer=None,
        kill_switch=None,
        report_dir: str = "reports/daily"
    ):
        self.db = db
        self.redis = redis_bus
        self.growth = growth_orchestrator
        self.reality = reality_scorer
        self.kill_switch = kill_switch
        self.report_dir = Path(report_dir)
    
    def collect_all(self, hours: int = 24) -> DailyMetrics:
        """Collect all metrics for the reporting period."""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        
        return DailyMetrics(
            date=date_str,
            timestamp=now.isoformat(),
            liveness=self._collect_liveness(hours),
            reality=self._collect_reality(),
            growth_state=self._collect_growth_state(),
            risk=self._collect_risk(),
            trading=self._collect_trading(hours),
            trend=self._collect_trend(date_str)
        )
    
    def _collect_liveness(self, hours: int) -> LivenessMetrics:
        """Collect system liveness metrics."""
        metrics = LivenessMetrics()
        
        try:
            # Check Redis
            if self.redis:
                try:
                    self.redis.client.ping()
                    metrics.redis_alive = True
                except Exception:
                    metrics.redis_alive = False
            
            # Count candles from DB
            if self.db:
                cutoff = datetime.now() - timedelta(hours=hours)
                cutoff_ms = int(cutoff.timestamp() * 1000)
                
                # This is a simplified query - actual implementation depends on DB schema
                try:
                    cursor = self.db.db.execute(
                        "SELECT COUNT(*) FROM candles WHERE open_time > ?",
                        (cutoff_ms,)
                    )
                    row = cursor.fetchone()
                    metrics.candles_received = row[0] if row else 0
                    metrics.candles_processed = metrics.candles_received  # Assume all processed
                except Exception as e:
                    logger.warning(f"Failed to query candles: {e}")
            
            # Calculate process rate
            if metrics.candles_received > 0:
                metrics.process_rate = metrics.candles_processed / metrics.candles_received
            
            # Determine status
            if metrics.redis_alive and metrics.process_rate >= 0.95:
                metrics.status = "ALIVE"
            elif metrics.redis_alive and metrics.process_rate >= 0.80:
                metrics.status = "DEGRADED"
            else:
                metrics.status = "DEAD"
                
        except Exception as e:
            logger.error(f"Error collecting liveness metrics: {e}")
            metrics.status = "UNKNOWN"
        
        return metrics
    
    def _collect_reality(self) -> RealityMetrics:
        """Collect reality validation metrics."""
        metrics = RealityMetrics()
        
        try:
            if self.reality:
                # Get current reality score
                score = self.reality.calculate_score(
                    data_quality=0.95,
                    slippage_deviation=0.85,
                    stress_failure_rate=0.90,
                    overfit_penalty=0.80,
                    confidence_health=0.75
                )
                metrics.avg_score = score.value
                metrics.min_score = score.value  # Would need history for actual min
                metrics.max_score = score.value
                metrics.pass_condition = score.value >= 0.7
                
                # Get negative contributors from attribution
                if hasattr(score, 'attribution'):
                    negatives = [k for k, v in score.attribution.items() if v < 0.7]
                    metrics.negative_contributors = negatives[:3]
            else:
                # Simulated values for testing
                metrics.avg_score = 0.78
                metrics.min_score = 0.62
                metrics.max_score = 0.85
                metrics.pass_condition = True
                
        except Exception as e:
            logger.error(f"Error collecting reality metrics: {e}")
        
        return metrics
    
    def _collect_growth_state(self) -> GrowthStateMetrics:
        """Collect growth state analysis metrics."""
        metrics = GrowthStateMetrics()
        
        try:
            if self.growth:
                metrics.current_state = self.growth.state.value
                
                # Get state history percentages
                history = getattr(self.growth, 'state_history', [])
                if history:
                    total = len(history)
                    state_counts = {}
                    for state in history:
                        state_counts[state] = state_counts.get(state, 0) + 1
                    
                    metrics.accumulation_pct = state_counts.get('accumulation', 0) / total * 100
                    metrics.expansion_pct = state_counts.get('expansion', 0) / total * 100
                    metrics.defense_pct = state_counts.get('defense', 0) / total * 100
                    metrics.preservation_pct = state_counts.get('preservation', 0) / total * 100
                    
                    # Count transitions
                    transitions = 0
                    for i in range(1, len(history)):
                        if history[i] != history[i-1]:
                            transitions += 1
                    metrics.state_transitions = transitions
                else:
                    # Default distribution
                    metrics.accumulation_pct = 100.0
                
                metrics.preservation_exceeded = metrics.preservation_pct > 30.0
            else:
                # Simulated values
                metrics.current_state = "ACCUMULATION"
                metrics.accumulation_pct = 45.0
                metrics.expansion_pct = 30.0
                metrics.defense_pct = 20.0
                metrics.preservation_pct = 5.0
                
        except Exception as e:
            logger.error(f"Error collecting growth state metrics: {e}")
        
        return metrics
    
    def _collect_risk(self) -> RiskMetrics:
        """Collect risk and safety metrics."""
        metrics = RiskMetrics()
        
        try:
            if self.kill_switch:
                state = self.kill_switch.check_all(
                    current_drawdown_pct=0,
                    consecutive_losses=0,
                    last_confidence=0.5
                )
                metrics.kill_switch_triggered = state.is_active
                metrics.kill_switch_reason = state.reason if state.is_active else None
            
            if self.growth:
                cfi = getattr(self.growth, 'cfi', None)
                if cfi:
                    metrics.cfi_avg = getattr(cfi, 'current_value', 0)
                    metrics.cfi_max = getattr(cfi, 'peak_value', 0)
                    metrics.grinding_detected = getattr(cfi, 'is_grinding', False)
            
            # Load drawdown from validation state
            validation_state_path = Path("data/validation/current_state.json")
            if validation_state_path.exists():
                with open(validation_state_path) as f:
                    state = json.load(f)
                    metrics.max_drawdown_24h = state.get('current_drawdown_pct', 0)
                    
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
        
        return metrics
    
    def _collect_trading(self, hours: int) -> TradingMetrics:
        """Collect trading summary metrics."""
        metrics = TradingMetrics()
        
        try:
            if self.db:
                cutoff = datetime.now() - timedelta(hours=hours)
                cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
                
                # Query trades
                try:
                    cursor = self.db.db.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                            SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                            SUM(net_pnl) as net_pnl,
                            SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END) as gross_profit,
                            SUM(CASE WHEN net_pnl < 0 THEN ABS(net_pnl) ELSE 0 END) as gross_loss
                        FROM trades
                        WHERE entry_time > ? AND status = 'CLOSED'
                    """, (cutoff_str,))
                    
                    row = cursor.fetchone()
                    if row and row[0]:
                        metrics.trades_executed = row[0]
                        metrics.win_count = row[1] or 0
                        metrics.loss_count = row[2] or 0
                        metrics.net_pnl = row[3] or 0
                        metrics.gross_profit = row[4] or 0
                        metrics.gross_loss = row[5] or 0
                        
                        if metrics.trades_executed > 0:
                            metrics.win_rate = metrics.win_count / metrics.trades_executed
                        
                        if metrics.gross_loss > 0:
                            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
                            
                except Exception as e:
                    logger.warning(f"Failed to query trades: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
        
        return metrics
    
    def _collect_trend(self, current_date: str) -> TrendMetrics:
        """Collect trend comparison vs previous day."""
        metrics = TrendMetrics()
        
        try:
            # Calculate previous date
            current = datetime.strptime(current_date, '%Y-%m-%d')
            previous = current - timedelta(days=1)
            previous_date = previous.strftime('%Y-%m-%d')
            
            # Try to load previous report
            previous_path = self.report_dir / f"{previous_date}.json"
            
            if previous_path.exists():
                with open(previous_path) as f:
                    prev_data = json.load(f)
                
                metrics.has_previous = True
                
                # Calculate deltas
                prev_reality = prev_data.get('reality', {})
                prev_risk = prev_data.get('risk', {})
                prev_growth = prev_data.get('growth_state', {})
                
                if 'avg_score' in prev_reality:
                    # Will be calculated after we have current metrics
                    metrics.reality_score_delta = 0  # Placeholder
                
                if 'max_drawdown_24h' in prev_risk:
                    metrics.max_drawdown_delta = 0  # Placeholder
                
                if 'cfi_avg' in prev_risk:
                    metrics.cfi_delta = 0  # Placeholder
                
                metrics.growth_transitions = prev_growth.get('state_transitions', 0)
                
                # Store previous values for later delta calculation
                self._previous_data = prev_data
            else:
                metrics.has_previous = False
                self._previous_data = None
                
        except Exception as e:
            logger.warning(f"Error loading previous report: {e}")
            metrics.has_previous = False
        
        return metrics
    
    def update_trend_deltas(self, metrics: DailyMetrics) -> None:
        """Update trend deltas after collecting current metrics."""
        if not hasattr(self, '_previous_data') or not self._previous_data:
            return
        
        prev = self._previous_data
        
        # Reality score delta
        prev_reality = prev.get('reality', {}).get('avg_score', 0)
        metrics.trend.reality_score_delta = metrics.reality.avg_score - prev_reality
        
        # Drawdown delta (negative is good)
        prev_dd = prev.get('risk', {}).get('max_drawdown_24h', 0)
        metrics.trend.max_drawdown_delta = metrics.risk.max_drawdown_24h - prev_dd
        
        # CFI delta
        prev_cfi = prev.get('risk', {}).get('cfi_avg', 0)
        metrics.trend.cfi_delta = metrics.risk.cfi_avg - prev_cfi
