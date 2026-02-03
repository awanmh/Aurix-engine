"""
AURIX Daily Health Reporter

Generates automatic health reports every 24 hours for paper trading validation.
READ-ONLY: Does not modify any trading logic or system state.

Usage:
    # Standalone
    py -3.11 -m aurix.reporting.daily_reporter
    
    # With validation
    py -3.11 run_validation.py --model-version v1.0.0 --with-reporter
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ReporterConfig, ReporterThresholds
from .metrics_collector import MetricsCollector, DailyMetrics
from .verdict_engine import VerdictEngine, Verdict, VerdictResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class DailyHealthReporter:
    """
    Daily System Health Reporter for AURIX trading system.
    
    Features:
    - 24-hour rolling window metrics
    - 6 report sections + verdict
    - Trend comparison vs previous day
    - Console + JSON output
    - Critical event immediate reporting
    
    CRITICAL: This is READ-ONLY. Does not modify any system state.
    """
    
    def __init__(
        self,
        config: ReporterConfig = None,
        db=None,
        redis_bus=None,
        growth_orchestrator=None,
        reality_scorer=None,
        kill_switch=None
    ):
        self.config = config or ReporterConfig()
        self.report_dir = Path(self.config.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = MetricsCollector(
            db=db,
            redis_bus=redis_bus,
            growth_orchestrator=growth_orchestrator,
            reality_scorer=reality_scorer,
            kill_switch=kill_switch,
            report_dir=str(self.report_dir)
        )
        self.verdict_engine = VerdictEngine(self.config.thresholds)
        
        # Scheduler state
        self._running = False
        self._scheduler_thread = None
    
    def run(self) -> tuple[DailyMetrics, VerdictResult]:
        """
        Run single health report.
        
        Returns:
            Tuple of (metrics, verdict)
        """
        logger.info("Generating daily health report...")
        
        # Collect metrics
        metrics = self.collector.collect_all(hours=24)
        
        # Update trend deltas
        self.collector.update_trend_deltas(metrics)
        
        # Evaluate verdict
        verdict = self.verdict_engine.evaluate(metrics)
        
        # Generate console report
        report = self._format_console_report(metrics, verdict)
        print(report)
        
        # Save JSON
        self._save_json(metrics, verdict)
        
        return metrics, verdict
    
    def _format_console_report(self, metrics: DailyMetrics, verdict: VerdictResult) -> str:
        """Format report for console output."""
        lines = []
        
        # Header
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"       AURIX DAILY HEALTH REPORT - {metrics.date}")
        lines.append("=" * 60)
        lines.append("")
        
        # 1. System Liveness
        lines.append("1. SYSTEM LIVENESS")
        lines.append(f"   Candles Received:     {metrics.liveness.candles_received:,}")
        lines.append(f"   Candles Processed:    {metrics.liveness.candles_processed:,} ({metrics.liveness.process_rate:.1%})")
        redis_icon = "OK" if metrics.liveness.redis_alive else "FAIL"
        lines.append(f"   Redis Pipeline:       {redis_icon}")
        status_icon = {"ALIVE": "OK", "DEGRADED": "!!", "DEAD": "XX"}.get(metrics.liveness.status, "??")
        lines.append(f"   Status:               {status_icon} {metrics.liveness.status}")
        lines.append("")
        
        # 2. Reality Validation
        lines.append("2. REALITY VALIDATION")
        lines.append(f"   Avg Reality Score:    {metrics.reality.avg_score:.2f}")
        lines.append(f"   Min Reality Score:    {metrics.reality.min_score:.2f}")
        pass_icon = "PASS" if metrics.reality.pass_condition else "FAIL"
        lines.append(f"   Pass Condition:       {pass_icon} (avg >= 0.7)")
        if metrics.reality.negative_contributors:
            lines.append(f"   Negative Contributors: {', '.join(metrics.reality.negative_contributors)}")
        lines.append("")
        
        # 3. Growth State Analysis
        lines.append("3. GROWTH STATE ANALYSIS")
        lines.append(f"   Accumulation:         {metrics.growth_state.accumulation_pct:.1f}%")
        lines.append(f"   Expansion:            {metrics.growth_state.expansion_pct:.1f}%")
        lines.append(f"   Defense:              {metrics.growth_state.defense_pct:.1f}%")
        lines.append(f"   Preservation:         {metrics.growth_state.preservation_pct:.1f}%")
        lines.append(f"   Current State:        {metrics.growth_state.current_state}")
        pres_icon = "!!" if metrics.growth_state.preservation_exceeded else "OK"
        lines.append(f"   Flag:                 {pres_icon} {'Preservation > 30%' if metrics.growth_state.preservation_exceeded else 'OK'}")
        lines.append("")
        
        # 4. Risk & Safety
        lines.append("4. RISK & SAFETY")
        ks_icon = "YES" if metrics.risk.kill_switch_triggered else "NO"
        lines.append(f"   Kill Switch:          {ks_icon}")
        if metrics.risk.kill_switch_triggered:
            lines.append(f"   Kill Reason:          {metrics.risk.kill_switch_reason}")
        lines.append(f"   Max Drawdown (24h):   {metrics.risk.max_drawdown_24h:.1f}%")
        lines.append(f"   CFI (avg/max):        {metrics.risk.cfi_avg:.2f} / {metrics.risk.cfi_max:.2f}")
        grind_icon = "YES" if metrics.risk.grinding_detected else "NO"
        lines.append(f"   Grinding Phase:       {grind_icon}")
        lines.append("")
        
        # 5. Trading Summary
        lines.append("5. TRADING SUMMARY")
        lines.append(f"   Trades Executed:      {metrics.trading.trades_executed}")
        lines.append(f"   Win Rate:             {metrics.trading.win_rate:.1%}")
        pnl_sign = "+" if metrics.trading.net_pnl >= 0 else ""
        lines.append(f"   Net PnL:              {pnl_sign}${metrics.trading.net_pnl:.2f}")
        if metrics.trading.profit_factor > 0:
            lines.append(f"   Profit Factor:        {metrics.trading.profit_factor:.2f}")
        lines.append("")
        
        # 6. Trend Snapshot (vs Previous Day)
        lines.append("6. TREND (vs Previous Day)")
        if metrics.trend.has_previous:
            rs_arrow = self._trend_arrow(metrics.trend.reality_score_delta)
            dd_arrow = self._trend_arrow(-metrics.trend.max_drawdown_delta)  # Negative is good for DD
            cfi_arrow = self._trend_arrow(-metrics.trend.cfi_delta)  # Negative is good for CFI
            
            lines.append(f"   Reality Score:        {metrics.trend.reality_score_delta:+.2f} {rs_arrow}")
            lines.append(f"   Max Drawdown:         {metrics.trend.max_drawdown_delta:+.1f}% {dd_arrow}")
            lines.append(f"   CFI:                  {metrics.trend.cfi_delta:+.2f} {cfi_arrow}")
            lines.append(f"   Growth Transitions:   {metrics.trend.growth_transitions}")
        else:
            lines.append("   (No previous day data available)")
        lines.append("")
        
        # Verdict
        lines.append("=" * 60)
        lines.append(f"DAILY VERDICT: {verdict.emoji} {verdict.verdict.value}")
        lines.append(verdict.message)
        
        if verdict.reasons and verdict.reasons[0] != "No issues detected":
            lines.append("")
            lines.append("Issues:")
            for reason in verdict.reasons[:5]:
                lines.append(f"  - {reason}")
        
        if verdict.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in verdict.recommendations[:3]:
                lines.append(f"  - {rec}")
        
        lines.append("=" * 60)
        lines.append("")
        
        return "\n".join(lines)
    
    def _trend_arrow(self, delta: float) -> str:
        """Get trend arrow based on delta value."""
        if delta > 0.01:
            return "(+)"
        elif delta < -0.01:
            return "(-)"
        else:
            return "(=)"
    
    def _save_json(self, metrics: DailyMetrics, verdict: VerdictResult) -> None:
        """Save report as JSON file."""
        filename = f"{metrics.date}.json"
        filepath = self.report_dir / filename
        
        data = metrics.to_dict()
        data['verdict'] = {
            'level': verdict.verdict.value,
            'reasons': verdict.reasons,
            'recommendations': verdict.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def start_scheduler(self) -> None:
        """Start background scheduler for 24-hour reports."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        logger.info(f"Health reporter scheduler started (interval: {self.config.interval_hours}h)")
    
    def stop_scheduler(self) -> None:
        """Stop background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Health reporter scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Background loop for scheduled reports."""
        interval_seconds = self.config.interval_hours * 3600
        
        # Run first report immediately
        try:
            self.run()
        except Exception as e:
            logger.error(f"Error in scheduled report: {e}")
        
        while self._running:
            # Sleep in small increments to allow clean shutdown
            for _ in range(int(interval_seconds / 10)):
                if not self._running:
                    break
                time.sleep(10)
            
            if self._running:
                try:
                    self.run()
                except Exception as e:
                    logger.error(f"Error in scheduled report: {e}")
    
    def trigger_critical_report(self, event: str, details: str = "") -> None:
        """
        Trigger immediate report on critical event.
        
        Args:
            event: Critical event type (e.g., 'kill_switch_activated')
            details: Additional details
        """
        logger.warning(f"CRITICAL EVENT: {event} - {details}")
        logger.info("Generating immediate critical report...")
        
        try:
            metrics, verdict = self.run()
            
            # Also save with event suffix
            filename = f"{metrics.date}_CRITICAL_{event}.json"
            filepath = self.report_dir / filename
            
            data = metrics.to_dict()
            data['critical_event'] = {
                'type': event,
                'details': details,
                'timestamp': datetime.now().isoformat()
            }
            data['verdict'] = {
                'level': verdict.verdict.value,
                'reasons': verdict.reasons,
                'recommendations': verdict.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Critical report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error generating critical report: {e}")


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description='AURIX Daily Health Reporter')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as background scheduler')
    parser.add_argument('--report-dir', type=str, default='reports/daily',
                       help='Directory for report output')
    
    args = parser.parse_args()
    
    # Create config
    config = ReporterConfig(report_dir=args.report_dir)
    
    # Initialize reporter (without system components for standalone mode)
    reporter = DailyHealthReporter(config=config)
    
    if args.daemon:
        reporter.start_scheduler()
        print("Health reporter running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            reporter.stop_scheduler()
    else:
        # Single report
        reporter.run()


if __name__ == '__main__':
    main()
