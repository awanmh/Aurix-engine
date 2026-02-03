"""
AURIX Report Aggregator

Generates weekly and monthly summary reports from daily reports.
Provides prop-firm readiness assessment and progress tracking.

Usage:
    # Generate weekly report
    py -3.11 -m aurix.reporting.aggregator --weekly
    
    # Generate monthly report
    py -3.11 -m aurix.reporting.aggregator --monthly
    
    # Generate 30-day prop-firm assessment
    py -3.11 -m aurix.reporting.aggregator --prop-firm
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a period."""
    period_start: str
    period_end: str
    days_count: int
    
    # Trading
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    cumulative_pnl: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    profit_factor: float = 0.0
    
    # Risk
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    kill_switch_triggers: int = 0
    
    # Reality Score
    avg_reality_score: float = 0.0
    min_reality_score: float = 1.0
    reality_score_trend: str = "STABLE"
    
    # Growth State Distribution
    accumulation_pct: float = 0.0
    expansion_pct: float = 0.0
    defense_pct: float = 0.0
    preservation_pct: float = 0.0
    
    # System Health
    healthy_days: int = 0
    warning_days: int = 0
    critical_days: int = 0
    uptime_pct: float = 0.0
    
    # Verdict
    prop_firm_ready: bool = False
    verdict_reasons: List[str] = field(default_factory=list)


class ReportAggregator:
    """Aggregates daily reports into weekly/monthly summaries."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.daily_dir = self.reports_dir / "daily"
        self.weekly_dir = self.reports_dir / "weekly"
        self.monthly_dir = self.reports_dir / "monthly"
        
        # Create directories
        self.weekly_dir.mkdir(parents=True, exist_ok=True)
        self.monthly_dir.mkdir(parents=True, exist_ok=True)
    
    def load_daily_reports(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load all daily reports in date range."""
        reports = []
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            report_path = self.daily_dir / f"{date_str}.json"
            
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        data = json.load(f)
                        data['_date'] = date_str
                        reports.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {report_path}: {e}")
            
            current += timedelta(days=1)
        
        return reports
    
    def aggregate(self, reports: List[Dict], period_name: str) -> AggregatedMetrics:
        """Aggregate multiple daily reports."""
        if not reports:
            return AggregatedMetrics(
                period_start="N/A",
                period_end="N/A",
                days_count=0
            )
        
        # Period info
        dates = [r['_date'] for r in reports]
        metrics = AggregatedMetrics(
            period_start=min(dates),
            period_end=max(dates),
            days_count=len(reports)
        )
        
        # Aggregation
        total_reality_scores = []
        total_drawdowns = []
        total_gross_profit = 0.0
        total_gross_loss = 0.0
        pnl_by_day = []
        
        state_counts = {'accumulation': 0, 'expansion': 0, 'defense': 0, 'preservation': 0}
        
        for r in reports:
            trading = r.get('trading', {})
            risk = r.get('risk', {})
            reality = r.get('reality', {})
            growth = r.get('growth_state', {})
            verdict = r.get('verdict', {})
            
            # Trading
            metrics.total_trades += trading.get('trades_executed', 0)
            metrics.total_wins += trading.get('win_count', 0)
            metrics.total_losses += trading.get('loss_count', 0)
            day_pnl = trading.get('net_pnl', 0)
            metrics.cumulative_pnl += day_pnl
            pnl_by_day.append(day_pnl)
            total_gross_profit += trading.get('gross_profit', 0)
            total_gross_loss += trading.get('gross_loss', 0)
            
            # Risk
            dd = risk.get('max_drawdown_24h', 0)
            total_drawdowns.append(dd)
            if risk.get('kill_switch_triggered', False):
                metrics.kill_switch_triggers += 1
            
            # Reality
            rs = reality.get('avg_score', 0)
            if rs > 0:
                total_reality_scores.append(rs)
            
            # Growth State
            state = growth.get('current_state', 'accumulation').lower()
            if state in state_counts:
                state_counts[state] += 1
            
            # Verdict
            level = verdict.get('level', 'UNKNOWN')
            if level == 'HEALTHY':
                metrics.healthy_days += 1
            elif level == 'WARNING':
                metrics.warning_days += 1
            elif level == 'CRITICAL':
                metrics.critical_days += 1
        
        # Calculate derived metrics
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.total_wins / metrics.total_trades
        
        if total_gross_loss > 0:
            metrics.profit_factor = total_gross_profit / total_gross_loss
        
        if pnl_by_day:
            metrics.best_day_pnl = max(pnl_by_day)
            metrics.worst_day_pnl = min(pnl_by_day)
        
        if total_drawdowns:
            metrics.max_drawdown = max(total_drawdowns)
            metrics.avg_drawdown = sum(total_drawdowns) / len(total_drawdowns)
        
        if total_reality_scores:
            metrics.avg_reality_score = sum(total_reality_scores) / len(total_reality_scores)
            metrics.min_reality_score = min(total_reality_scores)
            
            # Trend detection
            if len(total_reality_scores) >= 3:
                first_half = sum(total_reality_scores[:len(total_reality_scores)//2]) / (len(total_reality_scores)//2)
                second_half = sum(total_reality_scores[len(total_reality_scores)//2:]) / (len(total_reality_scores) - len(total_reality_scores)//2)
                if second_half > first_half + 0.05:
                    metrics.reality_score_trend = "IMPROVING"
                elif second_half < first_half - 0.05:
                    metrics.reality_score_trend = "DECLINING"
        
        # Growth state distribution
        total_states = sum(state_counts.values())
        if total_states > 0:
            metrics.accumulation_pct = state_counts['accumulation'] / total_states * 100
            metrics.expansion_pct = state_counts['expansion'] / total_states * 100
            metrics.defense_pct = state_counts['defense'] / total_states * 100
            metrics.preservation_pct = state_counts['preservation'] / total_states * 100
        
        # Uptime
        if len(reports) > 0:
            metrics.uptime_pct = metrics.healthy_days / len(reports) * 100
        
        # Prop-firm readiness check
        metrics.prop_firm_ready, metrics.verdict_reasons = self._check_prop_firm_ready(metrics)
        
        return metrics
    
    def _check_prop_firm_ready(self, m: AggregatedMetrics) -> tuple[bool, List[str]]:
        """Check if metrics meet prop-firm requirements."""
        reasons = []
        passed = True
        
        # Minimum days
        if m.days_count < 14:
            reasons.append(f"Insufficient data: {m.days_count}/14 days minimum")
            passed = False
        
        # Win rate
        if m.win_rate < 0.45:
            reasons.append(f"Win rate too low: {m.win_rate:.1%} (min: 45%)")
            passed = False
        elif m.win_rate >= 0.50:
            reasons.append(f"Win rate OK: {m.win_rate:.1%}")
        
        # Max drawdown (FTMO: 10%, strict: 8%)
        if m.max_drawdown > 10:
            reasons.append(f"Drawdown exceeded: {m.max_drawdown:.1f}% (max: 10%)")
            passed = False
        elif m.max_drawdown <= 8:
            reasons.append(f"Drawdown within limit: {m.max_drawdown:.1f}%")
        
        # Profitability
        if m.cumulative_pnl <= 0:
            reasons.append(f"Not profitable: ${m.cumulative_pnl:.2f}")
            passed = False
        else:
            reasons.append(f"Profitable: +${m.cumulative_pnl:.2f}")
        
        # Reality Score
        if m.avg_reality_score < 0.7:
            reasons.append(f"Reality Score low: {m.avg_reality_score:.2f} (min: 0.7)")
            passed = False
        else:
            reasons.append(f"Reality Score healthy: {m.avg_reality_score:.2f}")
        
        # Profit Factor
        if m.profit_factor < 1.2:
            reasons.append(f"Profit Factor low: {m.profit_factor:.2f} (min: 1.2)")
            passed = False
        elif m.profit_factor >= 1.5:
            reasons.append(f"Profit Factor excellent: {m.profit_factor:.2f}")
        
        # Critical days
        if m.critical_days > 0:
            reasons.append(f"Critical days detected: {m.critical_days}")
            passed = False
        
        # Kill switch
        if m.kill_switch_triggers > 2:
            reasons.append(f"Too many kill switch triggers: {m.kill_switch_triggers}")
            passed = False
        
        # Minimum trades
        if m.total_trades < 30:
            reasons.append(f"Insufficient trades: {m.total_trades}/30 minimum")
            passed = False
        
        return passed, reasons
    
    def generate_weekly(self, week_date: datetime = None) -> str:
        """Generate weekly report."""
        if week_date is None:
            week_date = datetime.now()
        
        # Get ISO week
        year, week_num, _ = week_date.isocalendar()
        week_start = week_date - timedelta(days=week_date.weekday())
        week_end = week_start + timedelta(days=6)
        
        reports = self.load_daily_reports(week_start, week_end)
        metrics = self.aggregate(reports, f"{year}-W{week_num:02d}")
        
        # Save report
        filename = f"{year}-W{week_num:02d}.json"
        filepath = self.weekly_dir / filename
        self._save_report(metrics, filepath, "WEEKLY")
        
        # Print console report
        return self._format_period_report(metrics, f"WEEKLY REPORT (Week {week_num}, {year})")
    
    def generate_monthly(self, month_date: datetime = None) -> str:
        """Generate monthly report."""
        if month_date is None:
            month_date = datetime.now()
        
        year = month_date.year
        month = month_date.month
        
        # Get month range
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
        
        reports = self.load_daily_reports(month_start, month_end)
        metrics = self.aggregate(reports, f"{year}-{month:02d}")
        
        # Save report
        filename = f"{year}-{month:02d}.json"
        filepath = self.monthly_dir / filename
        self._save_report(metrics, filepath, "MONTHLY")
        
        return self._format_period_report(metrics, f"MONTHLY REPORT ({month_date.strftime('%B %Y')})")
    
    def generate_prop_firm_assessment(self, days: int = 30) -> str:
        """Generate 30-day prop-firm readiness assessment."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        reports = self.load_daily_reports(start_date, end_date)
        metrics = self.aggregate(reports, f"{days}-DAY-ASSESSMENT")
        
        # Save report
        filename = f"prop_firm_assessment_{end_date.strftime('%Y-%m-%d')}.json"
        filepath = self.reports_dir / filename
        self._save_report(metrics, filepath, "PROP_FIRM_ASSESSMENT")
        
        return self._format_prop_firm_report(metrics, days)
    
    def _save_report(self, metrics: AggregatedMetrics, filepath: Path, report_type: str):
        """Save aggregated report as JSON."""
        data = {
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': metrics.period_start,
                'end': metrics.period_end,
                'days': metrics.days_count
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'wins': metrics.total_wins,
                'losses': metrics.total_losses,
                'win_rate': metrics.win_rate,
                'cumulative_pnl': metrics.cumulative_pnl,
                'best_day_pnl': metrics.best_day_pnl,
                'worst_day_pnl': metrics.worst_day_pnl,
                'profit_factor': metrics.profit_factor
            },
            'risk': {
                'max_drawdown': metrics.max_drawdown,
                'avg_drawdown': metrics.avg_drawdown,
                'kill_switch_triggers': metrics.kill_switch_triggers
            },
            'reality_score': {
                'average': metrics.avg_reality_score,
                'minimum': metrics.min_reality_score,
                'trend': metrics.reality_score_trend
            },
            'growth_state_distribution': {
                'accumulation': metrics.accumulation_pct,
                'expansion': metrics.expansion_pct,
                'defense': metrics.defense_pct,
                'preservation': metrics.preservation_pct
            },
            'health': {
                'healthy_days': metrics.healthy_days,
                'warning_days': metrics.warning_days,
                'critical_days': metrics.critical_days,
                'uptime_pct': metrics.uptime_pct
            },
            'prop_firm_verdict': {
                'ready': metrics.prop_firm_ready,
                'reasons': metrics.verdict_reasons
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def _format_period_report(self, m: AggregatedMetrics, title: str) -> str:
        """Format period report for console."""
        lines = []
        lines.append("")
        lines.append("=" * 65)
        lines.append(f"  AURIX {title}")
        lines.append(f"  Period: {m.period_start} to {m.period_end} ({m.days_count} days)")
        lines.append("=" * 65)
        lines.append("")
        
        # Trading Summary
        lines.append("TRADING SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Trades:     {m.total_trades}")
        lines.append(f"  Wins/Losses:      {m.total_wins} / {m.total_losses}")
        lines.append(f"  Win Rate:         {m.win_rate:.1%}")
        pnl_sign = "+" if m.cumulative_pnl >= 0 else ""
        lines.append(f"  Cumulative PnL:   {pnl_sign}${m.cumulative_pnl:.2f}")
        lines.append(f"  Best Day:         +${m.best_day_pnl:.2f}")
        lines.append(f"  Worst Day:        ${m.worst_day_pnl:.2f}")
        lines.append(f"  Profit Factor:    {m.profit_factor:.2f}")
        lines.append("")
        
        # Risk Metrics
        lines.append("RISK METRICS")
        lines.append("-" * 40)
        lines.append(f"  Max Drawdown:     {m.max_drawdown:.1f}%")
        lines.append(f"  Avg Drawdown:     {m.avg_drawdown:.1f}%")
        lines.append(f"  Kill Switch:      {m.kill_switch_triggers}x triggered")
        lines.append("")
        
        # Reality Score
        lines.append("REALITY SCORE")
        lines.append("-" * 40)
        lines.append(f"  Average:          {m.avg_reality_score:.2f}")
        lines.append(f"  Minimum:          {m.min_reality_score:.2f}")
        lines.append(f"  Trend:            {m.reality_score_trend}")
        lines.append("")
        
        # Growth State
        lines.append("GROWTH STATE DISTRIBUTION")
        lines.append("-" * 40)
        lines.append(f"  Accumulation:     {m.accumulation_pct:.0f}%")
        lines.append(f"  Expansion:        {m.expansion_pct:.0f}%")
        lines.append(f"  Defense:          {m.defense_pct:.0f}%")
        lines.append(f"  Preservation:     {m.preservation_pct:.0f}%")
        lines.append("")
        
        # System Health
        lines.append("SYSTEM HEALTH")
        lines.append("-" * 40)
        lines.append(f"  Healthy Days:     {m.healthy_days}")
        lines.append(f"  Warning Days:     {m.warning_days}")
        lines.append(f"  Critical Days:    {m.critical_days}")
        lines.append(f"  Uptime:           {m.uptime_pct:.0f}%")
        lines.append("")
        
        lines.append("=" * 65)
        return "\n".join(lines)
    
    def _format_prop_firm_report(self, m: AggregatedMetrics, days: int) -> str:
        """Format prop-firm assessment report."""
        lines = []
        lines.append("")
        lines.append("=" * 65)
        lines.append(f"  AURIX PROP-FIRM READINESS ASSESSMENT")
        lines.append(f"  {days}-Day Analysis: {m.period_start} to {m.period_end}")
        lines.append("=" * 65)
        lines.append("")
        
        # Key Metrics Grid
        lines.append("KEY METRICS vs PROP-FIRM REQUIREMENTS")
        lines.append("-" * 65)
        lines.append(f"{'Metric':<25} {'Actual':<15} {'Required':<15} {'Status':<10}")
        lines.append("-" * 65)
        
        metrics_check = [
            ("Days Tracked", f"{m.days_count}", "14+", "PASS" if m.days_count >= 14 else "FAIL"),
            ("Win Rate", f"{m.win_rate:.1%}", ">45%", "PASS" if m.win_rate >= 0.45 else "FAIL"),
            ("Max Drawdown", f"{m.max_drawdown:.1f}%", "<10%", "PASS" if m.max_drawdown < 10 else "FAIL"),
            ("Profit Factor", f"{m.profit_factor:.2f}", ">1.2", "PASS" if m.profit_factor >= 1.2 else "FAIL"),
            ("Cumulative PnL", f"${m.cumulative_pnl:.2f}", ">$0", "PASS" if m.cumulative_pnl > 0 else "FAIL"),
            ("Reality Score", f"{m.avg_reality_score:.2f}", ">0.70", "PASS" if m.avg_reality_score >= 0.7 else "FAIL"),
            ("Critical Days", f"{m.critical_days}", "0", "PASS" if m.critical_days == 0 else "FAIL"),
            ("Trade Count", f"{m.total_trades}", "30+", "PASS" if m.total_trades >= 30 else "FAIL"),
        ]
        
        for metric, actual, required, status in metrics_check:
            status_icon = "OK" if status == "PASS" else "XX"
            lines.append(f"{metric:<25} {actual:<15} {required:<15} {status_icon}")
        
        lines.append("-" * 65)
        lines.append("")
        
        # Verdict
        if m.prop_firm_ready:
            verdict = "PROP-FIRM READY"
            verdict_icon = "OK"
        else:
            verdict = "NOT READY"
            verdict_icon = "XX"
        
        lines.append("=" * 65)
        lines.append(f"  FINAL VERDICT: {verdict_icon} {verdict}")
        lines.append("=" * 65)
        lines.append("")
        
        lines.append("Assessment Details:")
        for reason in m.verdict_reasons:
            icon = "(+)" if "OK" in reason or "OK" in reason or "Profitable" in reason or "excellent" in reason or "healthy" in reason else "(-):" if "low" in reason or "exceeded" in reason or "Insufficient" in reason or "Too many" in reason or "Not profitable" in reason else "   "
            lines.append(f"  {icon} {reason}")
        
        lines.append("")
        
        # Recommendations
        if not m.prop_firm_ready:
            lines.append("RECOMMENDATIONS:")
            if m.days_count < 14:
                lines.append(f"  - Continue running for {14 - m.days_count} more days")
            if m.win_rate < 0.45:
                lines.append("  - Review trade selection criteria")
            if m.max_drawdown > 10:
                lines.append("  - Reduce position sizes or tighten stop losses")
            if m.total_trades < 30:
                lines.append("  - Need more trade samples for statistical significance")
        else:
            lines.append("NEXT STEPS:")
            lines.append("  1. Begin Phase 2: Micro real capital ($100-500)")
            lines.append("  2. Run for 14+ more days with real money")
            lines.append("  3. Verify slippage and execution quality")
            lines.append("  4. Apply for prop-firm challenge")
        
        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AURIX Report Aggregator')
    parser.add_argument('--weekly', action='store_true', help='Generate weekly report')
    parser.add_argument('--monthly', action='store_true', help='Generate monthly report')
    parser.add_argument('--prop-firm', action='store_true', help='Generate 30-day prop-firm assessment')
    parser.add_argument('--days', type=int, default=30, help='Days for prop-firm assessment')
    parser.add_argument('--reports-dir', type=str, default='reports', help='Reports directory')
    
    args = parser.parse_args()
    
    aggregator = ReportAggregator(args.reports_dir)
    
    if args.weekly:
        report = aggregator.generate_weekly()
        print(report)
    
    if args.monthly:
        report = aggregator.generate_monthly()
        print(report)
    
    if args.prop_firm:
        report = aggregator.generate_prop_firm_assessment(args.days)
        print(report)
    
    if not (args.weekly or args.monthly or args.prop_firm):
        # Default: show prop-firm assessment
        report = aggregator.generate_prop_firm_assessment(30)
        print(report)


if __name__ == '__main__':
    main()
