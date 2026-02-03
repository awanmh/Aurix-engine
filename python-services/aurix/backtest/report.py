"""
AURIX Expectancy & Risk Report Generator

Generates comprehensive statistical performance reports from backtest results.
Includes:
- Expectancy per trade
- Win/loss ratios
- Loss tail analysis
- Confidence bucket accuracy
- Regime-specific recommendations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from .engine import BacktestMetrics, Trade


@dataclass
class RiskRecommendation:
    """Trading recommendation based on backtest analysis."""
    min_confidence_threshold: float
    regime_rules: Dict[str, str]  # regime -> "enabled" / "disabled" / "reduced"
    max_position_size_pct: float
    suggested_tp_pct: float
    suggested_sl_pct: float
    risk_score: int  # 1-10, higher = riskier
    summary: str


class ReportGenerator:
    """
    Generates comprehensive backtest analysis reports.
    
    Analyzes:
    1. Expectancy and edge quality
    2. Risk/reward characteristics
    3. Tail risk (worst outcomes)
    4. Confidence calibration
    5. Regime-specific behavior
    """
    
    def __init__(self, metrics: BacktestMetrics, trades: List[Trade]):
        """
        Initialize report generator.
        
        Args:
            metrics: Computed backtest metrics
            trades: List of executed trades
        """
        self.metrics = metrics
        self.trades = trades
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive report as dictionary."""
        return {
            'summary': self._summary_section(),
            'expectancy_analysis': self._expectancy_analysis(),
            'risk_analysis': self._risk_analysis(),
            'tail_analysis': self._tail_analysis(),
            'confidence_analysis': self._confidence_analysis(),
            'regime_analysis': self._regime_analysis(),
            'recommendations': self._generate_recommendations().__dict__,
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_markdown_report(self) -> str:
        """Generate report as formatted markdown."""
        report = self.generate_full_report()
        rec = self._generate_recommendations()
        
        md = []
        md.append("# AURIX Backtest Performance Report")
        md.append(f"\n*Generated: {report['generated_at']}*\n")
        
        # Summary
        md.append("## Executive Summary")
        summary = report['summary']
        md.append(f"""
| Metric | Value |
|--------|-------|
| Total Trades | {summary['total_trades']} |
| Win Rate | {summary['win_rate']:.1%} |
| Profit Factor | {summary['profit_factor']:.2f} |
| Total Return | {summary['total_return_pct']:.1%} |
| Max Drawdown | {summary['max_drawdown_pct']:.1%} |
| Sharpe Ratio | {summary['sharpe_ratio']:.2f} |
""")
        
        # Expectancy
        md.append("\n## Expectancy Analysis")
        exp = report['expectancy_analysis']
        md.append(f"""
| Metric | Value |
|--------|-------|
| Expectancy per Trade | ${exp['expectancy_per_trade']:.2f} |
| Expectancy Ratio | {exp['expectancy_ratio']:.2f} |
| Average Win | ${exp['avg_win']:.2f} |
| Average Loss | ${exp['avg_loss']:.2f} |
| Win/Loss Ratio | {exp['win_loss_ratio']:.2f} |
| Edge Quality | {exp['edge_quality']} |
""")
        
        if exp['expectancy_per_trade'] > 0:
            md.append("\n> [+] **Positive expectancy detected.** The system has a statistical edge.")
        else:
            md.append("\n> [!] **Negative expectancy.** The system is losing money on average.")
        
        # Tail Analysis
        md.append("\n## Loss Tail Analysis (Worst 5%)")
        tail = report['tail_analysis']
        md.append(f"""
| Metric | Value |
|--------|-------|
| Worst 5% Avg Loss | ${tail['worst_5pct_avg']:.2f} |
| Worst Single Trade | ${tail['worst_trade']:.2f} |
| Max Consecutive Losses | {tail['max_consecutive_losses']} |
| Tail Risk Score | {tail['tail_risk_score']}/10 |
""")
        
        # Confidence Buckets
        md.append("\n## Confidence Bucket Accuracy")
        md.append("\n| Bucket | Trades | Win Rate | Recommendation |")
        md.append("|--------|--------|----------|----------------|")
        
        conf = report['confidence_analysis']
        for bucket, data in conf['buckets'].items():
            if data['count'] > 0:
                status = "[+] TRADE" if data['win_rate'] >= 0.52 else "[-] SKIP"
                md.append(f"| {bucket} | {data['count']} | {data['win_rate']:.1%} | {status} |")
        
        # Regime Analysis
        md.append("\n## Regime Performance")
        md.append("\n| Regime | Trades | Win Rate | Avg PnL | Recommendation |")
        md.append("|--------|--------|----------|---------|----------------|")
        
        regime = report['regime_analysis']
        for reg_name, data in regime.items():
            if data['count'] > 0:
                if data['win_rate'] >= 0.55:
                    status = "[+] FULL"
                elif data['win_rate'] >= 0.48:
                    status = "[!] REDUCED"
                else:
                    status = "[-] DISABLE"
                md.append(f"| {reg_name} | {data['count']} | {data['win_rate']:.1%} | ${data['avg_pnl']:.2f} | {status} |")
        
        # Recommendations
        md.append("\n## Trading Recommendations")
        md.append(f"""
### Confidence Threshold
**Minimum: {rec.min_confidence_threshold:.1%}**

Based on the confidence bucket analysis, trades below this threshold 
showed negative or marginal expectancy.

### Regime Rules
""")
        for regime, rule in rec.regime_rules.items():
            emoji = "[+]" if rule == "enabled" else ("[!]" if rule == "reduced" else "[-]")
            md.append(f"- **{regime}**: {emoji} {rule.upper()}")
        
        md.append(f"""
### Position Sizing
- Max Position Size: **{rec.max_position_size_pct:.1%}** of equity
- Suggested TP: **{rec.suggested_tp_pct:.1%}**
- Suggested SL: **{rec.suggested_sl_pct:.1%}**

### Overall Risk Score: {rec.risk_score}/10

{rec.summary}
""")
        
        return "\n".join(md)
    
    def _summary_section(self) -> Dict:
        """Generate summary section."""
        return {
            'total_trades': self.metrics.total_trades,
            'winning_trades': self.metrics.winning_trades,
            'losing_trades': self.metrics.losing_trades,
            'win_rate': self.metrics.win_rate,
            'profit_factor': self.metrics.profit_factor,
            'total_return': self.metrics.total_return,
            'total_return_pct': self.metrics.total_return_pct,
            'max_drawdown': self.metrics.max_drawdown,
            'max_drawdown_pct': self.metrics.max_drawdown_pct,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'sortino_ratio': self.metrics.sortino_ratio
        }
    
    def _expectancy_analysis(self) -> Dict:
        """Analyze expectancy characteristics."""
        avg_win = self.metrics.avg_win
        avg_loss = self.metrics.avg_loss
        win_rate = self.metrics.win_rate
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Expectancy formula: (Win% × Avg Win) - (Loss% × Avg Loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Expectancy ratio (normalized by avg loss)
        exp_ratio = expectancy / avg_loss if avg_loss > 0 else 0
        
        # Edge quality assessment
        if exp_ratio > 0.3:
            edge_quality = "EXCELLENT"
        elif exp_ratio > 0.15:
            edge_quality = "GOOD"
        elif exp_ratio > 0.05:
            edge_quality = "MARGINAL"
        elif exp_ratio > 0:
            edge_quality = "WEAK"
        else:
            edge_quality = "NEGATIVE"
        
        # Break-even win rate needed
        breakeven_wr = 1 / (1 + win_loss_ratio) if win_loss_ratio > 0 else 0.5
        
        return {
            'expectancy_per_trade': expectancy,
            'expectancy_ratio': exp_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'win_rate': win_rate,
            'breakeven_win_rate': breakeven_wr,
            'edge_quality': edge_quality,
            'profit_factor': self.metrics.profit_factor
        }
    
    def _risk_analysis(self) -> Dict:
        """Analyze risk characteristics."""
        pnl_values = [t.net_pnl for t in self.trades if t.net_pnl is not None]
        
        if not pnl_values:
            return {
                'pnl_mean': 0.0,
                'pnl_std': 0.0,
                'coefficient_of_variation': 0.0,
                'var_95': 0.0,
                'expected_shortfall': 0.0,
                'max_drawdown_pct': self.metrics.max_drawdown_pct,
                'avg_drawdown': self.metrics.avg_drawdown
            }
        
        # Standard deviation of returns
        pnl_std = np.std(pnl_values)
        pnl_mean = np.mean(pnl_values)
        
        # Coefficient of variation
        cv = pnl_std / abs(pnl_mean) if pnl_mean != 0 else float('inf')
        
        # Value at Risk (simplified - 95th percentile loss)
        var_95 = np.percentile(pnl_values, 5)  # 5th percentile = worst 5%
        
        # Expected shortfall (average of worst 5%)
        worst_5pct_threshold = np.percentile(pnl_values, 5)
        worst_trades = [p for p in pnl_values if p <= worst_5pct_threshold]
        expected_shortfall = np.mean(worst_trades) if worst_trades else 0
        
        return {
            'pnl_mean': pnl_mean,
            'pnl_std': pnl_std,
            'coefficient_of_variation': cv,
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'max_drawdown_pct': self.metrics.max_drawdown_pct,
            'avg_drawdown': self.metrics.avg_drawdown
        }
    
    def _tail_analysis(self) -> Dict:
        """Analyze the worst outcomes (tail risk)."""
        pnl_values = sorted([t.net_pnl for t in self.trades if t.net_pnl is not None])
        
        if not pnl_values:
            return {
                'worst_5pct_count': 0,
                'worst_5pct_avg': 0.0,
                'worst_5pct_trades': [],
                'worst_trade': 0.0,
                'worst_trade_details': {'pnl': 0, 'direction': 'N/A', 'regime': 'N/A', 'confidence': 0, 'duration_minutes': 0},
                'max_consecutive_losses': 0,
                'tail_risk_score': 0
            }
        
        # Worst 5% of trades
        n_worst = max(1, int(len(pnl_values) * 0.05))
        worst_trades = pnl_values[:n_worst]
        
        # Worst trade details
        worst_trade = min(self.trades, key=lambda t: t.net_pnl if t.net_pnl else 0)
        
        # Analyze worst trade characteristics
        worst_details = {
            'pnl': worst_trade.net_pnl,
            'direction': worst_trade.direction,
            'regime': worst_trade.regime,
            'confidence': worst_trade.confidence,
            'duration_minutes': (worst_trade.exit_time - worst_trade.entry_time).total_seconds() / 60 if worst_trade.exit_time else 0
        }
        
        # Tail risk score (1-10)
        # Based on how extreme the tail losses are relative to average
        avg_loss = abs(self.metrics.avg_loss)
        worst_avg = abs(np.mean(worst_trades)) if worst_trades else 0
        
        if avg_loss > 0:
            tail_multiplier = worst_avg / avg_loss
            if tail_multiplier < 1.5:
                tail_risk_score = 3
            elif tail_multiplier < 2.0:
                tail_risk_score = 5
            elif tail_multiplier < 3.0:
                tail_risk_score = 7
            else:
                tail_risk_score = 9
        else:
            tail_risk_score = 5
        
        return {
            'worst_5pct_count': n_worst,
            'worst_5pct_avg': np.mean(worst_trades) if worst_trades else 0,
            'worst_5pct_trades': worst_trades,
            'worst_trade': worst_trade.net_pnl,
            'worst_trade_details': worst_details,
            'max_consecutive_losses': self.metrics.max_consecutive_losses,
            'tail_risk_score': tail_risk_score
        }
    
    def _confidence_analysis(self) -> Dict:
        """Analyze accuracy by confidence bucket."""
        buckets = {
            '0.55-0.60': {'trades': [], 'wins': 0, 'count': 0},
            '0.60-0.65': {'trades': [], 'wins': 0, 'count': 0},
            '0.65-0.70': {'trades': [], 'wins': 0, 'count': 0},
            '0.70-0.75': {'trades': [], 'wins': 0, 'count': 0},
            '0.75-0.80': {'trades': [], 'wins': 0, 'count': 0},
            '0.80-0.85': {'trades': [], 'wins': 0, 'count': 0},
            '0.85-1.00': {'trades': [], 'wins': 0, 'count': 0}
        }
        
        bucket_ranges = [
            (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75),
            (0.75, 0.80), (0.80, 0.85), (0.85, 1.00)
        ]
        
        for trade in self.trades:
            for (low, high) in bucket_ranges:
                if low <= trade.confidence < high:
                    key = f"{low:.2f}-{high:.2f}"
                    buckets[key]['trades'].append(trade)
                    buckets[key]['count'] += 1
                    if trade.outcome == "WIN":
                        buckets[key]['wins'] += 1
                    break
        
        # Calculate win rates
        result = {}
        for key, data in buckets.items():
            win_rate = data['wins'] / data['count'] if data['count'] > 0 else 0
            avg_pnl = np.mean([t.net_pnl for t in data['trades']]) if data['trades'] else 0
            
            result[key] = {
                'count': data['count'],
                'wins': data['wins'],
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'is_calibrated': abs(win_rate - float(key.split('-')[0])) < 0.10 if data['count'] >= 10 else None
            }
        
        # Find optimal threshold
        optimal_threshold = 0.60
        for key in sorted(buckets.keys()):
            data = result[key]
            if data['count'] >= 10 and data['win_rate'] >= 0.52:
                optimal_threshold = float(key.split('-')[0])
                break
        
        return {
            'buckets': result,
            'optimal_threshold': optimal_threshold,
            'calibration_quality': self._assess_calibration(result)
        }
    
    def _assess_calibration(self, buckets: Dict) -> str:
        """Assess probability calibration quality."""
        calibrated_count = 0
        total_with_data = 0
        
        for key, data in buckets.items():
            if data['count'] >= 10:
                total_with_data += 1
                if data['is_calibrated']:
                    calibrated_count += 1
        
        if total_with_data == 0:
            return "INSUFFICIENT_DATA"
        
        ratio = calibrated_count / total_with_data
        if ratio >= 0.8:
            return "WELL_CALIBRATED"
        elif ratio >= 0.5:
            return "PARTIALLY_CALIBRATED"
        else:
            return "POORLY_CALIBRATED"
    
    def _regime_analysis(self) -> Dict:
        """Analyze performance by market regime."""
        regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "UNKNOWN"]
        result = {}
        
        for regime in regimes:
            regime_trades = [t for t in self.trades if t.regime == regime]
            
            if not regime_trades:
                result[regime] = {
                    'count': 0,
                    'wins': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'recommendation': 'no_data'
                }
                continue
            
            wins = len([t for t in regime_trades if t.outcome == "WIN"])
            win_rate = wins / len(regime_trades)
            avg_pnl = np.mean([t.net_pnl for t in regime_trades])
            total_pnl = sum([t.net_pnl for t in regime_trades])
            
            # Generate recommendation
            if win_rate >= 0.55 and avg_pnl > 0:
                rec = "enabled"
            elif win_rate >= 0.48:
                rec = "reduced"
            else:
                rec = "disabled"
            
            result[regime] = {
                'count': len(regime_trades),
                'wins': wins,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'recommendation': rec
            }
        
        return result
    
    def _generate_recommendations(self) -> RiskRecommendation:
        """Generate actionable trading recommendations."""
        conf_analysis = self._confidence_analysis()
        regime_analysis = self._regime_analysis()
        tail_analysis = self._tail_analysis()
        
        # Minimum confidence threshold
        min_threshold = conf_analysis['optimal_threshold']
        
        # Regime rules
        regime_rules = {}
        for regime, data in regime_analysis.items():
            regime_rules[regime] = data.get('recommendation', 'disabled')
        
        # Position sizing (adjust based on tail risk)
        tail_score = tail_analysis.get('tail_risk_score', 5)
        if tail_score <= 3:
            max_pos = 5.0
        elif tail_score <= 5:
            max_pos = 3.0
        elif tail_score <= 7:
            max_pos = 2.0
        else:
            max_pos = 1.0
        
        # TP/SL suggestions based on win/loss ratio
        exp_analysis = self._expectancy_analysis()
        win_loss_ratio = exp_analysis['win_loss_ratio']
        
        if win_loss_ratio >= 2.0:
            suggested_tp = 1.5
            suggested_sl = 0.75
        elif win_loss_ratio >= 1.5:
            suggested_tp = 1.0
            suggested_sl = 0.5
        else:
            suggested_tp = 0.8
            suggested_sl = 0.4
        
        # Overall risk score
        risk_factors = [
            tail_score,
            10 if self.metrics.max_drawdown_pct > 0.1 else 5 if self.metrics.max_drawdown_pct > 0.05 else 2,
            8 if self.metrics.win_rate < 0.48 else 4 if self.metrics.win_rate < 0.52 else 2,
            7 if self.metrics.max_consecutive_losses > 5 else 3
        ]
        risk_score = int(np.mean(risk_factors))
        
        # Summary
        if self.metrics.expectancy > 0 and self.metrics.profit_factor > 1.0:
            if risk_score <= 4:
                summary = "[+] **READY FOR PAPER TRADING** - Positive expectancy with acceptable risk profile."
            else:
                summary = "[!] **PROCEED WITH CAUTION** - Positive expectancy but elevated risk. Use reduced position sizes."
        else:
            summary = "[-] **NOT RECOMMENDED** - Negative expectancy or profit factor < 1.0. Further optimization needed."
        
        return RiskRecommendation(
            min_confidence_threshold=min_threshold,
            regime_rules=regime_rules,
            max_position_size_pct=max_pos,
            suggested_tp_pct=suggested_tp,
            suggested_sl_pct=suggested_sl,
            risk_score=risk_score,
            summary=summary
        )
    
    def save_report(self, filepath: str, format: str = 'json'):
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: 'json' or 'markdown'
        """
        if format == 'json':
            report = self.generate_full_report()
            # Convert non-serializable types
            report['recommendations']['regime_rules'] = dict(report['recommendations']['regime_rules'])
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'markdown':
            md = self.generate_markdown_report()
            with open(filepath, 'w') as f:
                f.write(md)
        else:
            raise ValueError(f"Unknown format: {format}")
