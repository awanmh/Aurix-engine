# AURIX - Adaptive Crypto Trading Bot System

A production-ready, ML-driven cryptocurrency trading bot with continuous learning, reality validation, and adaptive capital management.

## 🎯 Overview

AURIX is designed for automated BTC/USDT futures trading on Binance, using machine learning predictions with:

- Dynamic confidence thresholds
- Multi-layer risk management
- Reality validation & anti-overfitting
- Adaptive capital growth orchestration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AURIX System                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────┐   ┌─────────────────┐   ┌────────────┐             │
│  │ Go Collector│   │   Python ML     │   │ Go Executor│             │
│  │ (WebSocket) │──▶│ (Decision Eng)  │──▶│  (Orders)  │             │
│  └────────────┘   └─────────────────┘   └────────────┘             │
│                           │                                         │
│           ┌───────────────┼───────────────┐                         │
│           ▼               ▼               ▼                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Reality     │  │ Capital     │  │ Capital     │                 │
│  │ Layer       │  │ Efficiency  │  │ Growth      │                 │
│  │ (9 modules) │  │ Gate        │  │ Orchestrator│                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## �️ Multi-Layer Protection

### Layer 1: Reality Validation (`aurix/reality/`)

| Module                 | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| `reality_score.py`     | Unified 0-1 health score with attribution         |
| `recovery_protocol.py` | 3-phase recovery (cooldown → validation → rampup) |
| `kill_switch.py`       | Hard stop on drawdown/losses                      |
| `overfit_monitor.py`   | Train/forward divergence detection                |
| `stress_tester.py`     | Noise & delay injection                           |
| `slippage_model.py`    | Volatility-based execution costs                  |
| `data_guard.py`        | Lookahead prevention                              |

### Layer 2: Capital Efficiency (`aurix/capital/`)

| Module                   | Purpose                                                       |
| ------------------------ | ------------------------------------------------------------- |
| `growth_orchestrator.py` | 4-state machine (Accumulation/Expansion/Defense/Preservation) |
| `gate.py`                | Trade approval gate                                           |
| `scorer.py`              | Capital efficiency scoring                                    |
| `pair_manager.py`        | Multi-pair rotation                                           |
| `psych_drift.py`         | Psychological drift proxy                                     |

### Layer 3: ML Pipeline (`aurix/ml/`, `aurix/features/`, `aurix/regime/`)

- 67 technical features
- LightGBM/XGBoost ensemble
- Regime-aware predictions
- Probability calibration

## 📊 Capital Growth State Machine

| State            | Risk/Trade | Aggression | Exposure Cap |
| ---------------- | ---------- | ---------- | ------------ |
| **Accumulation** | 1.0%       | 0.8x       | 30%          |
| **Expansion**    | 1.5%       | 1.2x       | 50%          |
| **Defense**      | 0.5%       | 0.5x       | 20%          |
| **Preservation** | 0.25%      | 0.3x       | 10%          |

Transitions based on: Equity slope, Drawdown velocity, Reality Score, Capital Fatigue Index

## 📁 Project Structure

```
Aurix/
├── config/config.example.yaml
├── go-services/
│   └── cmd/{collector,executor}/
├── python-services/
│   ├── decision_engine.py      # Main orchestrator
│   ├── run_backtest.py         # Walk-forward backtest
│   ├── run_validation.py       # Paper trading validation
│   └── aurix/
│       ├── reality/            # 9 modules
│       ├── capital/            # 6 modules
│       ├── reporting/          # Daily health reporter
│       ├── ml/                 # Training & prediction
│       ├── features/           # 67 indicators
│       ├── regime/             # Market regime
│       ├── backtest/           # Walk-forward engine
│       └── validation/         # Capital validation
├── reports/daily/              # Daily health reports (JSON)
└── docker-compose.yml
```

## 📊 Daily Health Reporter

Automatic 24-hour health reports for paper trading validation:

```bash
# Single report
py -3.11 -m aurix.reporting.daily_reporter

# Background daemon (24h interval)
py -3.11 -m aurix.reporting.daily_reporter --daemon
```

### Report Sections

| Section               | Metrics                               |
| --------------------- | ------------------------------------- |
| 1. System Liveness    | Candles processed, Redis status       |
| 2. Reality Validation | Avg/Min Reality Score, pass condition |
| 3. Growth State       | State distribution %, current state   |
| 4. Risk & Safety      | Kill Switch, Drawdown, CFI            |
| 5. Trading Summary    | Win rate, PnL, Profit Factor          |
| 6. Trend Snapshot     | Delta vs previous day                 |

### Verdict Levels

- ✅ **HEALTHY** - All constraints satisfied
- ⚠️ **WARNING** - Soft degradation, monitor closely
- ❌ **CRITICAL** - Pause validation immediately

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r python-services/requirements.txt

# 2. Configure
cp config/config.example.yaml config/config.yaml

# 3. Run backtest
py -3.11 python-services/run_backtest.py --start 2024-01-01 --end 2024-02-01

# 4. Start paper trading validation
py -3.11 python-services/run_validation.py --model-version v1.0.0
```

## 🎯 FINAL VALIDATION PHASES

### Phase 1: Long Paper Trading (MANDATORY)

- Duration: 30+ days on testnet
- Requirements:
  - Reality Score > 0.7 average
  - Max drawdown < 8%
  - Capital Fatigue Index < 0.5
  - Win rate > 50%

### Phase 2: Micro Real Capital

- Deploy with $100-500 real capital
- Duration: 14+ days
- Verify slippage model accuracy
- Confirm order execution quality

### Phase 3: Post-Mortem Discipline

- Weekly review of all losing trades
- Reality Score attribution analysis
- Parameter tuning based on CFI
- Document lessons learned

## Cek jumlah candles
py -3.11 -c "import sqlite3; c=sqlite3.connect('data/aurix.db'); print('Candles:', c.execute('SELECT COUNT(*) FROM candles').fetchone()[0])"

## Run ML accuracy test (jika >500 candles)
py -3.11 python-services/run_backtest.py --use-db --model-version v1.0.0

## ⚠️ Critical Rules

> **DO NOT skip paper trading phase.**
> **DO NOT deploy real capital until Paper Trading shows stable profits for 30+ days.**
> **DO NOT override kill switch without post-mortem analysis.**

## 📜 License

Private - All Rights Reserved
