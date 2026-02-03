# AURIX Final Validation Guide

## Phase 1: Long Paper Trading (30+ days)

### Start Command

```bash
py -3.11 python-services/run_validation.py --model-version v1.0.0 --mode paper
```

### Daily Checklist

- [ ] Reality Score > 0.7
- [ ] Growth State = Accumulation or Expansion
- [ ] Capital Fatigue Index < 0.5
- [ ] No kill switch triggers

### Weekly Review

- Export trade log
- Analyze Reality Score attribution
- Check grinding phase occurrences
- Document parameter adjustments

### Graduation Criteria (ALL required)

| Metric               | Target   |
| -------------------- | -------- |
| Duration             | 30+ days |
| Reality Score Avg    | > 0.7    |
| Max Drawdown         | < 8%     |
| Win Rate             | > 50%    |
| Profit Factor        | > 1.2    |
| Kill Switch Triggers | 0        |

---

## Phase 2: Micro Real Capital (14+ days)

### Deploy Amount: $100-500 USD

### Start Command

```bash
py -3.11 python-services/run_validation.py --model-version v1.0.0 --mode live --capital 100
```

### Focus Areas

1. **Slippage Accuracy** - Compare model vs actual
2. **Order Fill Rate** - Track partial fills
3. **Latency Impact** - Measure execution delay
4. **Fee Impact** - Verify cost model

### Graduation Criteria

| Metric             | Target           |
| ------------------ | ---------------- |
| Duration           | 14+ days         |
| Slippage Deviation | < 20% from model |
| Order Fill Rate    | > 95%            |
| Reality Score Avg  | > 0.7            |

---

## Phase 3: Post-Mortem Discipline

### After Every Losing Day

1. Export Reality Score attribution
2. Identify primary cause (overfit, slippage, etc)
3. Document in `postmortems/YYYY-MM-DD.md`

### Weekly Post-Mortem Template

```markdown
# Week of YYYY-MM-DD

## Summary

- Trades: X
- Win Rate: X%
- Reality Score Avg: X
- Growth States: Accumulation X%, Defense X%

## Top Issues

1. [Issue 1] - [Root cause] - [Action taken]
2. [Issue 2] - [Root cause] - [Action taken]

## Parameter Changes

- Changed X from Y to Z because...

## Next Week Focus

- Monitor X more closely
```

### Kill Switch Post-Mortem (MANDATORY)

If kill switch triggers:

1. **DO NOT** resume trading until analysis complete
2. Identify exact trigger (drawdown/losses/confidence)
3. Review last 10 trades
4. Adjust parameters if needed
5. Document in `postmortems/killswitch-YYYY-MM-DD.md`
6. Wait full recovery protocol (12h cooldown)

---

## Commands Reference

```bash
# Paper trading
py -3.11 python-services/run_validation.py --model-version v1.0.0 --mode paper

# Backtest comparison
py -3.11 python-services/run_backtest.py --start 2024-01-01 --end 2024-02-01

# Generate post-mortem data
py -3.11 -c "from aurix.reality import RealityScorer; s=RealityScorer(); print(s.generate_postmortem())"

# Check growth state
py -3.11 -c "from aurix.capital import GrowthOrchestrator; print(GrowthOrchestrator().get_status_report())"
```
