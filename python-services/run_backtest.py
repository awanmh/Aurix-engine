"""
AURIX Backtest Runner

Command-line interface for running walk-forward backtests.

Usage:
    python run_backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
    python run_backtest.py --data data/candles.csv --compare-static
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7

from aurix.backtest import (
    WalkForwardBacktester,
    BacktestConfig,
    LearningMode,
    ReportGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


import sqlite3

def load_candles_from_csv(filepath: str, timeframe: str = '15m') -> pd.DataFrame:
    """Load candles from CSV file."""
    df = pd.read_csv(filepath)
    
    # Handle different column naming conventions
    column_mapping = {
        'timestamp': 'time',
        'date': 'time',
        'datetime': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Parse time column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    
    return df


def load_candles_from_db(db_path: str, timeframe: str = '15m') -> pd.DataFrame:
    """Load candles from SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
        
    conn = sqlite3.connect(db_path)
    try:
        query = f"""
            SELECT open_time, open, high, low, close, volume 
            FROM candles 
            WHERE timeframe = '{timeframe}'
            ORDER BY open_time ASC
        """
        df = pd.read_sql_query(query, conn)
        
        # Rename columns to match backtester expectations
        df = df.rename(columns={'open_time': 'time'})
        
        # Convert timestamp (assuming millisecond timestamp from Binance)
        # Check if timestamp is in ms or seconds (Binance uses ms)
        if not df.empty:
            sample_ts = df['time'].iloc[0]
            if sample_ts > 10000000000: # heuristic for ms
                df['time'] = pd.to_datetime(df['time'], unit='ms')
            else:
                df['time'] = pd.to_datetime(df['time'], unit='s')
        
        df = df.set_index('time')
        return df
    finally:
        conn.close()


def generate_mock_data(
    start_date: datetime,
    end_date: datetime,
    symbol: str = 'BTCUSDT'
) -> tuple:
    """
    Generate realistic mock price data for backtesting.
    
    This creates synthetic data following a geometric Brownian motion
    with regime changes and volatility clustering.
    """
    logger.info(f"Generating mock data from {start_date} to {end_date}")
    
    # Calculate number of 15-minute periods
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    n_candles_15m = total_minutes // 15
    
    # Parameters
    initial_price = 50000
    annual_drift = 0.05  # 5% annual drift
    annual_vol = 0.60    # 60% annual volatility
    
    # Convert to 15-minute parameters
    dt = 15 / (252 * 24 * 60)  # 15 minutes as fraction of trading year
    drift = annual_drift * dt
    vol = annual_vol * np.sqrt(dt)
    
    # Generate price path with regime changes
    prices = [initial_price]
    regimes = []
    current_regime = 'TRENDING_UP'
    regime_duration = 0
    
    for i in range(n_candles_15m - 1):
        # Regime transition probability
        regime_duration += 1
        if regime_duration > 200:  # Average 200 candles per regime
            if np.random.random() < 0.02:  # 2% chance to switch
                regimes_options = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE']
                current_regime = np.random.choice(regimes_options)
                regime_duration = 0
        
        regimes.append(current_regime)
        
        # Adjust drift/vol based on regime
        if current_regime == 'TRENDING_UP':
            regime_drift = drift * 2
            regime_vol = vol * 0.8
        elif current_regime == 'TRENDING_DOWN':
            regime_drift = -drift * 2
            regime_vol = vol * 0.8
        elif current_regime == 'VOLATILE':
            regime_drift = 0
            regime_vol = vol * 2.0
        else:  # RANGING
            regime_drift = 0
            regime_vol = vol * 0.6
        
        # GBM step
        random_return = np.random.normal(regime_drift, regime_vol)
        new_price = prices[-1] * (1 + random_return)
        prices.append(max(new_price, 1000))  # Price floor
    
    regimes.append(current_regime)  # Last one
    
    # Generate OHLC from close prices
    # Add some intracandle volatility
    opens = []
    highs = []
    lows = []
    closes = prices
    volumes = []
    
    for i, close in enumerate(closes):
        intra_vol = close * 0.002  # 0.2% intracandle range
        
        open_price = close + np.random.uniform(-intra_vol, intra_vol)
        high_price = max(open_price, close) + np.random.uniform(0, intra_vol * 2)
        low_price = min(open_price, close) - np.random.uniform(0, intra_vol * 2)
        volume = np.random.uniform(50, 500)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
    
    # Create DataFrame
    timestamps = pd.date_range(start=start_date, periods=n_candles_15m, freq='15T')
    
    df_15m = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)
    
    # Generate 1h candles from 15m
    df_1h = df_15m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logger.info(f"Generated {len(df_15m)} 15m candles, {len(df_1h)} 1h candles")
    
    return df_15m, df_1h


def run_backtest(args):
    """Run the backtest with given arguments."""
    
    # Load or generate data
    if args.use_db:
        db_path = args.db_path or 'data/aurix.db'
        logger.info(f"Loading data from database: {db_path}")
        
        # Load 15m candles for backtesting logic/features
        candles_15m = load_candles_from_db(db_path, '15m')
        
        # Load or resample 1h candles
        try:
            candles_1h = load_candles_from_db(db_path, '1h')
            if candles_1h.empty:
                logger.warning("No 1h candles in DB, resampling from 15m")
                candles_1h = candles_15m.resample('1H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
        except Exception:
            # Fallback to resampling if loading fails
            candles_1h = candles_15m.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
        logger.info(f"Loaded {len(candles_15m)} 15m candles from DB")
        
    elif args.data:
        logger.info(f"Loading data from {args.data}")
        candles_15m = load_candles_from_csv(args.data, '15m')
        candles_1h = candles_15m.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else datetime(2024, 1, 1)
        end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime(2024, 6, 30)
        candles_15m, candles_1h = generate_mock_data(start_date, end_date, args.symbol)
    
    # Configure backtest
    learning_mode = LearningMode[args.learning_mode.upper()]
    
    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade_percent=args.risk,
        base_confidence_threshold=args.threshold,
        learning_mode=learning_mode,
        initial_train_days=args.warmup_days,
        retrain_interval_hours=args.retrain_hours
    )
    
    # Create backtester
    backtester = WalkForwardBacktester(config)
    
    # Run backtest
    logger.info("Running walk-forward backtest...")
    online_metrics, static_metrics = backtester.run(
        candles_1m=pd.DataFrame(),  # Not needed
        candles_15m=candles_15m,
        candles_1h=candles_1h,
        symbol=args.symbol,
        compare_static=args.compare_static
    )
    
    # Generate report
    logger.info("Generating performance report...")
    report_gen = ReportGenerator(online_metrics, backtester.state.closed_trades)
    report = report_gen.generate_markdown_report()
    
    # Print report
    print("\n" + "="*80)
    print(report)
    print("="*80 + "\n")
    
    # Save report
    os.makedirs('data/reports', exist_ok=True)
    report_path = f"data/reports/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_gen.save_report(report_path, format='markdown')
    logger.info(f"Report saved to {report_path}")
    
    # If comparing static vs online, print comparison
    if static_metrics:
        print("\n" + "="*80)
        print("ONLINE LEARNING vs STATIC MODEL COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<30} {'Online':>15} {'Static':>15} {'Diff':>15}")
        print("-"*75)
        
        comparisons = [
            ('Total Return %', online_metrics.total_return_pct * 100, static_metrics.total_return_pct * 100),
            ('Win Rate %', online_metrics.win_rate * 100, static_metrics.win_rate * 100),
            ('Profit Factor', online_metrics.profit_factor, static_metrics.profit_factor),
            ('Max Drawdown %', online_metrics.max_drawdown_pct * 100, static_metrics.max_drawdown_pct * 100),
            ('Sharpe Ratio', online_metrics.sharpe_ratio, static_metrics.sharpe_ratio),
            ('Total Trades', online_metrics.total_trades, static_metrics.total_trades),
            ('Expectancy $', online_metrics.expectancy, static_metrics.expectancy),
        ]
        
        for name, online_val, static_val in comparisons:
            diff = online_val - static_val
            diff_str = f"{diff:+.2f}"
            print(f"{name:<30} {online_val:>15.2f} {static_val:>15.2f} {diff_str:>15}")
        
        print("\n")
        if online_metrics.total_return_pct > static_metrics.total_return_pct:
            print("📈 ONLINE LEARNING OUTPERFORMED static model")
        else:
            print("📉 STATIC MODEL OUTPERFORMED online learning")
    
    # Return recommendation
    rec = report_gen._generate_recommendations()
    print(f"\n{rec.summary}")
    
    return online_metrics


def main():
    parser = argparse.ArgumentParser(description='AURIX Walk-Forward Backtester')
    
    # Data source
    parser.add_argument('--data', type=str, help='Path to CSV file with OHLCV data')
    parser.add_argument('--use-db', action='store_true', help='Use SQLite database as data source')
    parser.add_argument('--db-path', type=str, default='data/aurix.db', help='Path to SQLite database')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    
    # Capital and risk
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--risk', type=float, default=1.0, help='Risk per trade (%)')
    parser.add_argument('--threshold', type=float, default=0.60, help='Base confidence threshold')
    
    # Learning mode
    parser.add_argument('--learning-mode', type=str, default='periodic',
                       choices=['static', 'periodic', 'adaptive', 'continuous'],
                       help='Model learning strategy')
    parser.add_argument('--warmup-days', type=int, default=14, help='Initial training period (days)')
    parser.add_argument('--retrain-hours', type=int, default=24, help='Retrain interval (hours)')
    
    # Comparison
    parser.add_argument('--compare-static', action='store_true',
                       help='Also run with static model for comparison')
    
    # Output
    parser.add_argument('--output', type=str, help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        run_backtest(args)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
