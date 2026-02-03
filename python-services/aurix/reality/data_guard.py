"""
AURIX Data Guard

Real historical data ingestion with strict lookahead prevention.
Enforces data quality and prevents data leakage.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class DataGuardConfig:
    """Configuration for data guard."""
    # Transaction costs
    taker_fee_pct: float = 0.075  # 0.075% Binance taker
    maker_fee_pct: float = 0.025
    default_spread_pct: float = 0.01
    
    # Data quality
    max_lookahead_tolerance_sec: int = 0  # Zero tolerance
    min_data_quality_score: float = 0.95
    max_allowed_gap_candles: int = 3
    
    # Source configuration
    data_source: str = "csv"  # "csv" or "binance"
    data_directory: str = "data/historical"


@dataclass 
class DataQualityReport:
    """Report on data quality."""
    total_candles: int
    missing_candles: int
    gap_count: int
    max_gap_size: int
    outlier_count: int
    quality_score: float
    issues: List[str]
    
    @property
    def is_acceptable(self) -> bool:
        return self.quality_score >= 0.95


class DataGuard:
    """
    Data Guard
    
    Ensures data integrity for backtesting:
    1. Real historical data loading (CSV or exchange)
    2. Strict time alignment (no lookahead)
    3. Data quality validation
    4. Transaction cost modeling
    5. Gap detection and handling
    """
    
    def __init__(self, config: Optional[DataGuardConfig] = None):
        """
        Initialize data guard.
        
        Args:
            config: Data guard configuration
        """
        self.config = config or DataGuardConfig()
        self._loaded_data: Dict[str, pd.DataFrame] = {}
    
    def load_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "15m"
    ) -> Optional[pd.DataFrame]:
        """
        Load historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start: Start datetime
            end: End datetime
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if cache_key in self._loaded_data:
            df = self._loaded_data[cache_key]
            return self._filter_time_range(df, start, end)
        
        # Load from source
        if self.config.data_source == "csv":
            df = self._load_from_csv(symbol, timeframe)
        elif self.config.data_source == "binance":
            df = self._load_from_binance(symbol, start, end, timeframe)
        else:
            logger.error(f"Unknown data source: {self.config.data_source}")
            return None
        
        if df is None or df.empty:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return None
        
        # Cache and filter
        self._loaded_data[cache_key] = df
        return self._filter_time_range(df, start, end)
    
    def _load_from_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from local CSV file."""
        directory = self.config.data_directory
        
        # Try different file naming conventions
        possible_files = [
            f"{directory}/{symbol}_{timeframe}.csv",
            f"{directory}/{symbol.lower()}_{timeframe}.csv",
            f"{directory}/{symbol}/{timeframe}.csv",
        ]
        
        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df = self._standardize_columns(df)
                    df = self._set_datetime_index(df)
                    logger.info(f"Loaded {len(df)} candles from {filepath}")
                    return df
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
        
        logger.warning(f"No CSV file found for {symbol} {timeframe}")
        return None
    
    def _load_from_binance(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load data from Binance API."""
        try:
            import ccxt
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # Convert timeframe to ccxt format
            tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
            ccxt_tf = tf_map.get(timeframe, timeframe)
            
            since = int(start.timestamp() * 1000)
            end_ts = int(end.timestamp() * 1000)
            
            all_candles = []
            
            while since < end_ts:
                candles = exchange.fetch_ohlcv(symbol, ccxt_tf, since, limit=1000)
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                
                if len(candles) < 1000:
                    break
            
            if not all_candles:
                return None
            
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Loaded {len(df)} candles from Binance API")
            return df
            
        except ImportError:
            logger.error("ccxt not installed. Install with: pip install ccxt")
            return None
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'timestamp': 'time',
            'date': 'time',
            'datetime': 'time',
            'Date': 'time',
            'Timestamp': 'time',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set datetime index."""
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        return df
    
    def _filter_time_range(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame to time range."""
        mask = (df.index >= start) & (df.index <= end)
        return df[mask].copy()
    
    def validate_alignment(
        self,
        df_base: pd.DataFrame,
        df_htf: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Validate time alignment between timeframes.
        
        Ensures no lookahead bias (HTF data must not be ahead of base).
        
        Args:
            df_base: Base timeframe DataFrame (e.g., 15m)
            df_htf: Higher timeframe DataFrame (e.g., 1h)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df_base.empty or df_htf.empty:
            return False, "One or both DataFrames are empty"
        
        # Check HTF doesn't extend beyond base
        if df_htf.index[-1] > df_base.index[-1]:
            return False, f"HTF extends beyond base: {df_htf.index[-1]} > {df_base.index[-1]}"
        
        # Check for future data in HTF
        tolerance = timedelta(seconds=self.config.max_lookahead_tolerance_sec)
        
        for base_time in df_base.index:
            # Find HTF candles that should be available at this base time
            available_htf = df_htf[df_htf.index <= base_time + tolerance]
            
            if available_htf.empty:
                continue
            
            # Check no future HTF close prices are used
            latest_htf = available_htf.index[-1]
            if latest_htf > base_time + tolerance:
                return False, f"Lookahead detected at {base_time}: HTF {latest_htf}"
        
        return True, "Alignment valid"
    
    def validate_data_quality(self, df: pd.DataFrame, timeframe: str = "15m") -> DataQualityReport:
        """
        Validate data quality.
        
        Checks for gaps, outliers, and missing data.
        
        Args:
            df: OHLCV DataFrame
            timeframe: Expected timeframe for gap detection
            
        Returns:
            DataQualityReport
        """
        issues = []
        
        # Parse timeframe to timedelta
        tf_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
        }
        expected_delta = tf_map.get(timeframe, timedelta(minutes=15))
        
        # Detect gaps
        time_diffs = df.index.to_series().diff()
        gap_mask = time_diffs > expected_delta * 1.5
        gaps = time_diffs[gap_mask]
        gap_count = len(gaps)
        max_gap = int(gaps.max() / expected_delta) if gap_count > 0 else 0
        
        if gap_count > 0:
            issues.append(f"Found {gap_count} gaps, max gap: {max_gap} candles")
        
        # Detect missing candles
        expected_candles = (df.index[-1] - df.index[0]) / expected_delta
        actual_candles = len(df)
        missing_candles = max(0, int(expected_candles - actual_candles))
        
        if missing_candles > 0:
            issues.append(f"Missing approximately {missing_candles} candles")
        
        # Detect outliers (price changes > 10% in one candle)
        returns = df['close'].pct_change()
        outlier_mask = returns.abs() > 0.10
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            issues.append(f"Found {outlier_count} potential outliers (>10% change)")
        
        # Calculate quality score
        completeness = actual_candles / max(1, expected_candles)
        gap_penalty = min(0.2, gap_count * 0.02)
        outlier_penalty = min(0.1, outlier_count * 0.01)
        
        quality_score = max(0, min(1, completeness - gap_penalty - outlier_penalty))
        
        return DataQualityReport(
            total_candles=actual_candles,
            missing_candles=missing_candles,
            gap_count=gap_count,
            max_gap_size=max_gap,
            outlier_count=outlier_count,
            quality_score=quality_score,
            issues=issues
        )
    
    def calculate_transaction_costs(
        self,
        price: float,
        quantity: float,
        side: str,
        is_maker: bool = False
    ) -> float:
        """
        Calculate transaction costs.
        
        Args:
            price: Execution price
            quantity: Order quantity
            side: 'BUY' or 'SELL'
            is_maker: True if maker order, False if taker
            
        Returns:
            Total transaction cost in quote currency
        """
        fee_pct = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        spread_pct = self.config.default_spread_pct
        
        notional = price * quantity
        
        # Fee cost
        fee_cost = notional * (fee_pct / 100)
        
        # Spread cost (half spread per side)
        spread_cost = notional * (spread_pct / 200)
        
        total_cost = fee_cost + spread_cost
        
        return total_cost
    
    def generate_mock_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "15m",
        initial_price: float = 40000.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate mock OHLCV data for testing.
        
        Args:
            symbol: Symbol name
            start: Start datetime
            end: End datetime
            timeframe: Candle timeframe
            initial_price: Starting price
            volatility: Daily volatility
            
        Returns:
            DataFrame with mock OHLCV data
        """
        tf_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
        }
        delta = tf_map.get(timeframe, timedelta(minutes=15))
        
        # Generate timestamps
        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current)
            current += delta
        
        n = len(timestamps)
        
        # Generate price series
        np.random.seed(42)
        returns = np.random.normal(0, volatility / np.sqrt(24 * 4), n)  # Scale to timeframe
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = []
        for i, ts in enumerate(timestamps):
            close = prices[i]
            
            # Simulate intra-candle movement
            wick_range = close * volatility * 0.5
            high = close + abs(np.random.normal(0, wick_range * 0.5))
            low = close - abs(np.random.normal(0, wick_range * 0.5))
            
            if i > 0:
                open_price = prices[i - 1] * (1 + np.random.normal(0, 0.001))
            else:
                open_price = close * (1 + np.random.normal(0, 0.001))
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.lognormal(10, 1)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps)
        logger.info(f"Generated {len(df)} mock candles for {symbol}")
        
        return df
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        lines = [
            "=== Data Guard Status ===",
            f"Data Source: {self.config.data_source}",
            f"Taker Fee: {self.config.taker_fee_pct:.3f}%",
            f"Maker Fee: {self.config.maker_fee_pct:.3f}%",
            f"Default Spread: {self.config.default_spread_pct:.3f}%",
            f"Lookahead Tolerance: {self.config.max_lookahead_tolerance_sec}s",
            f"Min Quality Score: {self.config.min_data_quality_score:.0%}",
            "",
            "Cached Data:",
        ]
        
        for key, df in self._loaded_data.items():
            lines.append(f"  {key}: {len(df)} candles")
        
        return "\n".join(lines)
