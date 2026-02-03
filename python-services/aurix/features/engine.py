"""
AURIX Feature Engine

Computes 40+ technical indicators for ML model input.
All features are designed to be non-repainting and forward-looking safe.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Complete feature set for a single candle."""
    timestamp: int
    features: Dict[str, float]
    regime_features: Dict[str, float]


class FeatureEngine:
    """
    Technical feature computation engine.
    
    Categories:
    1. Volatility features (ATR, BB width, etc.)
    2. Momentum features (RSI, MACD, ROC, etc.)
    3. Volume features (OBV, VWAP deviation, etc.)
    4. Candle features (body ratio, wick ratio, etc.)
    5. Higher timeframe features (HTF trend, etc.)
    """
    
    def __init__(
        self,
        lookback_periods: List[int] = [5, 10, 20, 50],
        atr_period: int = 14,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        """
        Initialize feature engine.
        
        Args:
            lookback_periods: Periods for rolling computations
            atr_period: ATR calculation period
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
        """
        self.lookback_periods = lookback_periods
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Feature names for consistency
        self.feature_names: List[str] = []
    
    def compute_features(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Compute all features for the given candle data.
        
        Args:
            df_15m: 15-minute OHLCV DataFrame
            df_1h: 1-hour OHLCV DataFrame for HTF features
            
        Returns:
            DataFrame with all features
        """
        if len(df_15m) < 100:
            logger.warning(f"Insufficient data: {len(df_15m)} candles (need 100+)")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df_15m.index)
        
        # 1. Volatility Features
        features = self._add_volatility_features(features, df_15m)
        
        # 2. Momentum Features
        features = self._add_momentum_features(features, df_15m)
        
        # 3. Volume Features
        features = self._add_volume_features(features, df_15m)
        
        # 4. Candle Pattern Features
        features = self._add_candle_features(features, df_15m)
        
        # 5. Higher Timeframe Features
        if df_1h is not None and len(df_1h) >= 50:
            features = self._add_htf_features(features, df_15m, df_1h)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Drop NaN rows from lookback
        features = features.dropna()
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR
        features['atr'] = tr.rolling(self.atr_period).mean()
        features['atr_pct'] = features['atr'] / df['close']
        
        # ATR percentile (current ATR vs historical)
        features['atr_percentile'] = features['atr'].rolling(100).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x)
        )
        
        # Bollinger Bands
        sma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        upper_bb = sma + (std * self.bb_std)
        lower_bb = sma - (std * self.bb_std)
        
        features['bb_width'] = (upper_bb - lower_bb) / sma
        features['bb_position'] = (df['close'] - lower_bb) / (upper_bb - lower_bb)
        
        # Historical volatility
        log_returns = np.log(df['close'] / df['close'].shift(1))
        for period in self.lookback_periods:
            features[f'volatility_{period}'] = log_returns.rolling(period).std() * np.sqrt(252 * 24 * 4)  # Annualized
        
        # Volatility ratio (short vs long)
        features['vol_ratio_5_20'] = features.get('volatility_5', log_returns.rolling(5).std()) / features.get('volatility_20', log_returns.rolling(20).std())
        
        return features
    
    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI divergence (price vs RSI trend)
        price_slope = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        rsi_slope = features['rsi'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features['rsi_divergence'] = np.sign(price_slope) != np.sign(rsi_slope)
        
        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        features['macd'] = ema_fast - ema_slow
        features['macd_signal'] = features['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # MACD crossover
        features['macd_cross_up'] = (features['macd'] > features['macd_signal']) & (features['macd'].shift(1) <= features['macd_signal'].shift(1))
        features['macd_cross_down'] = (features['macd'] < features['macd_signal']) & (features['macd'].shift(1) >= features['macd_signal'].shift(1))
        
        # Rate of Change
        for period in self.lookback_periods:
            features[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm < 0), 0).abs()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        features['adx'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        
        # Volume SMA ratio
        vol_sma = df['volume'].rolling(20).mean()
        features['volume_sma_ratio'] = df['volume'] / vol_sma
        
        # OBV (On-Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv'] = obv
        features['obv_sma'] = obv.rolling(20).mean()
        features['obv_trend'] = (obv > features['obv_sma']).astype(int)
        
        # VWAP deviation
        vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap_deviation'] = (df['close'] - vwap) / vwap
        
        # Volume momentum
        features['volume_momentum'] = df['volume'].pct_change(5)
        
        # High volume candles
        features['high_volume'] = (df['volume'] > vol_sma * 2).astype(int)
        
        return features
    
    def _add_candle_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle pattern features."""
        
        # Body and wick ratios
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        features['body_ratio'] = body / total_range.replace(0, np.nan)
        
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        features['upper_wick_ratio'] = upper_wick / total_range.replace(0, np.nan)
        features['lower_wick_ratio'] = lower_wick / total_range.replace(0, np.nan)
        
        # Candle direction
        features['bullish'] = (df['close'] > df['open']).astype(int)
        
        # Consecutive candles
        features['consecutive_bullish'] = features['bullish'].rolling(5).sum()
        features['consecutive_bearish'] = 5 - features['consecutive_bullish']
        
        # Doji detection (small body)
        features['is_doji'] = (features['body_ratio'] < 0.1).astype(int)
        
        # Engulfing pattern
        prev_body = body.shift(1)
        features['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['open'].shift(1) > df['close'].shift(1)) &
            (body > prev_body) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        features['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['open'].shift(1) < df['close'].shift(1)) &
            (body > prev_body) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        return features
    
    def _add_htf_features(self, features: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        """Add higher timeframe features."""
        
        # 1H trend
        htf_sma_20 = df_1h['close'].rolling(20).mean()
        htf_sma_50 = df_1h['close'].rolling(50).mean()
        
        df_1h_features = pd.DataFrame(index=df_1h.index)
        df_1h_features['htf_trend'] = (htf_sma_20 > htf_sma_50).astype(int)
        df_1h_features['htf_sma_20'] = htf_sma_20
        df_1h_features['htf_sma_50'] = htf_sma_50
        
        # 1H RSI
        delta = df_1h['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_1h_features['htf_rsi'] = 100 - (100 / (1 + rs))
        
        # Resample to 15m
        df_1h_features = df_1h_features.reindex(df_15m.index, method='ffill')
        
        for col in df_1h_features.columns:
            features[col] = df_1h_features[col]
        
        # Distance from HTF levels
        features['dist_from_htf_sma20'] = (df_15m['close'] - features['htf_sma_20']) / features['htf_sma_20']
        features['dist_from_htf_sma50'] = (df_15m['close'] - features['htf_sma_50']) / features['htf_sma_50']
        
        return features
    
    def get_latest_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame = None) -> Optional[Dict[str, float]]:
        """
        Get features for the latest candle only.
        
        Returns:
            Dictionary of feature name -> value, or None if insufficient data
        """
        features_df = self.compute_features(df_15m, df_1h)
        if len(features_df) == 0:
            return None
        
        latest = features_df.iloc[-1].to_dict()
        return latest
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names
