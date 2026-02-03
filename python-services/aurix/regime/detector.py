"""
AURIX Market Regime Detector

Classifies market conditions for dynamic threshold adjustment.
Implements Audit Fix #4: Regime detection filter.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with metrics."""
    regime: MarketRegime
    confidence: float
    volatility_percentile: float
    trend_strength: float
    range_score: float
    size_multiplier: float = 1.0  # Position size multiplier based on regime


class RegimeDetector:
    """
    Detects market regime using multiple indicators.
    
    Regimes:
    - TRENDING_UP: Strong upward price movement
    - TRENDING_DOWN: Strong downward price movement
    - RANGING: Sideways, mean-reverting market
    - VOLATILE: High volatility, unpredictable
    """
    
    # Regime-specific confidence adjustments
    REGIME_ADJUSTMENTS = {
        MarketRegime.TRENDING_UP: -0.02,    # Lower threshold (edge is higher)
        MarketRegime.TRENDING_DOWN: -0.02,  # Lower threshold
        MarketRegime.RANGING: 0.05,         # Higher threshold (harder to trade)
        MarketRegime.VOLATILE: 0.10,        # Much higher threshold
        MarketRegime.UNKNOWN: 0.15          # Very conservative
    }
    
    def __init__(
        self,
        sma_short_period: int = 20,
        sma_long_period: int = 50,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        volatility_lookback: int = 100,
        volatility_high_percentile: float = 75
    ):
        """
        Initialize regime detector.
        
        Args:
            sma_short_period: Short SMA period for trend detection
            sma_long_period: Long SMA period for trend detection
            adx_period: ADX calculation period
            adx_trend_threshold: ADX value above which market is trending
            volatility_lookback: Periods for volatility percentile calculation
            volatility_high_percentile: Percentile threshold for high volatility
        """
        self.sma_short = sma_short_period
        self.sma_long = sma_long_period
        self.adx_period = adx_period
        self.adx_threshold = adx_trend_threshold
        self.vol_lookback = volatility_lookback
        self.vol_high_pct = volatility_high_percentile
        
        # Cache for regime history
        self.regime_history: list = []
    
    def detect_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            RegimeState with current regime and metrics
        """
        if len(df) < self.sma_long + 10:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                volatility_percentile=0.5,
                trend_strength=0.0,
                range_score=0.5
            )
        
        # Calculate indicators
        metrics = self._calculate_metrics(df)
        
        # Determine regime
        regime, confidence = self._classify_regime(metrics)
        
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_percentile=metrics['volatility_percentile'],
            trend_strength=metrics['adx'],
            range_score=metrics['range_score']
        )
        
        # Update history
        self.regime_history.append(state)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return state
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all regime detection metrics."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # SMAs
        sma_short = close.rolling(self.sma_short).mean()
        sma_long = close.rolling(self.sma_long).mean()
        
        # Current values
        current_price = close.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        
        # Trend direction
        trend_direction = 1 if current_sma_short > current_sma_long else -1
        sma_spread = abs(current_sma_short - current_sma_long) / current_price
        
        # ADX (Average Directional Index)
        adx = self._calculate_adx(df)
        
        # Volatility (ATR percentile)
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        volatility_percentile = (atr.iloc[-self.vol_lookback:] < current_atr).sum() / self.vol_lookback
        
        # Range score (how much price stays within Bollinger Bands)
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        
        # Count how often price touched bands in last 20 periods
        touches_upper = (high.iloc[-20:] >= upper_bb.iloc[-20:]).sum()
        touches_lower = (low.iloc[-20:] <= lower_bb.iloc[-20:]).sum()
        range_score = 1 - (touches_upper + touches_lower) / 40  # More touches = less ranging
        
        return {
            'trend_direction': trend_direction,
            'sma_spread': sma_spread,
            'adx': adx,
            'volatility_percentile': volatility_percentile,
            'range_score': range_score,
            'price_vs_sma_short': (current_price - current_sma_short) / current_sma_short,
            'price_vs_sma_long': (current_price - current_sma_long) / current_sma_long
        }
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smooth values
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * plus_dm.rolling(self.adx_period).mean() / atr
        minus_di = 100 * minus_dm.rolling(self.adx_period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        return tr.rolling(14).mean()
    
    def _classify_regime(self, metrics: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on metrics.
        
        Returns:
            Tuple of (regime, confidence)
        """
        adx = metrics['adx']
        vol_pct = metrics['volatility_percentile']
        trend_dir = metrics['trend_direction']
        sma_spread = metrics['sma_spread']
        range_score = metrics['range_score']
        
        # High volatility check first
        if vol_pct > self.vol_high_pct / 100:
            confidence = min(1.0, vol_pct)
            return MarketRegime.VOLATILE, confidence
        
        # Strong trend check
        if adx > self.adx_threshold:
            confidence = min(1.0, adx / 50)
            if trend_dir > 0:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Ranging market
        if range_score > 0.7:
            confidence = range_score
            return MarketRegime.RANGING, confidence
        
        # Low ADX but not clearly ranging
        if adx < 20:
            # Check if we might be transitioning
            if sma_spread > 0.005:  # 0.5% spread between SMAs
                if trend_dir > 0:
                    return MarketRegime.TRENDING_UP, 0.5
                else:
                    return MarketRegime.TRENDING_DOWN, 0.5
            return MarketRegime.RANGING, 0.6
        
        return MarketRegime.UNKNOWN, 0.3
    
    def get_confidence_adjustment(self, regime: MarketRegime = None) -> float:
        """
        Get confidence threshold adjustment for current regime.
        
        Args:
            regime: Specific regime, or None to use latest detected
            
        Returns:
            Adjustment to add to base confidence threshold
        """
        if regime is None:
            if not self.regime_history:
                return self.REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN]
            regime = self.regime_history[-1].regime
        
        return self.REGIME_ADJUSTMENTS.get(regime, 0.0)
    
    def get_regime_stats(self, lookback: int = 100) -> Dict[str, float]:
        """
        Get regime distribution statistics.
        
        Args:
            lookback: Number of periods to look back
            
        Returns:
            Dict with regime percentages
        """
        if len(self.regime_history) < lookback:
            lookback = len(self.regime_history)
        
        if lookback == 0:
            return {}
        
        recent = self.regime_history[-lookback:]
        
        counts = {}
        for state in recent:
            regime_name = state.regime.value
            counts[regime_name] = counts.get(regime_name, 0) + 1
        
        return {k: v / lookback for k, v in counts.items()}
    
    def is_favorable_regime(self, regime: MarketRegime = None) -> bool:
        """Check if current regime is favorable for trading."""
        if regime is None and self.regime_history:
            regime = self.regime_history[-1].regime
        
        # Trending markets are more favorable
        return regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
