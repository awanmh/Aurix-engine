"""
AURIX Pair Manager & Ranker

Implements pair ranking and rotation logic for multi-pair optimization.
Focuses capital on highest-opportunity pairs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PairStatus(Enum):
    """Trading status for a pair."""
    ACTIVE = "active"          # Currently trading
    WATCHLIST = "watchlist"    # Monitoring, not trading
    DEMOTED = "demoted"        # Recently removed from active
    PROMOTED = "promoted"      # Recently added to active
    BLACKLISTED = "blacklisted"  # Do not trade


@dataclass
class PairMetrics:
    """Metrics for a trading pair."""
    symbol: str
    volatility_quality: float  # 0-1, higher = better
    liquidity_score: float     # 0-1, higher = better
    edge_stability: float      # 0-1, higher = better (backtest consistency)
    recent_ces: float          # Last 7d capital efficiency score
    correlation_penalty: float # 0-1, higher = more correlated to portfolio
    composite_score: float     # Weighted combination
    rank: int                  # Position in ranking
    status: PairStatus
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'volatility_quality': self.volatility_quality,
            'liquidity_score': self.liquidity_score,
            'edge_stability': self.edge_stability,
            'recent_ces': self.recent_ces,
            'correlation_penalty': self.correlation_penalty,
            'composite_score': self.composite_score,
            'rank': self.rank,
            'status': self.status.value,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class PairRanking:
    """Result of pair ranking operation."""
    rankings: List[PairMetrics]
    active_pairs: List[str]
    demoted_pairs: List[str]
    promoted_pairs: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PairManager:
    """
    Manages pair selection and rotation for optimal capital deployment.
    
    Ranking Criteria:
    - Volatility Quality (25%): ATR relative to spread, clean trends
    - Liquidity Score (20%): Volume, order book depth
    - Edge Stability (25%): Backtest expectancy consistency
    - Recent Performance (20%): Last 7-day CES on this pair
    - Correlation Penalty (10%): Reduce if correlated to existing positions
    """
    
    DEFAULT_WEIGHTS = {
        'volatility_quality': 0.25,
        'liquidity_score': 0.20,
        'edge_stability': 0.25,
        'recent_ces': 0.20,
        'correlation_penalty': 0.10
    }
    
    def __init__(
        self,
        max_active_pairs: int = 5,
        rotation_frequency_days: int = 7,
        min_rank_percentile: float = 0.7,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize pair manager.
        
        Args:
            max_active_pairs: Maximum number of actively traded pairs
            rotation_frequency_days: Days between rotation evaluations
            min_rank_percentile: Minimum percentile to be traded (0.7 = top 30%)
            weights: Custom weights for ranking criteria
        """
        self.max_active = max_active_pairs
        self.rotation_days = rotation_frequency_days
        self.min_rank_pct = min_rank_percentile
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # State
        self.pairs: Dict[str, PairMetrics] = {}
        self.active_pairs: List[str] = []
        self.last_rotation: Optional[datetime] = None
        
        # Historical data for edge stability calculation
        self.backtest_history: Dict[str, List[float]] = {}  # symbol -> list of expectancies
        self.trade_history: Dict[str, List[dict]] = {}  # symbol -> list of trade records
    
    def add_pair(self, symbol: str):
        """Add a new pair to track."""
        if symbol not in self.pairs:
            self.pairs[symbol] = PairMetrics(
                symbol=symbol,
                volatility_quality=0.5,
                liquidity_score=0.5,
                edge_stability=0.5,
                recent_ces=0.5,
                correlation_penalty=0.0,
                composite_score=0.5,
                rank=len(self.pairs) + 1,
                status=PairStatus.WATCHLIST
            )
            self.backtest_history[symbol] = []
            self.trade_history[symbol] = []
    
    def update_pair_metrics(
        self,
        symbol: str,
        candles: pd.DataFrame,
        avg_spread_pct: float = 0.0005,
        avg_volume_24h: float = 0.0,
        backtest_expectancy: Optional[float] = None,
        recent_trades: Optional[List[dict]] = None
    ):
        """
        Update metrics for a pair.
        
        Args:
            symbol: Trading pair symbol
            candles: OHLCV DataFrame
            avg_spread_pct: Average bid-ask spread as percentage
            avg_volume_24h: 24-hour trading volume in USD
            backtest_expectancy: Latest backtest expectancy
            recent_trades: Recent trade records for this pair
        """
        if symbol not in self.pairs:
            self.add_pair(symbol)
        
        pair = self.pairs[symbol]
        
        # Update volatility quality
        pair.volatility_quality = self._calculate_volatility_quality(candles, avg_spread_pct)
        
        # Update liquidity score
        pair.liquidity_score = self._calculate_liquidity_score(candles, avg_volume_24h)
        
        # Update edge stability
        if backtest_expectancy is not None:
            self.backtest_history[symbol].append(backtest_expectancy)
            if len(self.backtest_history[symbol]) > 20:
                self.backtest_history[symbol] = self.backtest_history[symbol][-20:]
        pair.edge_stability = self._calculate_edge_stability(symbol)
        
        # Update recent CES
        if recent_trades:
            self.trade_history[symbol].extend(recent_trades)
            cutoff = datetime.now() - timedelta(days=7)
            self.trade_history[symbol] = [
                t for t in self.trade_history[symbol]
                if datetime.fromisoformat(t.get('exit_time', datetime.now().isoformat())) > cutoff
            ]
        pair.recent_ces = self._calculate_recent_ces(symbol)
        
        pair.last_updated = datetime.now()
    
    def update_correlations(self, correlation_matrix: pd.DataFrame):
        """
        Update correlation penalties based on correlation matrix.
        
        Args:
            correlation_matrix: DataFrame with pair correlations
        """
        for symbol in self.pairs:
            if symbol not in correlation_matrix.index:
                continue
            
            # Calculate penalty based on correlation to active pairs
            penalty = 0.0
            for active in self.active_pairs:
                if active in correlation_matrix.columns and active != symbol:
                    corr = abs(correlation_matrix.loc[symbol, active])
                    penalty += corr * 0.5  # Weight correlation penalty
            
            # Normalize penalty
            if self.active_pairs:
                penalty /= len(self.active_pairs)
            
            self.pairs[symbol].correlation_penalty = min(1.0, penalty)
    
    def rank_pairs(self) -> PairRanking:
        """
        Rank all pairs and determine active set.
        
        Returns:
            PairRanking with updated rankings
        """
        if not self.pairs:
            return PairRanking(
                rankings=[],
                active_pairs=[],
                demoted_pairs=[],
                promoted_pairs=[]
            )
        
        # Calculate composite scores
        for symbol, pair in self.pairs.items():
            if pair.status == PairStatus.BLACKLISTED:
                continue
            
            # Composite score (correlation penalty reduces score)
            pair.composite_score = (
                pair.volatility_quality * self.weights['volatility_quality'] +
                pair.liquidity_score * self.weights['liquidity_score'] +
                pair.edge_stability * self.weights['edge_stability'] +
                pair.recent_ces * self.weights['recent_ces'] -
                pair.correlation_penalty * self.weights['correlation_penalty']
            )
        
        # Sort by composite score
        sorted_pairs = sorted(
            [p for p in self.pairs.values() if p.status != PairStatus.BLACKLISTED],
            key=lambda p: p.composite_score,
            reverse=True
        )
        
        # Assign ranks
        for i, pair in enumerate(sorted_pairs):
            pair.rank = i + 1
        
        # Determine active pairs
        min_rank = int(len(sorted_pairs) * (1 - self.min_rank_pct))
        new_active = [p.symbol for p in sorted_pairs[:self.max_active] if p.rank <= min_rank or min_rank == 0]
        
        # Track demotions and promotions
        demoted = [s for s in self.active_pairs if s not in new_active]
        promoted = [s for s in new_active if s not in self.active_pairs]
        
        # Update statuses
        for symbol in demoted:
            if symbol in self.pairs:
                self.pairs[symbol].status = PairStatus.DEMOTED
                logger.info(f"Demoted pair: {symbol}")
        
        for symbol in promoted:
            if symbol in self.pairs:
                self.pairs[symbol].status = PairStatus.PROMOTED
                logger.info(f"Promoted pair: {symbol}")
        
        for symbol in new_active:
            if symbol in self.pairs:
                self.pairs[symbol].status = PairStatus.ACTIVE
        
        for symbol, pair in self.pairs.items():
            if symbol not in new_active and pair.status not in [PairStatus.BLACKLISTED, PairStatus.DEMOTED]:
                pair.status = PairStatus.WATCHLIST
        
        self.active_pairs = new_active
        self.last_rotation = datetime.now()
        
        return PairRanking(
            rankings=sorted_pairs,
            active_pairs=new_active,
            demoted_pairs=demoted,
            promoted_pairs=promoted
        )
    
    def should_rotate(self) -> bool:
        """Check if rotation evaluation is due."""
        if self.last_rotation is None:
            return True
        
        days_since = (datetime.now() - self.last_rotation).days
        return days_since >= self.rotation_days
    
    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol can be traded.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (can_trade, reason)
        """
        if symbol not in self.pairs:
            return False, f"Unknown pair: {symbol}"
        
        pair = self.pairs[symbol]
        
        if pair.status == PairStatus.BLACKLISTED:
            return False, "Pair is blacklisted"
        
        if pair.status not in [PairStatus.ACTIVE, PairStatus.PROMOTED]:
            return False, f"Pair not active (status: {pair.status.value})"
        
        return True, "OK"
    
    def get_position_weight(self, symbol: str) -> float:
        """
        Get position sizing weight for a pair.
        
        Higher-ranked pairs get larger allocations.
        """
        if symbol not in self.pairs:
            return 0.5
        
        pair = self.pairs[symbol]
        total_pairs = len(self.active_pairs)
        
        if total_pairs == 0:
            return 1.0
        
        # Weight based on rank within active pairs
        if symbol in self.active_pairs:
            rank_within_active = self.active_pairs.index(symbol) + 1
            weight = 1.0 - (rank_within_active - 1) * 0.1
            return max(0.5, min(1.5, weight))
        
        return 0.5
    
    def blacklist_pair(self, symbol: str, reason: str = ""):
        """Blacklist a pair from trading."""
        if symbol in self.pairs:
            self.pairs[symbol].status = PairStatus.BLACKLISTED
            if symbol in self.active_pairs:
                self.active_pairs.remove(symbol)
            logger.warning(f"Blacklisted pair {symbol}: {reason}")
    
    def _calculate_volatility_quality(
        self,
        candles: pd.DataFrame,
        avg_spread_pct: float
    ) -> float:
        """
        Calculate volatility quality score.
        
        Good volatility = high ATR relative to spread, clean trends.
        """
        if candles.empty or len(candles) < 20:
            return 0.5
        
        # Calculate ATR
        high = candles['high']
        low = candles['low']
        close = candles['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # ATR as percentage of price
        atr_pct = atr / close.iloc[-1]
        
        # Good: ATR is at least 5x the spread
        spread_ratio = atr_pct / (avg_spread_pct + 0.0001)
        volatility_score = min(1.0, spread_ratio / 10)
        
        # Trend quality: measure how often price moves in consistent direction
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            # Rolling consistency (autocorrelation)
            consistency = returns.rolling(5).apply(
                lambda x: abs(x.sum()) / x.abs().sum() if x.abs().sum() > 0 else 0
            ).mean()
            trend_score = min(1.0, consistency * 2)
        else:
            trend_score = 0.5
        
        return (volatility_score * 0.6 + trend_score * 0.4)
    
    def _calculate_liquidity_score(
        self,
        candles: pd.DataFrame,
        avg_volume_24h: float
    ) -> float:
        """
        Calculate liquidity score.
        
        Based on volume relative to peers.
        """
        if candles.empty:
            return 0.5
        
        # Use candle volume as proxy
        avg_volume = candles['volume'].rolling(20).mean().iloc[-1]
        
        # Normalize (assuming $100M 24h volume = 1.0 score)
        target_volume = 100_000_000
        
        if avg_volume_24h > 0:
            volume_score = min(1.0, avg_volume_24h / target_volume)
        else:
            # Fallback to candle volume
            volume_score = min(1.0, avg_volume / 1000)
        
        return volume_score
    
    def _calculate_edge_stability(self, symbol: str) -> float:
        """
        Calculate edge stability from backtest history.
        
        Consistent positive expectancy = high stability.
        """
        history = self.backtest_history.get(symbol, [])
        
        if len(history) < 3:
            return 0.5
        
        # Calculate coefficient of variation (lower = more stable)
        mean_exp = np.mean(history)
        std_exp = np.std(history)
        
        if mean_exp <= 0:
            return 0.2  # Low score for negative expectancy
        
        cv = std_exp / mean_exp if mean_exp > 0 else 1.0
        
        # Invert CV for score (low CV = high stability)
        stability = 1.0 / (1.0 + cv)
        
        # Bonus for consistently positive
        positive_pct = sum(1 for e in history if e > 0) / len(history)
        if positive_pct > 0.8:
            stability *= 1.2
        
        return min(1.0, stability)
    
    def _calculate_recent_ces(self, symbol: str) -> float:
        """Calculate recent capital efficiency for this pair."""
        trades = self.trade_history.get(symbol, [])
        
        if len(trades) < 5:
            return 0.5
        
        # Simple CES approximation
        wins = sum(1 for t in trades if t.get('net_pnl', 0) > 0)
        total_pnl = sum(t.get('net_pnl', 0) for t in trades)
        total_capital = sum(t.get('capital_at_risk', 1000) for t in trades)
        
        win_rate = wins / len(trades)
        return_rate = total_pnl / total_capital if total_capital > 0 else 0
        
        # Combine win rate and return rate
        ces = (win_rate * 0.4 + min(1.0, return_rate + 0.5) * 0.6)
        
        return min(1.0, max(0.0, ces))
    
    def get_summary(self) -> Dict:
        """Get summary of pair rankings."""
        return {
            'total_pairs': len(self.pairs),
            'active_pairs': self.active_pairs,
            'last_rotation': self.last_rotation.isoformat() if self.last_rotation else None,
            'should_rotate': self.should_rotate(),
            'rankings': [p.to_dict() for p in sorted(
                self.pairs.values(),
                key=lambda p: p.rank
            )]
        }
