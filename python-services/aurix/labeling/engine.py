"""
AURIX PnL-Aware Labeling Engine

Creates training labels based on NET return after transaction costs.
Implements Audit Fix #1 (PnL-aware labeling) and #7 (contamination prevention).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LabelType(Enum):
    """Label classification."""
    WIN = 1
    LOSS = 0
    MARGINAL = -1  # Too close to transaction costs


@dataclass
class Label:
    """Training label with full context."""
    candle_time: int
    direction: str  # LONG or SHORT
    label: LabelType
    holding_period_minutes: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    transaction_cost: float
    is_contaminated: bool = False
    regime: str = None


class TransactionCostModel:
    """
    Models realistic transaction costs.
    
    Includes:
    - Trading fees (maker/taker)
    - Slippage based on position size and volatility
    """
    
    def __init__(
        self,
        fee_rate_bps: float = 4,  # 0.04% = 4 bps
        base_slippage_bps: float = 5,  # 0.05% = 5 bps
        slippage_volatility_factor: float = 0.5
    ):
        """
        Initialize cost model.
        
        Args:
            fee_rate_bps: Trading fee in basis points
            base_slippage_bps: Base slippage in basis points
            slippage_volatility_factor: Multiplier for volatility adjustment
        """
        self.fee_rate = fee_rate_bps / 10000
        self.base_slippage = base_slippage_bps / 10000
        self.vol_factor = slippage_volatility_factor
    
    def estimate_cost(
        self,
        entry_price: float,
        position_size_usd: float = 1000,
        atr_pct: float = 0.01,
        is_round_trip: bool = True
    ) -> float:
        """
        Estimate total transaction cost.
        
        Args:
            entry_price: Entry price
            position_size_usd: Position size in USD
            atr_pct: ATR as percentage of price
            is_round_trip: Whether to include exit costs
            
        Returns:
            Total cost as a fraction of position
        """
        # Fees (entry + exit if round trip)
        fee_cost = self.fee_rate * (2 if is_round_trip else 1)
        
        # Slippage (increases with volatility)
        slippage = self.base_slippage * (1 + self.vol_factor * (atr_pct / 0.01))
        slippage_cost = slippage * (2 if is_round_trip else 1)
        
        total_cost = fee_cost + slippage_cost
        
        return total_cost


class ContaminationDetector:
    """
    Detects labels that may be contaminated by our own trading activity.
    
    Implements Audit Fix #7: Learning contamination prevention.
    """
    
    def __init__(self, our_trade_times: List[int] = None):
        """
        Initialize detector.
        
        Args:
            our_trade_times: List of timestamps when we executed trades
        """
        self.our_trade_times = our_trade_times or []
    
    def add_trade_time(self, timestamp: int):
        """Record when we executed a trade."""
        self.our_trade_times.append(timestamp)
    
    def is_contaminated(
        self,
        candle_time: int,
        holding_period_minutes: int,
        buffer_minutes: int = 5
    ) -> bool:
        """
        Check if a label period was affected by our own trades.
        
        Args:
            candle_time: Start time of the label period
            holding_period_minutes: How long the position would be held
            buffer_minutes: Additional buffer for market impact
            
        Returns:
            True if the period overlaps with our trading activity
        """
        if not self.our_trade_times:
            return False
        
        # Label window
        start_time = candle_time
        end_time = candle_time + (holding_period_minutes * 60 * 1000)
        buffer_ms = buffer_minutes * 60 * 1000
        
        # Check if any of our trades fall within this window
        for trade_time in self.our_trade_times:
            if (start_time - buffer_ms) <= trade_time <= (end_time + buffer_ms):
                return True
        
        return False


class LabelingEngine:
    """
    PnL-aware labeling engine for ML training.
    
    Creates labels based on NET return (after costs), not just directional accuracy.
    """
    
    def __init__(
        self,
        holding_periods_minutes: List[int] = [15, 60, 240],
        fee_rate_bps: float = 4,
        slippage_bps: float = 5,
        min_return_for_label: float = 0.002,  # 0.2%
        exclude_marginal: bool = True
    ):
        """
        Initialize labeling engine.
        
        Args:
            holding_periods_minutes: List of holding periods to label
            fee_rate_bps: Trading fee in basis points
            slippage_bps: Expected slippage in basis points
            min_return_for_label: Minimum net return to be considered a valid label
            exclude_marginal: Whether to exclude labels within transaction cost noise
        """
        self.holding_periods = holding_periods_minutes
        self.cost_model = TransactionCostModel(fee_rate_bps, slippage_bps)
        self.min_return = min_return_for_label
        self.exclude_marginal = exclude_marginal
        self.contamination_detector = ContaminationDetector()
    
    def compute_labels(
        self,
        df: pd.DataFrame,
        direction: str = "LONG"
    ) -> List[Label]:
        """
        Compute labels for all candles and holding periods.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
            direction: LONG or SHORT
            
        Returns:
            List of Label objects
        """
        labels = []
        
        for period in self.holding_periods:
            period_labels = self._compute_period_labels(df, direction, period)
            labels.extend(period_labels)
        
        return labels
    
    def _compute_period_labels(
        self,
        df: pd.DataFrame,
        direction: str,
        holding_period: int
    ) -> List[Label]:
        """Compute labels for a specific holding period."""
        labels = []
        
        # Calculate lookforward periods
        periods_needed = holding_period // 15  # Assuming 15m candles
        
        if len(df) <= periods_needed:
            return labels
        
        # Calculate ATR for slippage estimation
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / df['close']
        
        for i in range(len(df) - periods_needed):
            try:
                candle_time = int(df.index[i].timestamp() * 1000)
                entry_price = df['close'].iloc[i]
                exit_price = df['close'].iloc[i + periods_needed]
                
                # Calculate gross return
                if direction == "LONG":
                    gross_return = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    gross_return = (entry_price - exit_price) / entry_price
                
                # Estimate transaction cost
                current_atr_pct = atr_pct.iloc[i] if not pd.isna(atr_pct.iloc[i]) else 0.01
                transaction_cost = self.cost_model.estimate_cost(
                    entry_price=entry_price,
                    atr_pct=current_atr_pct
                )
                
                # Net return
                net_return = gross_return - transaction_cost
                
                # Determine label
                if net_return > self.min_return:
                    label_type = LabelType.WIN
                elif net_return < -self.min_return:
                    label_type = LabelType.LOSS
                else:
                    label_type = LabelType.MARGINAL
                
                # Skip marginal labels if configured
                if self.exclude_marginal and label_type == LabelType.MARGINAL:
                    continue
                
                # Check contamination
                is_contaminated = self.contamination_detector.is_contaminated(
                    candle_time, holding_period
                )
                
                labels.append(Label(
                    candle_time=candle_time,
                    direction=direction,
                    label=label_type,
                    holding_period_minutes=holding_period,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_return=gross_return,
                    net_return=net_return,
                    transaction_cost=transaction_cost,
                    is_contaminated=is_contaminated
                ))
                
            except Exception as e:
                logger.warning(f"Error computing label at index {i}: {e}")
                continue
        
        return labels
    
    def labels_to_dataframe(self, labels: List[Label]) -> pd.DataFrame:
        """Convert labels to DataFrame for training."""
        if not labels:
            return pd.DataFrame()
        
        data = []
        for label in labels:
            data.append({
                'candle_time': label.candle_time,
                'direction': label.direction,
                'label': label.label.value if label.label != LabelType.MARGINAL else -1,
                'holding_period': label.holding_period_minutes,
                'entry_price': label.entry_price,
                'exit_price': label.exit_price,
                'gross_return': label.gross_return,
                'net_return': label.net_return,
                'transaction_cost': label.transaction_cost,
                'is_contaminated': label.is_contaminated,
                'regime': label.regime
            })
        
        return pd.DataFrame(data)
    
    def get_training_labels(
        self,
        labels: List[Label],
        exclude_contaminated: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get labels suitable for training (X indices and y labels).
        
        Args:
            labels: List of Label objects
            exclude_contaminated: Whether to exclude contaminated labels
            
        Returns:
            Tuple of (candle_times, labels)
        """
        filtered = []
        for label in labels:
            if exclude_contaminated and label.is_contaminated:
                continue
            if label.label == LabelType.MARGINAL:
                continue
            filtered.append(label)
        
        if not filtered:
            return np.array([]), np.array([])
        
        candle_times = np.array([l.candle_time for l in filtered])
        y = np.array([1 if l.label == LabelType.WIN else 0 for l in filtered])
        
        return candle_times, y
    
    def record_our_trade(self, timestamp: int):
        """Record when we executed a trade for contamination detection."""
        self.contamination_detector.add_trade_time(timestamp)
