"""
AURIX Slippage Model

Realistic slippage estimation based on volatility, order size, and market conditions.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlippageResult:
    """Result of slippage estimation."""
    estimated_slippage_pct: float  # Slippage as percentage
    spread_cost_pct: float         # Spread cost as percentage
    total_execution_cost_pct: float  # Total execution cost
    execution_price: float         # Estimated execution price after slippage
    adverse_fill: bool             # True if execution is adverse
    
    def __str__(self) -> str:
        return (
            f"Slippage: {self.estimated_slippage_pct:.4%}, "
            f"Spread: {self.spread_cost_pct:.4%}, "
            f"Total: {self.total_execution_cost_pct:.4%}"
        )


class SlippageModel:
    """
    Realistic Slippage Model
    
    Estimates execution costs based on:
    1. Volatility (higher vol = more slippage)
    2. Order size (larger orders = more market impact)
    3. Spread (bid-ask spread cost)
    4. Time of day (optional, lower liquidity at certain times)
    
    Designed to be PESSIMISTIC - overestimate costs to be safe.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.02,  # 2 bps base
        default_spread_pct: float = 0.01,  # 1 bp spread
        volatility_multiplier: float = 2.0,
        size_impact_factor: float = 0.1,
        high_vol_threshold: float = 0.75,  # 75th percentile
        high_vol_slippage_multiplier: float = 3.0
    ):
        """
        Initialize slippage model.
        
        Args:
            base_slippage_pct: Base slippage for normal conditions
            default_spread_pct: Default bid-ask spread
            volatility_multiplier: How much volatility affects slippage
            size_impact_factor: Impact of order size on slippage
            high_vol_threshold: Volatility percentile threshold for high vol mode
            high_vol_slippage_multiplier: Multiplier for high volatility
        """
        self.base_slippage_pct = base_slippage_pct
        self.default_spread_pct = default_spread_pct
        self.volatility_multiplier = volatility_multiplier
        self.size_impact_factor = size_impact_factor
        self.high_vol_threshold = high_vol_threshold
        self.high_vol_slippage_multiplier = high_vol_slippage_multiplier
    
    def estimate_slippage(
        self,
        price: float,
        quantity: float,
        volatility_pct: float,
        side: str,
        spread_pct: Optional[float] = None,
        volatility_percentile: Optional[float] = None
    ) -> SlippageResult:
        """
        Estimate slippage for a trade.
        
        Args:
            price: Current market price
            quantity: Order quantity in base currency
            volatility_pct: Current volatility as percentage (e.g., ATR/price)
            side: 'BUY' or 'SELL'
            spread_pct: Optional explicit spread, else uses default
            volatility_percentile: Optional volatility percentile (0-1)
            
        Returns:
            SlippageResult with estimated costs and execution price
        """
        spread_pct = spread_pct if spread_pct is not None else self.default_spread_pct
        
        # Base slippage
        slippage = self.base_slippage_pct
        
        # Volatility component
        vol_component = volatility_pct * self.volatility_multiplier
        slippage += vol_component
        
        # High volatility multiplier
        if volatility_percentile is not None and volatility_percentile > self.high_vol_threshold:
            excess_percentile = (volatility_percentile - self.high_vol_threshold) / (1 - self.high_vol_threshold)
            high_vol_mult = 1 + (self.high_vol_slippage_multiplier - 1) * excess_percentile
            slippage *= high_vol_mult
            logger.debug(f"High volatility detected ({volatility_percentile:.0%}), multiplier: {high_vol_mult:.2f}")
        
        # Size impact (larger orders = more slippage due to market impact)
        order_value = price * quantity
        size_impact = self.size_impact_factor * np.log1p(order_value / 10000) * 0.0001
        slippage += size_impact
        
        # Spread cost (half spread for each direction)
        spread_cost = spread_pct / 2
        
        # Total execution cost
        total_cost = slippage + spread_cost
        
        # Calculate execution price
        if side.upper() == 'BUY':
            # Buying: pay more
            execution_price = price * (1 + total_cost / 100)
            adverse = total_cost > self.base_slippage_pct + self.default_spread_pct / 2
        else:
            # Selling: receive less
            execution_price = price * (1 - total_cost / 100)
            adverse = total_cost > self.base_slippage_pct + self.default_spread_pct / 2
        
        return SlippageResult(
            estimated_slippage_pct=slippage,
            spread_cost_pct=spread_cost,
            total_execution_cost_pct=total_cost,
            execution_price=execution_price,
            adverse_fill=adverse
        )
    
    def get_effective_spread(self, volatility_percentile: float) -> float:
        """
        Get effective spread based on volatility conditions.
        
        Spread widens during high volatility.
        
        Args:
            volatility_percentile: Current volatility percentile (0-1)
            
        Returns:
            Effective spread as percentage
        """
        if volatility_percentile > self.high_vol_threshold:
            # Spread widens during high volatility
            excess = (volatility_percentile - self.high_vol_threshold) / (1 - self.high_vol_threshold)
            multiplier = 1 + 2 * excess  # Up to 3x spread at max vol
            return self.default_spread_pct * multiplier
        
        return self.default_spread_pct
    
    def calculate_round_trip_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_volatility_pct: float,
        exit_volatility_pct: float
    ) -> float:
        """
        Calculate total round-trip execution cost.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            entry_volatility_pct: Volatility at entry
            exit_volatility_pct: Volatility at exit
            
        Returns:
            Total cost as dollar amount
        """
        entry_result = self.estimate_slippage(
            entry_price, quantity, entry_volatility_pct, 'BUY'
        )
        exit_result = self.estimate_slippage(
            exit_price, quantity, exit_volatility_pct, 'SELL'
        )
        
        entry_cost = entry_price * quantity * entry_result.total_execution_cost_pct / 100
        exit_cost = exit_price * quantity * exit_result.total_execution_cost_pct / 100
        
        return entry_cost + exit_cost
