"""
AURIX Stress Tester

Injects noise and stress to simulate real market conditions.
Makes backtests more realistic by simulating adverse scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressConfig:
    """Configuration for stress testing."""
    # Wick noise
    enable_wick_noise: bool = True
    wick_noise_pct: float = 0.10  # As percentage of ATR
    
    # Gap injection
    enable_gap_injection: bool = True
    gap_probability: float = 0.005  # 0.5% per candle
    gap_size_atr_mult: float = 0.5  # Gap size as multiple of ATR
    
    # Execution delay
    enable_execution_delay: bool = True
    min_delay_ms: int = 50
    max_delay_ms: int = 500
    
    # Adverse slippage
    enable_adverse_slippage: bool = True
    high_vol_slippage_multiplier: float = 3.0
    vol_threshold_for_adverse: float = 0.75  # 75th percentile
    
    # Spread widening
    enable_spread_widening: bool = True
    max_spread_multiplier: float = 5.0
    
    # Stress intensity (0-1, scales all effects)
    intensity: float = 0.5


class StressTester:
    """
    Stress Tester
    
    Injects realistic market stress into backtesting:
    1. Random candle distortion (wicks, gaps)
    2. Random execution delay
    3. Adverse slippage during high volatility
    4. Spread widening simulation
    
    Makes paper performance more realistic.
    """
    
    def __init__(self, config: Optional[StressConfig] = None):
        """
        Initialize stress tester.
        
        Args:
            config: Stress testing configuration
        """
        self.config = config or StressConfig()
        self._random = random.Random(42)  # Reproducible randomness
        self._applied_stresses: Dict[str, int] = {
            'wick_noise': 0,
            'gap_injection': 0,
            'execution_delay': 0,
            'adverse_slippage': 0,
            'spread_widening': 0
        }
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._random = random.Random(seed)
        np.random.seed(seed)
    
    def distort_candle(
        self,
        candle: Dict,
        atr: float,
        volatility_percentile: float = 0.5
    ) -> Dict:
        """
        Apply stress distortions to a candle.
        
        Args:
            candle: Dict with 'open', 'high', 'low', 'close', 'volume'
            atr: Current ATR for scaling
            volatility_percentile: Current volatility percentile (0-1)
            
        Returns:
            Distorted candle dict
        """
        result = candle.copy()
        intensity = self.config.intensity
        
        # Wick noise
        if self.config.enable_wick_noise:
            result = self._apply_wick_noise(result, atr, intensity)
        
        # Gap injection
        if self.config.enable_gap_injection:
            result = self._apply_gap(result, atr, intensity)
        
        return result
    
    def _apply_wick_noise(self, candle: Dict, atr: float, intensity: float) -> Dict:
        """Apply random wick extensions/retractions."""
        noise_scale = atr * self.config.wick_noise_pct * intensity
        
        # Random wick extension
        if self._random.random() < 0.3:  # 30% chance of wick noise
            high_noise = self._random.gauss(0, noise_scale)
            low_noise = self._random.gauss(0, noise_scale)
            
            candle['high'] = candle['high'] + abs(high_noise)
            candle['low'] = candle['low'] - abs(low_noise)
            
            self._applied_stresses['wick_noise'] += 1
            logger.debug(f"Applied wick noise: high+{abs(high_noise):.4f}, low-{abs(low_noise):.4f}")
        
        return candle
    
    def _apply_gap(self, candle: Dict, atr: float, intensity: float) -> Dict:
        """Inject price gaps with low probability."""
        gap_prob = self.config.gap_probability * intensity
        
        if self._random.random() < gap_prob:
            gap_size = atr * self.config.gap_size_atr_mult * intensity
            gap_direction = self._random.choice([-1, 1])
            
            gap = gap_size * gap_direction
            
            # Apply gap to all prices
            candle['open'] += gap
            candle['high'] += gap
            candle['low'] += gap
            candle['close'] += gap
            
            self._applied_stresses['gap_injection'] += 1
            logger.debug(f"Injected gap: {gap:.4f} ({'up' if gap > 0 else 'down'})")
        
        return candle
    
    def simulate_execution_delay(self) -> float:
        """
        Simulate random execution delay.
        
        Returns:
            Delay in milliseconds
        """
        if not self.config.enable_execution_delay:
            return 0.0
        
        min_delay = self.config.min_delay_ms
        max_delay = self.config.max_delay_ms
        intensity = self.config.intensity
        
        # Scale delay range by intensity
        effective_max = min_delay + (max_delay - min_delay) * intensity
        
        delay = self._random.uniform(min_delay, effective_max)
        
        self._applied_stresses['execution_delay'] += 1
        
        return delay
    
    def calculate_stress_slippage(
        self,
        base_slippage_pct: float,
        volatility_percentile: float
    ) -> float:
        """
        Calculate stress-adjusted slippage.
        
        Args:
            base_slippage_pct: Base slippage percentage
            volatility_percentile: Current volatility percentile (0-1)
            
        Returns:
            Stress-adjusted slippage percentage
        """
        if not self.config.enable_adverse_slippage:
            return base_slippage_pct
        
        intensity = self.config.intensity
        threshold = self.config.vol_threshold_for_adverse
        multiplier = self.config.high_vol_slippage_multiplier
        
        if volatility_percentile > threshold:
            # Calculate excess volatility
            excess = (volatility_percentile - threshold) / (1 - threshold)
            
            # Scale multiplier by excess and intensity
            effective_multiplier = 1 + (multiplier - 1) * excess * intensity
            
            adjusted = base_slippage_pct * effective_multiplier
            
            self._applied_stresses['adverse_slippage'] += 1
            logger.debug(f"Adverse slippage: {base_slippage_pct:.4%} -> {adjusted:.4%} (vol={volatility_percentile:.0%})")
            
            return adjusted
        
        return base_slippage_pct
    
    def get_stress_spread(
        self,
        base_spread_pct: float,
        volatility_percentile: float
    ) -> float:
        """
        Get stress-widened spread.
        
        Args:
            base_spread_pct: Base spread percentage
            volatility_percentile: Current volatility percentile (0-1)
            
        Returns:
            Stress-adjusted spread percentage
        """
        if not self.config.enable_spread_widening:
            return base_spread_pct
        
        intensity = self.config.intensity
        max_mult = self.config.max_spread_multiplier
        
        # Spread widens with volatility and intensity
        if volatility_percentile > 0.5:
            excess = (volatility_percentile - 0.5) / 0.5
            multiplier = 1 + (max_mult - 1) * excess * intensity
            
            widened = base_spread_pct * multiplier
            
            self._applied_stresses['spread_widening'] += 1
            
            return widened
        
        return base_spread_pct
    
    def inject_flash_crash(
        self,
        candle: Dict,
        atr: float,
        crash_magnitude: float = 3.0
    ) -> Tuple[Dict, bool]:
        """
        Inject a flash crash scenario.
        
        Args:
            candle: Original candle
            atr: Current ATR
            crash_magnitude: Crash size as multiple of ATR
            
        Returns:
            Tuple of (distorted candle, was_crashed)
        """
        # Very low probability
        if self._random.random() > 0.0005:  # 0.05% chance
            return candle, False
        
        result = candle.copy()
        crash_size = atr * crash_magnitude * self.config.intensity
        
        # Flash crash pattern: price drops suddenly then partially recovers
        result['low'] = result['low'] - crash_size
        result['close'] = result['close'] - crash_size * 0.3  # Partial recovery
        
        logger.warning(f"[STRESS] Flash crash injected: -{crash_size:.4f}")
        
        return result, True
    
    def get_stress_stats(self) -> Dict[str, int]:
        """Get count of applied stresses."""
        return self._applied_stresses.copy()
    
    def reset_stats(self):
        """Reset stress statistics."""
        for key in self._applied_stresses:
            self._applied_stresses[key] = 0
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        lines = [
            "=== Stress Tester Status ===",
            f"Intensity: {self.config.intensity:.0%}",
            f"Wick Noise: {'ON' if self.config.enable_wick_noise else 'OFF'}",
            f"Gap Injection: {'ON' if self.config.enable_gap_injection else 'OFF'}",
            f"Execution Delay: {'ON' if self.config.enable_execution_delay else 'OFF'}",
            f"Adverse Slippage: {'ON' if self.config.enable_adverse_slippage else 'OFF'}",
            f"Spread Widening: {'ON' if self.config.enable_spread_widening else 'OFF'}",
            "",
            "Applied Stresses:",
        ]
        
        for name, count in self._applied_stresses.items():
            lines.append(f"  {name}: {count}")
        
        return "\n".join(lines)
