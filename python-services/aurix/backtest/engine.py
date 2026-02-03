"""
AURIX Walk-Forward Backtest Engine

Event-driven backtesting that mirrors live trading:
1. Candle-by-candle simulation (1m base → 15m aggregated)
2. At each 15m close: features → predict → wait → label → optionally retrain
3. Supports regime detection and dynamic thresholds
4. Compares online learning vs static model performance

Author: AURIX Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import logging

from ..features import FeatureEngine
from ..labeling import LabelingEngine, Label, LabelType
from ..regime import RegimeDetector, MarketRegime
from ..ml import MLTrainer, TrainingConfig, ModelWrapper
from ..reality import (
    RealityConfig,
    KillSwitch,
    OverfitMonitor,
    StressTester,
    StressConfig,
    SlippageModel,
)

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Model learning strategy during backtest."""
    STATIC = "static"           # Train once, never retrain
    PERIODIC = "periodic"       # Retrain at fixed intervals
    ADAPTIVE = "adaptive"       # Retrain when performance degrades
    CONTINUOUS = "continuous"   # Retrain after every N trades


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtest."""
    # Time settings
    start_date: datetime = None
    end_date: datetime = None
    
    # Initial capital
    initial_capital: float = 10000.0
    
    # Risk settings
    risk_per_trade_percent: float = 1.0
    max_position_size_percent: float = 5.0
    default_tp_percent: float = 1.0
    default_sl_percent: float = 0.5
    
    # Transaction costs
    commission_rate: float = 0.001
    base_slippage: float = 0.0005
    
    # ML settings
    learning_mode: LearningMode = LearningMode.PERIODIC
    initial_train_days: int = 14
    retrain_interval_hours: int = 24
    min_samples_for_train: int = 200
    sliding_window_days: int = 30
    
    # Confidence thresholds
    base_confidence_threshold: float = 0.60
    regime_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "TRENDING_UP": 0.0,
        "TRENDING_DOWN": 0.0,
        "RANGING": 0.08,
        "VOLATILE": 0.12,
        "UNKNOWN": 0.999
    })


@dataclass
class Trade:
    """A single simulated trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    tp_price: float
    sl_price: float
    confidence: float
    regime: str
    
    # Outcomes
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    outcome: Optional[str] = None  # "WIN", "LOSS", "OPEN"
    
    # Model info
    model_version: str = ""
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0


@dataclass
class BacktestState:
    """Current state during backtesting."""
    equity: float
    cash: float
    peak_equity: float
    current_drawdown: float
    max_drawdown: float
    open_trades: List[Trade]
    closed_trades: List[Trade]
    daily_pnl: float
    daily_start_equity: float
    consecutive_losses: int
    is_trading_enabled: bool
    current_date: datetime = None


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    expectancy_ratio: float  # Expectancy / avg_loss
    
    # Sharpe-like ratio (simplified)
    sharpe_ratio: float
    sortino_ratio: float
    
    # Time metrics
    avg_trade_duration: float  # minutes
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # By regime
    regime_performance: Dict[str, Dict]
    
    # Confidence analysis
    confidence_bucket_accuracy: Dict[str, float]
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]]
    drawdown_curve: List[Tuple[datetime, float]]


class WalkForwardBacktester:
    """
    Walk-Forward Backtesting Engine
    
    Simulates AURIX trading logic on historical data:
    1. Processes candles chronologically
    2. Computes features at each 15m bar
    3. Generates predictions with calibrated probabilities
    4. Simulates trade execution with realistic costs
    5. Labels outcomes using net PnL
    6. Optionally retrains model using sliding window
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.labeling_engine = LabelingEngine(
            fee_rate_bps=config.commission_rate * 10000,  # Convert to bps
            slippage_bps=config.base_slippage * 10000     # Convert to bps
        )
        self.model_trainer = MLTrainer(TrainingConfig(
            min_samples=config.min_samples_for_train,
            training_window_days=config.sliding_window_days
        ))
        
        # State
        self.state: Optional[BacktestState] = None
        self.model: Optional[ModelWrapper] = None
        self.model_static: Optional[ModelWrapper] = None  # For comparison
        
        # Data storage
        self.features_history: List[Dict] = []
        self.labels_history: List[Dict] = []
        self.predictions_history: List[Dict] = []
        
        # Metrics tracking
        self.equity_snapshots: List[Tuple[datetime, float]] = []
        self.drawdown_snapshots: List[Tuple[datetime, float]] = []
        
        # Reality Validation Layer
        self.reality_config = RealityConfig()
        self.kill_switch = KillSwitch(
            max_drawdown_pct=self.reality_config.max_drawdown_pct,
            max_consecutive_losses=self.reality_config.max_consecutive_losses,
            min_avg_confidence=self.reality_config.min_avg_confidence
        )
        self.overfit_monitor = OverfitMonitor(
            max_train_forward_divergence=self.reality_config.max_train_forward_divergence,
            auc_collapse_threshold=self.reality_config.auc_collapse_threshold
        )
        self.stress_tester = StressTester(StressConfig(
            intensity=self.reality_config.stress_intensity if self.reality_config.enable_stress_testing else 0.0
        ))
        self.slippage_model = SlippageModel(
            base_slippage_pct=self.reality_config.base_slippage_pct,
            default_spread_pct=self.reality_config.default_spread_pct
        )
        self.enable_reality_layer = True  # Can be disabled for comparison
        
        # Retraining tracking
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_count = 0
    
    def run(
        self,
        candles_1m: pd.DataFrame,
        candles_15m: pd.DataFrame,
        candles_1h: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
        compare_static: bool = True
    ) -> Tuple[BacktestMetrics, Optional[BacktestMetrics]]:
        """
        Run walk-forward backtest.
        
        Args:
            candles_1m: 1-minute candles (index=datetime)
            candles_15m: 15-minute candles
            candles_1h: 1-hour candles (optional, for HTF)
            symbol: Trading pair
            compare_static: Also run with static model for comparison
            
        Returns:
            Tuple of (online_metrics, static_metrics)
        """
        logger.info(f"Starting walk-forward backtest from {candles_15m.index[0]} to {candles_15m.index[-1]}")
        
        # Initialize state
        self._initialize_state()
        
        # Determine warm-up period
        warmup_end_idx = self.config.initial_train_days * 96  # 96 15m candles per day
        if warmup_end_idx >= len(candles_15m):
            raise ValueError("Not enough data for initial training period")
        
        # Phase 1: Generate features and labels for warm-up period (no trading)
        logger.info(f"Warm-up phase: generating {warmup_end_idx} feature/label pairs...")
        for i in range(50, warmup_end_idx):  # Start at 50 to have enough history
            candle_time = candles_15m.index[i]
            self._compute_and_store_features(
                candles_15m.iloc[:i+1],
                candles_1h[:candle_time] if candles_1h is not None else None,
                symbol
            )
            
            # Generate label for previous candle
            if i > 50:
                self._generate_label(
                    candles_15m.iloc[:i+1],
                    symbol,
                    self.features_history[-2] if len(self.features_history) > 1 else None
                )
        
        # Phase 2: Initial model training
        logger.info("Training initial model...")
        self._train_model()
        
        if compare_static:
            # Save static model for comparison
            self.model_static = self.model
        
        # Phase 3: Walk-forward simulation
        logger.info("Starting walk-forward trading simulation...")
        for i in range(warmup_end_idx, len(candles_15m)):
            candle_time = candles_15m.index[i]
            self.state.current_date = candle_time
            
            # Check for day change (reset daily stats)
            self._check_daily_reset(candle_time)
            
            # Update open trades with current price
            current_price = candles_15m['close'].iloc[i]
            high = candles_15m['high'].iloc[i]
            low = candles_15m['low'].iloc[i]
            self._update_open_trades(high, low, current_price, candle_time)
            
            # Record equity
            current_equity = self._calculate_equity(current_price)
            self.equity_snapshots.append((candle_time, current_equity))
            self._update_drawdown(current_equity)
            
            # Compute features
            features = self._compute_and_store_features(
                candles_15m.iloc[:i+1],
                candles_1h[:candle_time] if candles_1h is not None else None,
                symbol
            )
            
            if features is None:
                continue
            
            # Detect regime
            regime_state = self.regime_detector.detect_regime(
                candles_15m.iloc[max(0, i-100):i+1]
            )
            
            # Generate prediction and potentially trade
            if self.model is not None and self.state.is_trading_enabled:
                self._generate_prediction_and_trade(
                    features, regime_state, current_price, candle_time, symbol
                )
            
            # Generate label for candle that's now complete
            if len(self.features_history) > 15:  # Need horizon data
                self._generate_label(
                    candles_15m.iloc[:i+1],
                    symbol,
                    self.features_history[-16] if len(self.features_history) > 15 else None
                )
            
            # Check if retraining is needed
            self._check_retrain(candle_time)
            
            # Progress logging
            if i % 1000 == 0:
                pct = (i - warmup_end_idx) / (len(candles_15m) - warmup_end_idx) * 100
                logger.info(f"Progress: {pct:.1f}% - Equity: ${current_equity:.2f} - Trades: {len(self.state.closed_trades)}")
        
        # Close any remaining open trades
        final_price = candles_15m['close'].iloc[-1]
        self._close_all_trades(final_price, candles_15m.index[-1])
        
        # Calculate metrics
        online_metrics = self._calculate_metrics()
        
        # Run static comparison if requested
        static_metrics = None
        if compare_static and self.model_static is not None:
            static_metrics = self._run_static_comparison(
                candles_15m, candles_1h, symbol, warmup_end_idx
            )
        
        return online_metrics, static_metrics
    
    def _initialize_state(self):
        """Initialize backtest state."""
        self.state = BacktestState(
            equity=self.config.initial_capital,
            cash=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
            current_drawdown=0.0,
            max_drawdown=0.0,
            open_trades=[],
            closed_trades=[],
            daily_pnl=0.0,
            daily_start_equity=self.config.initial_capital,
            consecutive_losses=0,
            is_trading_enabled=True
        )
        
        self.features_history = []
        self.labels_history = []
        self.predictions_history = []
        self.equity_snapshots = []
        self.drawdown_snapshots = []
        self.model = None
        self.last_retrain_time = None
        self.retrain_count = 0
    
    def _compute_and_store_features(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: Optional[pd.DataFrame],
        symbol: str
    ) -> Optional[Dict]:
        """Compute and store features for current candle."""
        features_df = self.feature_engine.compute_features(
            df_15m=candles_15m,
            df_1h=candles_1h
        )
        
        if features_df is None or len(features_df) == 0:
            return None
        
        # Get the latest row as a dictionary
        latest_features = features_df.iloc[-1].to_dict()
        
        feature_record = {
            'time': candles_15m.index[-1],
            'symbol': symbol,
            'features': latest_features
        }
        self.features_history.append(feature_record)
        
        return feature_record
    
    def _generate_label(
        self,
        candles: pd.DataFrame,
        symbol: str,
        feature_record: Optional[Dict]
    ):
        """Generate label for a past prediction using realized data."""
        if feature_record is None:
            return
        
        entry_time = feature_record['time']
        try:
            entry_idx = candles.index.get_loc(entry_time)
        except KeyError:
            return
        
        # Get candles for horizon evaluation
        horizon_candles = candles.iloc[entry_idx:entry_idx + 16]  # 15 candles ahead
        if len(horizon_candles) < 2:
            return
        
        entry_price = candles['close'].iloc[entry_idx]
        
        # Calculate TP/SL for labeling
        tp_pct = self.config.default_tp_percent / 100
        sl_pct = self.config.default_sl_percent / 100
        
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        # Evaluate outcome by checking if TP or SL was hit first
        outcome = 'NEUTRAL'
        net_return = 0.0
        gross_return = 0.0
        
        for i in range(1, len(horizon_candles)):
            high = horizon_candles['high'].iloc[i]
            low = horizon_candles['low'].iloc[i]
            close = horizon_candles['close'].iloc[i]
            
            # Check SL first (more conservative)
            if low <= sl_price:
                outcome = 'LOSS'
                gross_return = -sl_pct
                break
            elif high >= tp_price:
                outcome = 'WIN'
                gross_return = tp_pct
                break
        
        # If neither TP nor SL hit, use final close
        if outcome == 'NEUTRAL':
            final_close = horizon_candles['close'].iloc[-1]
            gross_return = (final_close - entry_price) / entry_price
            outcome = 'WIN' if gross_return > 0 else 'LOSS'
        
        # Apply transaction costs
        tx_cost = self.config.commission_rate * 2 + self.config.base_slippage * 2
        net_return = gross_return - tx_cost
        
        label_record = {
            'time': entry_time,
            'symbol': symbol,
            'outcome': outcome,
            'net_return': net_return,
            'gross_return': gross_return
        }
        self.labels_history.append(label_record)
    
    def _generate_prediction_and_trade(
        self,
        features: Dict,
        regime_state,
        current_price: float,
        current_time: datetime,
        symbol: str
    ):
        """Generate prediction and open trade if conditions met."""
        # Prepare features DataFrame
        features_df = pd.DataFrame([features['features']])
        
        # Get prediction (defaults to LONG direction, uses trainer's stored model)
        raw_prob, calibrated_prob = self.model_trainer.predict(features_df)
        raw_prob = float(raw_prob[0])
        calibrated_prob = float(calibrated_prob[0])
        
        # Determine direction and confidence
        if calibrated_prob >= 0.5:
            direction = "LONG"
            confidence = calibrated_prob
        else:
            direction = "SHORT"
            confidence = 1 - calibrated_prob
        
        # Calculate dynamic threshold
        regime_adj = self.config.regime_adjustments.get(regime_state.regime.value, 0.1)
        threshold = self.config.base_confidence_threshold + regime_adj
        
        # Store prediction
        self.predictions_history.append({
            'time': current_time,
            'symbol': symbol,
            'direction': direction,
            'raw_probability': raw_prob,
            'calibrated_probability': calibrated_prob,
            'confidence': confidence,
            'threshold': threshold,
            'regime': regime_state.regime.value,
            'signal_issued': confidence >= threshold
        })
        
        # Check if we should trade
        if confidence < threshold:
            return
        
        if len(self.state.open_trades) >= 1:  # Max 1 position for simplicity
            return
        
        # Calculate position size
        risk_amount = self.state.cash * (self.config.risk_per_trade_percent / 100)
        position_value = min(
            risk_amount / (self.config.default_sl_percent / 100),
            self.state.cash * (self.config.max_position_size_percent / 100)
        )
        
        # Apply regime multiplier
        position_value *= regime_state.size_multiplier
        
        if position_value < 10:  # Minimum position
            return
        
        quantity = position_value / current_price
        
        # Calculate TP/SL
        if direction == "LONG":
            tp_price = current_price * (1 + self.config.default_tp_percent / 100)
            sl_price = current_price * (1 - self.config.default_sl_percent / 100)
        else:
            tp_price = current_price * (1 - self.config.default_tp_percent / 100)
            sl_price = current_price * (1 + self.config.default_sl_percent / 100)
        
        # Apply slippage to entry
        slippage = self.config.base_slippage
        if direction == "LONG":
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)
        
        # Calculate entry fee
        entry_fee = position_value * self.config.commission_rate
        
        # Open trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            tp_price=tp_price,
            sl_price=sl_price,
            confidence=confidence,
            regime=regime_state.regime.value,
            fees_paid=entry_fee,
            slippage_cost=abs(entry_price - current_price) * quantity,
            model_version=self.model.version if self.model else "",
            raw_probability=raw_prob,
            calibrated_probability=calibrated_prob,
            outcome="OPEN"
        )
        
        self.state.open_trades.append(trade)
        self.state.cash -= (position_value + entry_fee)
    
    def _update_open_trades(
        self,
        high: float,
        low: float,
        close: float,
        current_time: datetime
    ):
        """Update open trades, check TP/SL."""
        trades_to_close = []
        
        for trade in self.state.open_trades:
            if trade.direction == "LONG":
                # Check SL first (more conservative)
                if low <= trade.sl_price:
                    trade.exit_price = trade.sl_price
                    trade.outcome = "LOSS"
                    trades_to_close.append((trade, current_time))
                elif high >= trade.tp_price:
                    trade.exit_price = trade.tp_price
                    trade.outcome = "WIN"
                    trades_to_close.append((trade, current_time))
            else:  # SHORT
                if high >= trade.sl_price:
                    trade.exit_price = trade.sl_price
                    trade.outcome = "LOSS"
                    trades_to_close.append((trade, current_time))
                elif low <= trade.tp_price:
                    trade.exit_price = trade.tp_price
                    trade.outcome = "WIN"
                    trades_to_close.append((trade, current_time))
        
        for trade, exit_time in trades_to_close:
            self._close_trade(trade, exit_time)
    
    def _close_trade(self, trade: Trade, exit_time: datetime):
        """Close a trade and update state."""
        trade.exit_time = exit_time
        
        # Apply exit slippage
        slippage = self.config.base_slippage
        if trade.direction == "LONG":
            actual_exit = trade.exit_price * (1 - slippage)
        else:
            actual_exit = trade.exit_price * (1 + slippage)
        
        # Calculate PnL
        position_value = trade.quantity * trade.entry_price
        exit_fee = trade.quantity * actual_exit * self.config.commission_rate
        trade.fees_paid += exit_fee
        trade.slippage_cost += abs(trade.exit_price - actual_exit) * trade.quantity
        
        if trade.direction == "LONG":
            trade.gross_pnl = (actual_exit - trade.entry_price) * trade.quantity
        else:
            trade.gross_pnl = (trade.entry_price - actual_exit) * trade.quantity
        
        trade.net_pnl = trade.gross_pnl - trade.fees_paid
        
        # Update state
        self.state.cash += position_value + trade.net_pnl
        self.state.daily_pnl += trade.net_pnl
        
        if trade.net_pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Check risk limits
        if self.state.consecutive_losses >= 5:
            self.state.is_trading_enabled = False
        
        # Move to closed trades
        self.state.open_trades.remove(trade)
        self.state.closed_trades.append(trade)
    
    def _close_all_trades(self, price: float, time: datetime):
        """Close all open trades at market price."""
        for trade in list(self.state.open_trades):
            if trade.direction == "LONG":
                trade.exit_price = price
                trade.outcome = "WIN" if price > trade.entry_price else "LOSS"
            else:
                trade.exit_price = price
                trade.outcome = "WIN" if price < trade.entry_price else "LOSS"
            self._close_trade(trade, time)
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions."""
        equity = self.state.cash
        for trade in self.state.open_trades:
            position_value = trade.quantity * trade.entry_price
            if trade.direction == "LONG":
                unrealized = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized = (trade.entry_price - current_price) * trade.quantity
            equity += position_value + unrealized
        return equity
    
    def _update_drawdown(self, current_equity: float):
        """Update drawdown tracking."""
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        self.state.current_drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        
        if self.state.current_drawdown > self.state.max_drawdown:
            self.state.max_drawdown = self.state.current_drawdown
        
        self.drawdown_snapshots.append((self.state.current_date, self.state.current_drawdown))
        
        # Check max drawdown limit
        if self.state.current_drawdown > 0.05:  # 5% max drawdown
            self.state.is_trading_enabled = False
    
    def _check_daily_reset(self, current_time: datetime):
        """Check for day change and reset daily stats."""
        if self.state.current_date is None:
            self.state.current_date = current_time
            return
        
        if current_time.date() > self.state.current_date.date():
            # New day
            daily_pnl_pct = self.state.daily_pnl / self.state.daily_start_equity
            
            # Check daily loss limit
            if daily_pnl_pct < -0.03:  # 3% daily loss
                self.state.is_trading_enabled = False
            else:
                # Re-enable trading if it was disabled for daily loss
                if self.state.consecutive_losses < 5 and self.state.current_drawdown < 0.05:
                    self.state.is_trading_enabled = True
            
            # Reset daily stats
            current_equity = self.state.cash
            for trade in self.state.open_trades:
                current_equity += trade.quantity * trade.entry_price
            
            self.state.daily_pnl = 0.0
            self.state.daily_start_equity = current_equity
    
    def _check_retrain(self, current_time: datetime):
        """Check if model retraining is needed."""
        if self.config.learning_mode == LearningMode.STATIC:
            return
        
        should_retrain = False
        
        if self.config.learning_mode == LearningMode.PERIODIC:
            if self.last_retrain_time is None:
                should_retrain = True
            else:
                hours_since = (current_time - self.last_retrain_time).total_seconds() / 3600
                if hours_since >= self.config.retrain_interval_hours:
                    should_retrain = True
        
        elif self.config.learning_mode == LearningMode.CONTINUOUS:
            # Retrain every 50 new labels
            if len(self.labels_history) % 50 == 0 and len(self.labels_history) > 0:
                should_retrain = True
        
        elif self.config.learning_mode == LearningMode.ADAPTIVE:
            # Check recent accuracy
            recent_trades = self.state.closed_trades[-50:]
            if len(recent_trades) >= 20:
                wins = sum(1 for t in recent_trades if t.outcome == "WIN")
                accuracy = wins / len(recent_trades)
                if accuracy < 0.48:  # Below threshold
                    should_retrain = True
        
        if should_retrain:
            self._train_model()
            self.last_retrain_time = current_time
            self.retrain_count += 1
    
    def _train_model(self):
        """Train model on available data."""
        # Prepare training data
        if len(self.labels_history) < self.config.min_samples_for_train:
            return
        
        # Get recent data within sliding window
        window_start = len(self.labels_history) - min(
            len(self.labels_history),
            self.config.sliding_window_days * 96
        )
        
        features_data = []
        labels_data = []
        
        for i in range(window_start, len(self.labels_history)):
            label = self.labels_history[i]
            
            # Find matching features (by time)
            feature_idx = None
            for j, f in enumerate(self.features_history):
                if f['time'] == label['time']:
                    feature_idx = j
                    break
            
            if feature_idx is None:
                continue
            
            if label['outcome'] in ['WIN', 'LOSS']:
                features_data.append(self.features_history[feature_idx]['features'])
                labels_data.append(1 if label['outcome'] == 'WIN' else 0)
        
        if len(labels_data) < self.config.min_samples_for_train:
            return
        
        features_df = pd.DataFrame(features_data)
        labels_array = np.array(labels_data)
        
        try:
            self.model = self.model_trainer.train(features_df, labels_array)
            logger.info(f"Model retrained: {self.model.version} (samples: {len(labels_data)}, acc: {self.model.metrics.accuracy:.2%})")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    def _run_static_comparison(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: Optional[pd.DataFrame],
        symbol: str,
        warmup_end_idx: int
    ) -> BacktestMetrics:
        """Run backtest with static model for comparison."""
        # Save current state
        original_model = self.model
        
        # Use static model only
        self.model = self.model_static
        self.config.learning_mode = LearningMode.STATIC
        
        # Re-initialize state
        self._initialize_state()
        self.last_retrain_time = None
        
        # Fast-forward through warm-up
        for i in range(50, warmup_end_idx):
            self._compute_and_store_features(
                candles_15m.iloc[:i+1],
                candles_1h[:candles_15m.index[i]] if candles_1h is not None else None,
                symbol
            )
        
        # Run simulation
        for i in range(warmup_end_idx, len(candles_15m)):
            candle_time = candles_15m.index[i]
            self.state.current_date = candle_time
            
            self._check_daily_reset(candle_time)
            
            current_price = candles_15m['close'].iloc[i]
            high = candles_15m['high'].iloc[i]
            low = candles_15m['low'].iloc[i]
            
            self._update_open_trades(high, low, current_price, candle_time)
            
            current_equity = self._calculate_equity(current_price)
            self.equity_snapshots.append((candle_time, current_equity))
            self._update_drawdown(current_equity)
            
            features = self._compute_and_store_features(
                candles_15m.iloc[:i+1],
                candles_1h[:candle_time] if candles_1h is not None else None,
                symbol
            )
            
            if features is None:
                continue
            
            regime_state = self.regime_detector.detect_regime(
                candles_15m.iloc[max(0, i-100):i+1]
            )
            
            if self.model is not None and self.state.is_trading_enabled:
                self._generate_prediction_and_trade(
                    features, regime_state, current_price, candle_time, symbol
                )
        
        # Close remaining trades
        final_price = candles_15m['close'].iloc[-1]
        self._close_all_trades(final_price, candles_15m.index[-1])
        
        # Calculate metrics
        static_metrics = self._calculate_metrics()
        
        # Restore original model
        self.model = original_model
        
        return static_metrics
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        trades = self.state.closed_trades
        
        if len(trades) == 0:
            return self._empty_metrics()
        
        # Basic stats
        total_trades = len(trades)
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL stats
        all_pnl = [t.net_pnl for t in trades]
        win_pnl = [t.net_pnl for t in wins] if wins else [0]
        loss_pnl = [t.net_pnl for t in losses] if losses else [0]
        
        avg_win = np.mean(win_pnl) if win_pnl else 0
        avg_loss = abs(np.mean(loss_pnl)) if loss_pnl else 0
        
        total_wins = sum(win_pnl)
        total_losses = abs(sum(loss_pnl))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        expectancy = np.mean(all_pnl)
        expectancy_ratio = expectancy / avg_loss if avg_loss > 0 else 0
        
        # Returns
        final_equity = self.equity_snapshots[-1][1] if self.equity_snapshots else self.config.initial_capital
        total_return = final_equity - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital
        
        # Annualized return
        if len(self.equity_snapshots) > 1:
            days = (self.equity_snapshots[-1][0] - self.equity_snapshots[0][0]).days
            if days > 0:
                annualized_return = ((final_equity / self.config.initial_capital) ** (365 / days)) - 1
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # Drawdown stats
        max_dd = self.state.max_drawdown
        avg_dd = np.mean([d[1] for d in self.drawdown_snapshots]) if self.drawdown_snapshots else 0
        
        # Sharpe-like ratio (simplified - using daily returns)
        daily_returns = []
        prev_equity = self.config.initial_capital
        current_day = None
        
        for time, equity in self.equity_snapshots:
            if current_day is None or time.date() > current_day:
                if current_day is not None:
                    daily_returns.append((equity - prev_equity) / prev_equity)
                current_day = time.date()
                prev_equity = equity
        
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            negative_returns = [r for r in daily_returns if r < 0]
            sortino = np.mean(daily_returns) / np.std(negative_returns) * np.sqrt(252) if negative_returns and np.std(negative_returns) > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_streak = 0
        streak_type = None
        
        for trade in trades:
            if streak_type == trade.outcome:
                current_streak += 1
            else:
                if streak_type == "WIN":
                    max_consec_wins = max(max_consec_wins, current_streak)
                elif streak_type == "LOSS":
                    max_consec_losses = max(max_consec_losses, current_streak)
                current_streak = 1
                streak_type = trade.outcome
        
        if streak_type == "WIN":
            max_consec_wins = max(max_consec_wins, current_streak)
        elif streak_type == "LOSS":
            max_consec_losses = max(max_consec_losses, current_streak)
        
        # Trade duration
        durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0
        
        # Regime performance
        regime_perf = {}
        for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "UNKNOWN"]:
            regime_trades = [t for t in trades if t.regime == regime]
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t.outcome == "WIN"])
                regime_perf[regime] = {
                    'count': len(regime_trades),
                    'win_rate': regime_wins / len(regime_trades),
                    'avg_pnl': np.mean([t.net_pnl for t in regime_trades])
                }
        
        # Confidence bucket accuracy
        conf_buckets = {}
        bucket_ranges = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.00)]
        for low, high in bucket_ranges:
            bucket_trades = [t for t in trades if low <= t.confidence < high]
            if bucket_trades:
                bucket_wins = len([t for t in bucket_trades if t.outcome == "WIN"])
                conf_buckets[f"{low:.2f}-{high:.2f}"] = bucket_wins / len(bucket_trades)
        
        return BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            max_drawdown=max_dd * self.config.initial_capital,
            max_drawdown_pct=max_dd,
            avg_drawdown=avg_dd,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            expectancy_ratio=expectancy_ratio,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_trade_duration=avg_duration,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            regime_performance=regime_perf,
            confidence_bucket_accuracy=conf_buckets,
            equity_curve=self.equity_snapshots,
            drawdown_curve=self.drawdown_snapshots
        )
    
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics when no trades."""
        return BacktestMetrics(
            total_return=0, total_return_pct=0, annualized_return=0,
            max_drawdown=0, max_drawdown_pct=0, avg_drawdown=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0, expectancy=0, expectancy_ratio=0,
            sharpe_ratio=0, sortino_ratio=0, avg_trade_duration=0,
            max_consecutive_wins=0, max_consecutive_losses=0,
            regime_performance={}, confidence_bucket_accuracy={},
            equity_curve=[], drawdown_curve=[]
        )
