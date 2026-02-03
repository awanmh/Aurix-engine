"""
AURIX Decision Engine

Main ML orchestrator that:
1. Listens for new candles from Go collector
2. Computes features
3. Makes predictions with calibrated probabilities
4. Detects market regime
5. Applies dynamic confidence thresholds
6. Applies capital efficiency gating
7. Publishes trading signals

Usage:
    python decision_engine.py --config config/config.yaml
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from aurix.config import load_config, AurixConfig
from aurix.db import Database
from aurix.redis_bus import RedisBus
from aurix.features import FeatureEngine
from aurix.labeling import LabelingEngine
from aurix.ml import MLTrainer
from aurix.regime import RegimeDetector, MarketRegime
from aurix.capital import (
    CapitalEfficiencyGate,
    CapitalEfficiencyScorer,
    PairManager,
    OvertradingDetector,
    PsychDriftDetector,
    TradeRecord
)
from aurix.reality import (
    RealityConfig,
    KillSwitch,
    OverfitMonitor,
    RetrainController,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/decision_engine.log')
    ]
)
logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Main decision engine that orchestrates ML predictions and signal generation.
    """
    
    def __init__(self, config: AurixConfig):
        """
        Initialize the decision engine.
        
        Args:
            config: AURIX configuration
        """
        self.config = config
        
        # Initialize components
        self.db = Database(config.database)
        self.redis = RedisBus(config.redis)
        self.feature_engine = FeatureEngine(
            lookback_periods=config.ml.feature_lookback_periods
        )
        self.labeling_engine = LabelingEngine(
            holding_periods_minutes=config.labeling.holding_periods_minutes,
            fee_rate_bps=config.labeling.fee_rate_bps,
            slippage_bps=config.labeling.slippage_bps
        )
        self.ml_trainer = MLTrainer(
            model_type=config.ml.model_type,
            psi_threshold=config.ml.psi_threshold
        )
        self.regime_detector = RegimeDetector()
        
        # Capital Efficiency Gate
        self.efficiency_scorer = CapitalEfficiencyScorer(
            window_days=getattr(config.capital_efficiency, 'window_days', 30)
        )
        self.pair_manager = PairManager(
            max_active_pairs=getattr(config.capital_efficiency, 'max_active_pairs', 5)
        )
        self.overtrading_detector = OvertradingDetector(
            max_trades_per_day=getattr(config.capital_efficiency, 'max_trades_per_day', 10)
        )
        self.psych_drift_detector = PsychDriftDetector()
        
        self.efficiency_gate = CapitalEfficiencyGate(
            efficiency_scorer=self.efficiency_scorer,
            pair_manager=self.pair_manager,
            overtrading_detector=self.overtrading_detector,
            psych_drift_detector=self.psych_drift_detector,
            enable_ces=getattr(config.capital_efficiency, 'enabled', True),
            enable_pair_filter=getattr(config.capital_efficiency, 'pair_filter_enabled', True),
            enable_overtrading=getattr(config.capital_efficiency, 'overtrading_enabled', True),
            enable_psych_drift=getattr(config.capital_efficiency, 'psych_drift_enabled', True)
        )
        
        # Reality Validation Layer
        self.reality_config = RealityConfig()
        self.kill_switch = KillSwitch(
            max_drawdown_pct=self.reality_config.max_drawdown_pct,
            max_consecutive_losses=self.reality_config.max_consecutive_losses,
            min_avg_confidence=self.reality_config.min_avg_confidence
        )
        self.overfit_monitor = OverfitMonitor(
            max_train_forward_divergence=self.reality_config.max_train_forward_divergence,
            auc_collapse_threshold=self.reality_config.auc_collapse_threshold,
            overfit_confidence_penalty=self.reality_config.overfit_confidence_penalty
        )
        self.retrain_controller = RetrainController(
            min_cooldown_days=self.reality_config.min_retrain_cooldown_days,
            performance_decay_threshold=self.reality_config.performance_decay_threshold,
            regime_confirmation_bars=self.reality_config.regime_change_confirmation_bars
        )
        
        # Tracking state
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.peak_equity = getattr(config.trading, 'initial_capital', 10000.0)
        self.current_equity = self.peak_equity
        
        # State
        self.running = False
        self.last_retrain_time: Optional[datetime] = None
        self.candle_buffer: list = []
        self.min_candles_for_prediction = 100
        
        # Load existing models if available
        self._load_models()
        
        # Initialize pairs for tracking
        self._initialize_pairs()
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            self.ml_trainer.load_model("LONG")
            logger.info("Loaded LONG model")
        except Exception as e:
            logger.warning(f"No LONG model found: {e}")
        
        try:
            self.ml_trainer.load_model("SHORT")
            logger.info("Loaded SHORT model")
        except Exception as e:
            logger.warning(f"No SHORT model found: {e}")
    
    def _initialize_pairs(self):
        """Initialize pair manager with trading symbols."""
        # Add main trading symbol
        self.pair_manager.add_pair(self.config.trading.symbol)
        
        # Add additional pairs if configured
        additional_pairs = getattr(self.config.trading, 'additional_pairs', [])
        for pair in additional_pairs:
            self.pair_manager.add_pair(pair)
        
        # Initial ranking
        self.pair_manager.rank_pairs()
    
    def start(self):
        """Start the decision engine."""
        logger.info("=" * 60)
        logger.info("           AURIX Decision Engine")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.config.trading.symbol}")
        logger.info(f"Model Type: {self.config.ml.model_type}")
        logger.info(f"Base Confidence Threshold: {self.config.ml.base_confidence_threshold}")
        logger.info("=" * 60)
        
        self.running = True
        
        # Subscribe to candle updates
        self.redis.subscribe(
            self.config.redis.channel_signals,
            self._handle_signal
        )
        
        # Subscribe to control commands
        self.redis.subscribe(
            self.config.redis.channel_control,
            self._handle_control
        )
        
        # Start subscriber
        self.redis.start_subscriber()
        
        # Main loop
        self._run_main_loop()
    
    def stop(self):
        """Stop the decision engine."""
        logger.info("Stopping decision engine...")
        self.running = False
        self.redis.stop_subscriber()
    
    def _run_main_loop(self):
        """Main processing loop."""
        heartbeat_interval = 30
        retrain_check_interval = 3600  # Check hourly
        
        last_heartbeat = datetime.now()
        last_retrain_check = datetime.now()
        
        while self.running:
            try:
                now = datetime.now()
                
                # Send heartbeat
                if (now - last_heartbeat).seconds >= heartbeat_interval:
                    self.redis.publish_heartbeat("decision_engine")
                    last_heartbeat = now
                
                # Check if retraining is needed
                if (now - last_retrain_check).seconds >= retrain_check_interval:
                    self._check_retrain_needed()
                    last_retrain_check = now
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)
    
    def _handle_signal(self, channel: str, data: Dict):
        """Handle incoming signals from Redis."""
        signal_type = data.get('type')
        
        if signal_type == 'NEW_CANDLE':
            self._process_candle(data)
        elif signal_type == 'TRADE_COMPLETE':
            self._record_trade_for_labeling(data)
    
    def _handle_control(self, channel: str, data: Dict):
        """Handle control commands."""
        command = data.get('command')
        
        if command == 'HALT':
            logger.warning(f"HALT command received: {data.get('reason')}")
        elif command == 'RETRAIN':
            self._trigger_retrain()
    
    def _process_candle(self, candle_data: Dict):
        """Process a new candle and potentially generate a signal."""
        try:
            symbol = candle_data.get('symbol')
            timeframe = candle_data.get('timeframe')
            
            logger.debug(f"Processing candle: {symbol} {timeframe}")
            
            # Only process 15m candles for now
            if timeframe != '15m':
                return
            
            # Get historical candles from database
            candles = self.db.get_candles(
                symbol=symbol,
                timeframe='15m',
                limit=200
            )
            
            if len(candles) < self.min_candles_for_prediction:
                logger.warning(f"Insufficient candles: {len(candles)}")
                return
            
            # Convert to DataFrame
            df = self._candles_to_dataframe(candles)
            
            # Get 1h candles for HTF features
            candles_1h = self.db.get_candles(
                symbol=symbol,
                timeframe='1h',
                limit=100
            )
            df_1h = self._candles_to_dataframe(candles_1h) if candles_1h else None
            
            # Compute features
            features = self.feature_engine.get_latest_features(df, df_1h)
            if features is None:
                logger.warning("Feature computation failed")
                return
            
            # Detect regime
            regime_state = self.regime_detector.detect_regime(df)
            current_regime = regime_state.regime
            
            logger.info(f"Regime: {current_regime.value} (confidence: {regime_state.confidence:.2f})")
            
            # Calculate dynamic threshold
            base_threshold = self.config.ml.base_confidence_threshold
            regime_adjustment = self.regime_detector.get_confidence_adjustment(current_regime)
            dynamic_threshold = base_threshold + regime_adjustment
            
            # Check if models are loaded
            if self.ml_trainer.long_model is None:
                logger.warning("No model loaded, skipping prediction")
                return
            
            # Prepare feature vector
            feature_names = self.ml_trainer.long_model.feature_names
            X = np.array([[features.get(f, 0) for f in feature_names]])
            
            # Check PSI for distribution shift
            psi_result = self.ml_trainer.check_psi(X, "LONG")
            if psi_result.get('status') == 'CRITICAL':
                logger.warning(f"High PSI detected: {psi_result.get('critical_features')}")
                # Consider triggering retrain
            
            # Make predictions
            _, long_prob = self.ml_trainer.predict(X, "LONG")
            _, short_prob = self.ml_trainer.predict(X, "SHORT")
            
            long_confidence = long_prob[0]
            short_confidence = short_prob[0]
            
            logger.info(f"Prediction: LONG={long_confidence:.1%}, SHORT={short_confidence:.1%} (threshold={dynamic_threshold:.1%})")
            
            # Store prediction
            latest_candle = candles[-1]
            self.db.insert_prediction(
                symbol=symbol,
                candle_time=latest_candle.open_time,
                model_version=self.ml_trainer.get_model_version("LONG") or "unknown",
                direction="LONG",
                raw_probability=long_confidence,
                calibrated_probability=long_confidence,
                regime=current_regime.value,
                dynamic_threshold=dynamic_threshold,
                signal_generated=False
            )
            
            # Generate signal if threshold met
            current_price = float(candle_data.get('close', 0))
            atr = features.get('atr', current_price * 0.01)
            
            # Calculate base position size (will be modified by gate)
            base_size = 1.0  # Placeholder, actual size calculated by executor
            
            # Determine which direction has the signal
            has_long_signal = long_confidence >= dynamic_threshold and long_confidence > short_confidence
            has_short_signal = short_confidence >= dynamic_threshold and short_confidence > long_confidence
            
            if has_long_signal or has_short_signal:
                direction = "LONG" if has_long_signal else "SHORT"
                confidence = long_confidence if has_long_signal else short_confidence
                
                # === REALITY VALIDATION LAYER CHECKS ===
                
                # Check kill switch - hard stop if triggered
                kill_state = self.kill_switch.check_all(
                    current_drawdown_pct=self.current_drawdown,
                    consecutive_losses=self.consecutive_losses,
                    last_confidence=confidence
                )
                
                if kill_state.is_active:
                    logger.warning(f"[KILL SWITCH] Signal blocked: {kill_state.reason}")
                    return
                
                # Apply overfitting confidence penalty
                overfit_penalty = self.overfit_monitor.get_confidence_penalty()
                if overfit_penalty > 0:
                    logger.warning(f"[OVERFIT] Applying confidence penalty: -{overfit_penalty:.1%}")
                    confidence = confidence * (1 - overfit_penalty)
                
                # Check if model should be frozen
                if self.overfit_monitor.should_freeze_model():
                    logger.warning("[OVERFIT] Model frozen due to severe overfitting - blocking signal")
                    return
                
                # === END REALITY CHECKS ===
                
                # Apply Capital Efficiency Gate
                gate_result = self.efficiency_gate.should_trade(
                    symbol=symbol,
                    base_confidence=confidence,
                    base_size=base_size
                )
                
                if not gate_result.approved:
                    logger.info(f"[GATE] Signal blocked: {gate_result.reason}")
                    return
                
                # Apply modifiers
                adjusted_confidence = confidence * gate_result.confidence_modifier
                adjusted_size = base_size * gate_result.size_modifier * gate_result.pair_weight
                
                logger.info(f"Gate approved: conf_mod={gate_result.confidence_modifier:.2f}, "
                           f"size_mod={gate_result.size_modifier:.2f}, "
                           f"pair_weight={gate_result.pair_weight:.2f}")
                
                # Check if adjusted confidence still meets threshold
                if adjusted_confidence < dynamic_threshold:
                    logger.info(f"[THRESHOLD] Adjusted confidence {adjusted_confidence:.1%} below {dynamic_threshold:.1%}")
                    return
                
                self._generate_signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=adjusted_confidence,
                    entry_price=current_price,
                    atr=atr,
                    regime=current_regime.value,
                    size_modifier=adjusted_size
                )
                
        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)
    
    def _generate_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        atr: float,
        regime: str,
        size_modifier: float = 1.0
    ):
        """Generate and publish a trading signal."""
        # Calculate TP/SL based on ATR
        if direction == "LONG":
            take_profit = entry_price + (atr * 2.0)
            stop_loss = entry_price - (atr * 1.0)
        else:
            take_profit = entry_price - (atr * 2.0)
            stop_loss = entry_price + (atr * 1.0)
        
        # Signal quantity will be calculated by executor based on risk
        quantity = 0  # Placeholder, executor will calculate
        
        logger.info(f"📊 SIGNAL: {direction} {symbol} @ {entry_price:.2f} "
                   f"(confidence={confidence:.1%}, TP={take_profit:.2f}, SL={stop_loss:.2f}, "
                   f"size_mod={size_modifier:.2f})")
        
        # Publish signal
        success = self.redis.publish_signal(
            signal_type="OPEN",
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            quantity=quantity,
            regime=regime,
            model_version=self.ml_trainer.get_model_version(direction),
            size_modifier=size_modifier
        )
        
        if success:
            logger.info("Signal published successfully")
        else:
            logger.error("Failed to publish signal")
    
    def _record_trade_for_labeling(self, trade_data: Dict):
        """Record completed trade for contamination detection."""
        timestamp = int(datetime.now().timestamp() * 1000)
        self.labeling_engine.record_our_trade(timestamp)
    
    def _check_retrain_needed(self):
        """Check if model retraining is needed."""
        # Check time since last retrain
        if self.last_retrain_time:
            hours_since = (datetime.now() - self.last_retrain_time).seconds / 3600
            if hours_since < self.config.ml.retrain_interval_hours:
                return
        
        # Check model accuracy (would need prediction tracking)
        # For now, just trigger periodic retrain
        logger.info("Checking if retrain is needed...")
        self._trigger_retrain()
    
    def _trigger_retrain(self):
        """Trigger model retraining."""
        logger.info("Starting model retrain...")
        
        try:
            # Get historical candles
            candles = self.db.get_candles(
                symbol=self.config.trading.symbol,
                timeframe='15m',
                limit=10000
            )
            
            if len(candles) < self.config.ml.min_samples_for_retrain:
                logger.warning(f"Insufficient data for retrain: {len(candles)}")
                return
            
            df = self._candles_to_dataframe(candles)
            
            # Compute features
            features_df = self.feature_engine.compute_features(df)
            
            # Generate labels
            labels_long = self.labeling_engine.compute_labels(df, "LONG")
            labels_short = self.labeling_engine.compute_labels(df, "SHORT")
            
            # Get training data
            candle_times_long, y_long = self.labeling_engine.get_training_labels(labels_long)
            candle_times_short, y_short = self.labeling_engine.get_training_labels(labels_short)
            
            if len(y_long) < 100 or len(y_short) < 100:
                logger.warning("Insufficient labels for training")
                return
            
            # Align features with labels
            # (Simplified - in production would need proper index alignment)
            X = features_df.values[-len(y_long):]
            feature_names = list(features_df.columns)
            
            # Train models
            self.ml_trainer.train(X, y_long, feature_names, "LONG")
            self.ml_trainer.train(X, y_short, feature_names, "SHORT")
            
            self.last_retrain_time = datetime.now()
            logger.info("Model retrain complete")
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}", exc_info=True)
    
    def _candles_to_dataframe(self, candles) -> pd.DataFrame:
        """Convert candle list to DataFrame."""
        data = []
        for c in candles:
            data.append({
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            })
        
        df = pd.DataFrame(data)
        df.index = pd.to_datetime([c.open_time for c in candles], unit='ms')
        return df


def main():
    parser = argparse.ArgumentParser(description='AURIX Decision Engine')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Create and start engine
    engine = DecisionEngine(config)
    
    # Handle signals
    import signal as sig
    def signal_handler(signum, frame):
        engine.stop()
        sys.exit(0)
    
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    # Start
    try:
        engine.start()
    except Exception as e:
        logger.error(f"Engine failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
