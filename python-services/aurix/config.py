"""
AURIX Configuration Management

Loads and validates configuration from YAML files.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str = "binance"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    
    # WebSocket
    websocket_reconnect_delay_ms: int = 5000
    websocket_max_reconnect_attempts: int = 10
    websocket_ping_interval_seconds: int = 30
    
    # Rate limits
    orders_per_second: int = 5
    orders_per_minute: int = 100


@dataclass
class TradingConfig:
    """Trading configuration."""
    symbol: str = "BTCUSDT"
    leverage: int = 5
    margin_type: str = "CROSSED"
    timeframes: List[str] = field(default_factory=lambda: ["1m", "15m", "1h"])


@dataclass
class RiskConfig:
    """Risk management configuration."""
    initial_capital: float = 10000.0
    risk_per_trade_percent: float = 1.0
    max_position_size_percent: float = 5.0
    max_daily_loss_percent: float = 3.0
    max_drawdown_percent: float = 5.0
    max_consecutive_losses: int = 8
    pause_after_consecutive_losses: int = 5
    pause_duration_minutes: int = 60
    cooldown_after_loss_minutes: int = 5
    max_slippage_percent: float = 0.1
    expected_slippage_bps: int = 5


@dataclass
class MLConfig:
    """ML configuration."""
    model_type: str = "lightgbm"
    base_confidence_threshold: float = 0.60
    regime_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "trending_up": -0.02,
        "trending_down": -0.02,
        "ranging": 0.05,
        "volatile": 0.10,
        "unknown": 0.15
    })
    initial_train_days: int = 14
    retrain_interval_hours: int = 24
    min_samples_for_retrain: int = 500
    retrain_on_accuracy_below: float = 0.52
    psi_threshold: float = 0.25
    shadow_validation_hours: int = 24
    feature_lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class LabelingConfig:
    """Labeling configuration."""
    holding_periods_minutes: List[int] = field(default_factory=lambda: [15, 60, 240])
    primary_holding_period: int = 15
    fee_rate_bps: int = 4
    slippage_bps: int = 5
    min_return_for_label: float = 0.002
    exclude_marginal_labels: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "sqlite"
    path: str = "data/aurix.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 6


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    channel_signals: str = "aurix:signals"
    channel_heartbeat: str = "aurix:heartbeat"
    channel_control: str = "aurix:control"
    publish_timeout_ms: int = 1000
    subscribe_timeout_ms: int = 5000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: str = "data/logs/aurix.log"
    max_size_mb: int = 100
    backup_count: int = 5


@dataclass
class CapitalEfficiencyConfig:
    """Capital efficiency layer configuration."""
    enabled: bool = True
    window_days: int = 30
    max_active_pairs: int = 5
    max_trades_per_day: int = 10
    pair_filter_enabled: bool = True
    overtrading_enabled: bool = True
    psych_drift_enabled: bool = True


@dataclass
class ValidationConfig:
    """Validation mode configuration."""
    enabled: bool = False
    duration_days: int = 14
    min_trades_for_validation: int = 50
    cts_threshold: int = 80


@dataclass
class AurixConfig:
    """Main AURIX configuration."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    capital_efficiency: CapitalEfficiencyConfig = field(default_factory=CapitalEfficiencyConfig)


def load_config(config_path: str = "config/config.yaml") -> AurixConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        AurixConfig object with all settings
    """
    # Check if file exists
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return AurixConfig()
    
    # Load YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return AurixConfig()
    
    # Parse sections
    config = AurixConfig()
    
    # Exchange
    if 'exchange' in raw_config:
        ex = raw_config['exchange']
        config.exchange = ExchangeConfig(
            name=ex.get('name', 'binance'),
            testnet=ex.get('testnet', True),
            api_key=ex.get('api_key', ''),
            api_secret=ex.get('api_secret', ''),
            websocket_reconnect_delay_ms=ex.get('websocket', {}).get('reconnect_delay_ms', 5000),
            websocket_max_reconnect_attempts=ex.get('websocket', {}).get('max_reconnect_attempts', 10),
            websocket_ping_interval_seconds=ex.get('websocket', {}).get('ping_interval_seconds', 30),
            orders_per_second=ex.get('rate_limits', {}).get('orders_per_second', 5),
            orders_per_minute=ex.get('rate_limits', {}).get('orders_per_minute', 100)
        )
    
    # Trading
    if 'trading' in raw_config:
        tr = raw_config['trading']
        config.trading = TradingConfig(
            symbol=tr.get('symbol', 'BTCUSDT'),
            leverage=tr.get('leverage', 5),
            margin_type=tr.get('margin_type', 'CROSSED'),
            timeframes=tr.get('timeframes', ['1m', '15m', '1h'])
        )
    
    # Risk
    if 'risk' in raw_config:
        ri = raw_config['risk']
        config.risk = RiskConfig(
            initial_capital=ri.get('initial_capital', 10000.0),
            risk_per_trade_percent=ri.get('risk_per_trade_percent', 1.0),
            max_position_size_percent=ri.get('max_position_size_percent', 5.0),
            max_daily_loss_percent=ri.get('max_daily_loss_percent', 3.0),
            max_drawdown_percent=ri.get('max_drawdown_percent', 5.0),
            max_consecutive_losses=ri.get('max_consecutive_losses', 8),
            pause_after_consecutive_losses=ri.get('pause_after_consecutive_losses', 5),
            pause_duration_minutes=ri.get('pause_duration_minutes', 60),
            cooldown_after_loss_minutes=ri.get('cooldown_after_loss_minutes', 5),
            max_slippage_percent=ri.get('max_slippage_percent', 0.1),
            expected_slippage_bps=ri.get('expected_slippage_bps', 5)
        )
    
    # ML
    if 'ml' in raw_config:
        ml = raw_config['ml']
        config.ml = MLConfig(
            model_type=ml.get('model_type', 'lightgbm'),
            base_confidence_threshold=ml.get('base_confidence_threshold', 0.60),
            regime_adjustments=ml.get('regime_adjustments', config.ml.regime_adjustments),
            initial_train_days=ml.get('initial_train_days', 14),
            retrain_interval_hours=ml.get('retrain_interval_hours', 24),
            min_samples_for_retrain=ml.get('min_samples_for_retrain', 500),
            retrain_on_accuracy_below=ml.get('retrain_on_accuracy_below', 0.52),
            psi_threshold=ml.get('psi_threshold', 0.25),
            shadow_validation_hours=ml.get('shadow_validation_hours', 24),
            feature_lookback_periods=ml.get('feature_lookback_periods', [5, 10, 20, 50])
        )
    
    # Labeling
    if 'labeling' in raw_config:
        lb = raw_config['labeling']
        config.labeling = LabelingConfig(
            holding_periods_minutes=lb.get('holding_periods_minutes', [15, 60, 240]),
            primary_holding_period=lb.get('primary_holding_period', 15),
            fee_rate_bps=lb.get('fee_rate_bps', 4),
            slippage_bps=lb.get('slippage_bps', 5),
            min_return_for_label=lb.get('min_return_for_label', 0.002),
            exclude_marginal_labels=lb.get('exclude_marginal_labels', True)
        )
    
    # Database
    if 'database' in raw_config:
        db = raw_config['database']
        config.database = DatabaseConfig(
            type=db.get('type', 'sqlite'),
            path=db.get('path', 'data/aurix.db'),
            backup_enabled=db.get('backup_enabled', True),
            backup_interval_hours=db.get('backup_interval_hours', 6)
        )
    
    # Redis
    if 'redis' in raw_config:
        rd = raw_config['redis']
        config.redis = RedisConfig(
            host=rd.get('host', 'localhost'),
            port=rd.get('port', 6379),
            password=rd.get('password', ''),
            db=rd.get('db', 0),
            channel_signals=rd.get('channel_signals', 'aurix:signals'),
            channel_heartbeat=rd.get('channel_heartbeat', 'aurix:heartbeat'),
            channel_control=rd.get('channel_control', 'aurix:control'),
            publish_timeout_ms=rd.get('publish_timeout_ms', 1000),
            subscribe_timeout_ms=rd.get('subscribe_timeout_ms', 5000)
        )
    
    # Logging
    if 'logging' in raw_config:
        lg = raw_config['logging']
        config.logging = LoggingConfig(
            level=lg.get('level', 'INFO'),
            format=lg.get('format', config.logging.format),
            file=lg.get('file', 'data/logs/aurix.log'),
            max_size_mb=lg.get('max_size_mb', 100),
            backup_count=lg.get('backup_count', 5)
        )
    
    # Validation
    if 'validation' in raw_config:
        vl = raw_config['validation']
        config.validation = ValidationConfig(
            enabled=vl.get('enabled', False),
            duration_days=vl.get('duration_days', 14),
            min_trades_for_validation=vl.get('min_trades_for_validation', 50),
            cts_threshold=vl.get('cts_threshold', 80)
        )
    
    return config


def validate_config(config: AurixConfig) -> List[str]:
    """
    Validate configuration for common issues.
    
    Returns:
        List of warning/error messages
    """
    warnings = []
    
    # Check testnet
    if not config.exchange.testnet:
        warnings.append("⚠️ LIVE TRADING ENABLED - testnet is false!")
    
    # Check API keys
    if not config.exchange.api_key or not config.exchange.api_secret:
        warnings.append("❌ API keys not configured")
    
    # Check risk settings
    if config.risk.risk_per_trade_percent > 2.0:
        warnings.append(f"⚠️ High risk per trade: {config.risk.risk_per_trade_percent}%")
    
    if config.risk.max_drawdown_percent > 10.0:
        warnings.append(f"⚠️ High max drawdown: {config.risk.max_drawdown_percent}%")
    
    # Check ML settings
    if config.ml.base_confidence_threshold < 0.55:
        warnings.append(f"⚠️ Low confidence threshold: {config.ml.base_confidence_threshold}")
    
    return warnings
