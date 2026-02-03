"""
AURIX Database Initialization Script

Creates the SQLite database with all required tables.
"""

import sqlite3
import os
from datetime import datetime


def init_database(db_path: str = "data/aurix.db"):
    """Initialize the AURIX database with all tables."""
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Candles table - stores OHLCV data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, open_time)
        )
    """)
    
    # Create index for efficient queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time 
        ON candles(symbol, timeframe, open_time DESC)
    """)
    
    # Features table - computed technical indicators
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            candle_time INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, candle_time, feature_name)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_features_symbol_time 
        ON features(symbol, candle_time DESC)
    """)
    
    # Labels table - training labels with PnL
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            candle_time INTEGER NOT NULL,
            direction TEXT NOT NULL,  -- LONG, SHORT, NEUTRAL
            label INTEGER NOT NULL,   -- 1=WIN, 0=LOSS
            holding_period_minutes INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            gross_return REAL NOT NULL,
            net_return REAL NOT NULL,
            transaction_cost REAL NOT NULL,
            is_contaminated INTEGER DEFAULT 0,
            regime TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, candle_time, direction, holding_period_minutes)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels_symbol_time 
        ON labels(symbol, candle_time DESC)
    """)
    
    # Predictions table - model predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            candle_time INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            direction TEXT NOT NULL,
            raw_probability REAL NOT NULL,
            calibrated_probability REAL NOT NULL,
            regime TEXT,
            dynamic_threshold REAL NOT NULL,
            signal_generated INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time 
        ON predictions(symbol, candle_time DESC)
    """)
    
    # Trades table - executed trades
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            order_id TEXT UNIQUE,
            direction TEXT NOT NULL,  -- LONG, SHORT
            entry_time TIMESTAMP NOT NULL,
            entry_price REAL NOT NULL,
            exit_time TIMESTAMP,
            exit_price REAL,
            quantity REAL NOT NULL,
            gross_pnl REAL,
            net_pnl REAL,
            fees REAL,
            slippage REAL,
            status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED, CANCELLED
            exit_reason TEXT,  -- TP, SL, SIGNAL, MANUAL
            confidence REAL,
            regime TEXT,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_status 
        ON trades(symbol, status)
    """)
    
    # Positions table - current open positions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            unrealized_pnl REAL DEFAULT 0,
            take_profit REAL,
            stop_loss REAL,
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Account state table - equity tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS account_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equity REAL NOT NULL,
            available_balance REAL NOT NULL,
            unrealized_pnl REAL DEFAULT 0,
            daily_pnl REAL DEFAULT 0,
            peak_equity REAL NOT NULL,
            current_drawdown REAL DEFAULT 0,
            consecutive_losses INTEGER DEFAULT 0,
            is_halted INTEGER DEFAULT 0,
            halt_reason TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Model metrics table - performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            window_size INTEGER,  -- number of samples
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_metrics_version 
        ON model_metrics(model_version, recorded_at DESC)
    """)
    
    # System events table - logging important events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,  -- INFO, WARNING, ERROR, CRITICAL
            message TEXT NOT NULL,
            details TEXT,  -- JSON blob
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type_time 
        ON system_events(event_type, created_at DESC)
    """)
    
    # Validation state table - Capital Trust Score tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            day_number INTEGER NOT NULL,
            phase TEXT NOT NULL,
            equity REAL NOT NULL,
            cts_score REAL,
            stability_score REAL,
            consistency_score REAL,
            risk_score REAL,
            recovery_score REAL,
            pattern_score REAL,
            expectancy_drift REAL,
            degradation_patterns TEXT,  -- JSON array
            warnings TEXT,  -- JSON array
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized at {db_path}")
    print("   Tables created:")
    print("   - candles")
    print("   - features")
    print("   - labels")
    print("   - predictions")
    print("   - trades")
    print("   - positions")
    print("   - account_state")
    print("   - model_metrics")
    print("   - system_events")
    print("   - validation_state")


if __name__ == "__main__":
    init_database()
