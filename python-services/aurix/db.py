"""
AURIX Database Interface

Provides SQLAlchemy-based database operations.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candle data."""
    symbol: str
    timeframe: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class Trade:
    """Trade record."""
    id: Optional[int]
    symbol: str
    order_id: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    fees: Optional[float] = None
    slippage: Optional[float] = None
    status: str = "OPEN"
    exit_reason: Optional[str] = None
    confidence: Optional[float] = None
    regime: Optional[str] = None
    model_version: Optional[str] = None


class Database:
    """
    SQLite database interface for AURIX.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database connection."""
        self.db_path = config.path
        self._ensure_connection()
    
    def _ensure_connection(self):
        """Ensure database file and tables exist."""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")  # Ensure WAL is enabled
        conn.row_factory = sqlite3.Row
        return conn
    
    # ==================== CANDLES ====================
    
    def insert_candle(self, candle: Candle):
        """Insert or update a candle."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, open_time, open, high, low, close, volume, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle.symbol, candle.timeframe, candle.open_time,
                candle.open, candle.high, candle.low, candle.close,
                candle.volume, candle.close_time
            ))
            conn.commit()
        finally:
            conn.close()
    
    def insert_candles(self, candles: List[Candle]):
        """Bulk insert candles."""
        conn = self._get_connection()
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, open_time, open, high, low, close, volume, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (c.symbol, c.timeframe, c.open_time, c.open, c.high, c.low, c.close, c.volume, c.close_time)
                for c in candles
            ])
            conn.commit()
        finally:
            conn.close()
    
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Candle]:
        """Get candles for a symbol and timeframe."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND open_time >= ?"
                params.append(start_time)
            if end_time:
                query += " AND open_time <= ?"
                params.append(end_time)
            
            query += " ORDER BY open_time DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                Candle(
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    open_time=row['open_time'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    close_time=row['close_time']
                )
                for row in reversed(rows)  # Return in chronological order
            ]
        finally:
            conn.close()
    
    def get_latest_candle_time(self, symbol: str, timeframe: str) -> Optional[int]:
        """Get the latest candle timestamp."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT MAX(open_time) as latest FROM candles 
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            row = cursor.fetchone()
            return row['latest'] if row else None
        finally:
            conn.close()
    
    # ==================== TRADES ====================
    
    def insert_trade(self, trade: Trade) -> int:
        """Insert a new trade."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO trades 
                (symbol, order_id, direction, entry_time, entry_price, quantity,
                 exit_time, exit_price, gross_pnl, net_pnl, fees, slippage,
                 status, exit_reason, confidence, regime, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.order_id, trade.direction,
                trade.entry_time.isoformat(), trade.entry_price, trade.quantity,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price, trade.gross_pnl, trade.net_pnl,
                trade.fees, trade.slippage, trade.status, trade.exit_reason,
                trade.confidence, trade.regime, trade.model_version
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def update_trade(self, trade_id: int, updates: Dict[str, Any]):
        """Update a trade record."""
        conn = self._get_connection()
        try:
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [trade_id]
            
            conn.execute(f"""
                UPDATE trades SET {set_clause} WHERE id = ?
            """, values)
            conn.commit()
        finally:
            conn.close()
    
    def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """Get all open trades."""
        conn = self._get_connection()
        try:
            if symbol:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?",
                    (symbol,)
                )
            else:
                cursor = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            
            return [self._row_to_trade(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_recent_trades(self, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?
            """, (limit,))
            return [self._row_to_trade(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def _row_to_trade(self, row) -> Trade:
        """Convert a database row to Trade object."""
        return Trade(
            id=row['id'],
            symbol=row['symbol'],
            order_id=row['order_id'],
            direction=row['direction'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            entry_price=row['entry_price'],
            quantity=row['quantity'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            exit_price=row['exit_price'],
            gross_pnl=row['gross_pnl'],
            net_pnl=row['net_pnl'],
            fees=row['fees'],
            slippage=row['slippage'],
            status=row['status'],
            exit_reason=row['exit_reason'],
            confidence=row['confidence'],
            regime=row['regime'],
            model_version=row['model_version']
        )
    
    # ==================== ACCOUNT STATE ====================
    
    def save_account_state(
        self,
        equity: float,
        available_balance: float,
        unrealized_pnl: float = 0,
        daily_pnl: float = 0,
        peak_equity: float = 0,
        current_drawdown: float = 0,
        consecutive_losses: int = 0,
        is_halted: bool = False,
        halt_reason: str = None
    ):
        """Save current account state."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO account_state 
                (equity, available_balance, unrealized_pnl, daily_pnl, peak_equity,
                 current_drawdown, consecutive_losses, is_halted, halt_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                equity, available_balance, unrealized_pnl, daily_pnl, peak_equity,
                current_drawdown, consecutive_losses, 1 if is_halted else 0, halt_reason
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_account_state(self) -> Optional[Dict]:
        """Get latest account state."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM account_state ORDER BY recorded_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    # ==================== LABELS ====================
    
    def insert_label(
        self,
        symbol: str,
        candle_time: int,
        direction: str,
        label: int,
        holding_period_minutes: int,
        entry_price: float,
        exit_price: float,
        gross_return: float,
        net_return: float,
        transaction_cost: float,
        is_contaminated: bool = False,
        regime: str = None
    ):
        """Insert a training label."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO labels 
                (symbol, candle_time, direction, label, holding_period_minutes,
                 entry_price, exit_price, gross_return, net_return, transaction_cost,
                 is_contaminated, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, candle_time, direction, label, holding_period_minutes,
                entry_price, exit_price, gross_return, net_return, transaction_cost,
                1 if is_contaminated else 0, regime
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_training_labels(
        self,
        symbol: str,
        direction: str,
        holding_period: int,
        limit: int = 10000,
        exclude_contaminated: bool = True
    ) -> List[Dict]:
        """Get labels for training."""
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM labels 
                WHERE symbol = ? AND direction = ? AND holding_period_minutes = ?
            """
            params = [symbol, direction, holding_period]
            
            if exclude_contaminated:
                query += " AND is_contaminated = 0"
            
            query += " ORDER BY candle_time DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    # ==================== PREDICTIONS ====================
    
    def insert_prediction(
        self,
        symbol: str,
        candle_time: int,
        model_version: str,
        direction: str,
        raw_probability: float,
        calibrated_probability: float,
        regime: str,
        dynamic_threshold: float,
        signal_generated: bool = False
    ):
        """Insert a prediction record."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO predictions 
                (symbol, candle_time, model_version, direction, raw_probability,
                 calibrated_probability, regime, dynamic_threshold, signal_generated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, candle_time, model_version, direction, raw_probability,
                calibrated_probability, regime, dynamic_threshold,
                1 if signal_generated else 0
            ))
            conn.commit()
        finally:
            conn.close()
    
    # ==================== SYSTEM EVENTS ====================
    
    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Dict = None
    ):
        """Log a system event."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO system_events (event_type, severity, message, details)
                VALUES (?, ?, ?, ?)
            """, (
                event_type, severity, message,
                json.dumps(details) if details else None
            ))
            conn.commit()
        finally:
            conn.close()
    
    # ==================== MODEL METRICS ====================
    
    def save_model_metric(
        self,
        model_version: str,
        metric_name: str,
        metric_value: float,
        window_size: int = None
    ):
        """Save a model metric."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO model_metrics (model_version, metric_name, metric_value, window_size)
                VALUES (?, ?, ?, ?)
            """, (model_version, metric_name, metric_value, window_size))
            conn.commit()
        finally:
            conn.close()
    
    def get_model_metrics(self, model_version: str, limit: int = 100) -> List[Dict]:
        """Get metrics for a model version."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM model_metrics 
                WHERE model_version = ? 
                ORDER BY recorded_at DESC LIMIT ?
            """, (model_version, limit))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
