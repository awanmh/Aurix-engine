// Package db provides SQLite database operations.
package db

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"aurix/internal/types"
)

// Client handles SQLite database operations
type Client struct {
	db *sql.DB
}

// NewClient creates a new database client
func NewClient(path string) (*Client, error) {
	// Add pragmas to connection string for busy timeout and WAL mode
	// This prevents SQLITE_BUSY errors when multiple processes access the DB
	connStr := path + "?_busy_timeout=5000&_journal_mode=WAL"
	
	db, err := sql.Open("sqlite", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}
	
	// Set connection pool settings
	db.SetMaxOpenConns(1) // SQLite only supports one writer
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(time.Hour)
	
	// Verify connection and set pragmas
	if _, err := db.Exec("PRAGMA busy_timeout = 5000"); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to set busy_timeout: %w", err)
	}
	
	return &Client{db: db}, nil
}

// Close closes the database connection
func (c *Client) Close() error {
	return c.db.Close()
}

// InsertCandle inserts or updates a candle with retry logic for SQLITE_BUSY
func (c *Client) InsertCandle(candle types.Candle) error {
	maxRetries := 5
	baseDelay := 100 * time.Millisecond
	
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		_, err := c.db.Exec(`
			INSERT OR REPLACE INTO candles 
			(symbol, timeframe, open_time, open, high, low, close, volume, close_time)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		`,
			candle.Symbol, candle.Timeframe, candle.OpenTime,
			candle.Open, candle.High, candle.Low, candle.Close,
			candle.Volume, candle.CloseTime,
		)
		
		if err == nil {
			return nil
		}
		
		lastErr = err
		// Check if it's a busy/locked error
		errStr := err.Error()
		if strings.Contains(errStr, "SQLITE_BUSY") || strings.Contains(errStr, "database is locked") {
			// Exponential backoff with jitter
			delay := baseDelay * time.Duration(1<<attempt)
			time.Sleep(delay)
			continue
		}
		
		// Non-retryable error
		return err
	}
	
	return fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr)
}

// InsertCandles bulk inserts candles
func (c *Client) InsertCandles(candles []types.Candle) error {
	tx, err := c.db.Begin()
	if err != nil {
		return err
	}
	
	stmt, err := tx.Prepare(`
		INSERT OR REPLACE INTO candles 
		(symbol, timeframe, open_time, open, high, low, close, volume, close_time)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	
	for _, candle := range candles {
		_, err = stmt.Exec(
			candle.Symbol, candle.Timeframe, candle.OpenTime,
			candle.Open, candle.High, candle.Low, candle.Close,
			candle.Volume, candle.CloseTime,
		)
		if err != nil {
			tx.Rollback()
			return err
		}
	}
	
	return tx.Commit()
}

// GetCandles retrieves candles from database
func (c *Client) GetCandles(symbol, timeframe string, limit int) ([]types.Candle, error) {
	rows, err := c.db.Query(`
		SELECT symbol, timeframe, open_time, open, high, low, close, volume, close_time
		FROM candles
		WHERE symbol = ? AND timeframe = ?
		ORDER BY open_time DESC
		LIMIT ?
	`, symbol, timeframe, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	var candles []types.Candle
	for rows.Next() {
		var candle types.Candle
		err := rows.Scan(
			&candle.Symbol, &candle.Timeframe, &candle.OpenTime,
			&candle.Open, &candle.High, &candle.Low, &candle.Close,
			&candle.Volume, &candle.CloseTime,
		)
		if err != nil {
			return nil, err
		}
		candles = append(candles, candle)
	}
	
	// Reverse to get chronological order
	for i, j := 0, len(candles)-1; i < j; i, j = i+1, j-1 {
		candles[i], candles[j] = candles[j], candles[i]
	}
	
	return candles, nil
}

// InsertTrade inserts a new trade
func (c *Client) InsertTrade(trade types.Trade) (int64, error) {
	result, err := c.db.Exec(`
		INSERT INTO trades 
		(symbol, order_id, direction, entry_time, entry_price, quantity,
		 status, confidence, regime, model_version)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`,
		trade.Symbol, trade.OrderID, trade.Direction,
		trade.EntryTime.Format(time.RFC3339), trade.EntryPrice, trade.Quantity,
		trade.Status, trade.Confidence, trade.Regime, trade.ModelVersion,
	)
	if err != nil {
		return 0, err
	}
	
	return result.LastInsertId()
}

// UpdateTrade updates an existing trade
func (c *Client) UpdateTrade(orderID string, exitPrice, grossPnL, netPnL, fees, slippage float64, exitReason string) error {
	_, err := c.db.Exec(`
		UPDATE trades SET
			exit_time = ?,
			exit_price = ?,
			gross_pnl = ?,
			net_pnl = ?,
			fees = ?,
			slippage = ?,
			status = 'CLOSED',
			exit_reason = ?
		WHERE order_id = ?
	`,
		time.Now().Format(time.RFC3339),
		exitPrice, grossPnL, netPnL, fees, slippage, exitReason, orderID,
	)
	return err
}

// GetOpenTrades gets all open trades
func (c *Client) GetOpenTrades(symbol string) ([]types.Trade, error) {
	query := "SELECT * FROM trades WHERE status = 'OPEN'"
	args := []interface{}{}
	
	if symbol != "" {
		query += " AND symbol = ?"
		args = append(args, symbol)
	}
	
	rows, err := c.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	var trades []types.Trade
	for rows.Next() {
		var trade types.Trade
		var entryTime, exitTime sql.NullString
		
		err := rows.Scan(
			&trade.ID, &trade.Symbol, &trade.OrderID, &trade.Direction,
			&entryTime, &trade.EntryPrice, &exitTime, &trade.ExitPrice,
			&trade.Quantity, &trade.GrossPnL, &trade.NetPnL, &trade.Fees,
			&trade.Slippage, &trade.Status, &trade.ExitReason,
			&trade.Confidence, &trade.Regime, &trade.ModelVersion,
		)
		if err != nil {
			continue
		}
		
		if entryTime.Valid {
			trade.EntryTime, _ = time.Parse(time.RFC3339, entryTime.String)
		}
		if exitTime.Valid {
			trade.ExitTime, _ = time.Parse(time.RFC3339, exitTime.String)
		}
		
		trades = append(trades, trade)
	}
	
	return trades, nil
}

// SaveAccountState saves current account state
func (c *Client) SaveAccountState(state types.AccountState) error {
	_, err := c.db.Exec(`
		INSERT INTO account_state 
		(equity, available_balance, unrealized_pnl, daily_pnl, peak_equity,
		 current_drawdown, consecutive_losses, is_halted, halt_reason)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`,
		state.Equity, state.AvailableBalance, state.UnrealizedPnL,
		state.DailyPnL, state.PeakEquity, state.CurrentDrawdown,
		state.ConsecutiveLosses, state.IsHalted, state.HaltReason,
	)
	return err
}

// GetLatestAccountState gets the most recent account state
func (c *Client) GetLatestAccountState() (*types.AccountState, error) {
	row := c.db.QueryRow(`
		SELECT equity, available_balance, unrealized_pnl, daily_pnl, peak_equity,
		       current_drawdown, consecutive_losses, is_halted, halt_reason, recorded_at
		FROM account_state
		ORDER BY recorded_at DESC
		LIMIT 1
	`)
	
	var state types.AccountState
	var recordedAt string
	var haltReason sql.NullString
	
	err := row.Scan(
		&state.Equity, &state.AvailableBalance, &state.UnrealizedPnL,
		&state.DailyPnL, &state.PeakEquity, &state.CurrentDrawdown,
		&state.ConsecutiveLosses, &state.IsHalted, &haltReason, &recordedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	
	if haltReason.Valid {
		state.HaltReason = haltReason.String
	}
	state.RecordedAt, _ = time.Parse(time.RFC3339, recordedAt)
	
	return &state, nil
}

// LogEvent logs a system event
func (c *Client) LogEvent(eventType, severity, message string, details string) error {
	_, err := c.db.Exec(`
		INSERT INTO system_events (event_type, severity, message, details)
		VALUES (?, ?, ?, ?)
	`, eventType, severity, message, details)
	return err
}
