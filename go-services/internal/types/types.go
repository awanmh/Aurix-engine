// Package types provides shared data types for AURIX services.
package types

import "time"

// Candle represents OHLCV candlestick data
type Candle struct {
	Symbol    string    `json:"symbol"`
	Timeframe string    `json:"timeframe"`
	OpenTime  int64     `json:"open_time"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	CloseTime int64     `json:"close_time"`
	Timestamp time.Time `json:"timestamp"`
}

// Signal represents a trading signal from the ML engine
type Signal struct {
	Type         string  `json:"type"`          // OPEN, CLOSE
	Symbol       string  `json:"symbol"`
	Direction    string  `json:"direction"`     // LONG, SHORT
	Confidence   float64 `json:"confidence"`
	EntryPrice   float64 `json:"entry_price"`
	TakeProfit   float64 `json:"take_profit"`
	StopLoss     float64 `json:"stop_loss"`
	Quantity     float64 `json:"quantity"`
	Regime       string  `json:"regime"`
	ModelVersion string  `json:"model_version"`
	Timestamp    string  `json:"_timestamp"`
}

// Order represents an exchange order
type Order struct {
	ID            string    `json:"id"`
	Symbol        string    `json:"symbol"`
	Side          string    `json:"side"`          // BUY, SELL
	Type          string    `json:"type"`          // MARKET, LIMIT
	Quantity      float64   `json:"quantity"`
	Price         float64   `json:"price"`
	StopPrice     float64   `json:"stop_price"`
	Status        string    `json:"status"`
	FilledQty     float64   `json:"filled_qty"`
	AvgPrice      float64   `json:"avg_price"`
	Commission    float64   `json:"commission"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// Position represents an open trading position
type Position struct {
	Symbol        string    `json:"symbol"`
	Direction     string    `json:"direction"`    // LONG, SHORT
	EntryPrice    float64   `json:"entry_price"`
	Quantity      float64   `json:"quantity"`
	UnrealizedPnL float64   `json:"unrealized_pnl"`
	TakeProfit    float64   `json:"take_profit"`
	StopLoss      float64   `json:"stop_loss"`
	OpenedAt      time.Time `json:"opened_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// AccountState represents current account status
type AccountState struct {
	Equity            float64   `json:"equity"`
	AvailableBalance  float64   `json:"available_balance"`
	UnrealizedPnL     float64   `json:"unrealized_pnl"`
	DailyPnL          float64   `json:"daily_pnl"`
	PeakEquity        float64   `json:"peak_equity"`
	CurrentDrawdown   float64   `json:"current_drawdown"`
	ConsecutiveLosses int       `json:"consecutive_losses"`
	IsHalted          bool      `json:"is_halted"`
	HaltReason        string    `json:"halt_reason"`
	RecordedAt        time.Time `json:"recorded_at"`
}

// ControlCommand represents a control message for system management
type ControlCommand struct {
	Command   string `json:"command"`   // HALT, RESUME, CLOSE_ALL
	Reason    string `json:"reason"`
	Timestamp string `json:"_timestamp"`
}

// Heartbeat represents a service health check message
type Heartbeat struct {
	Service   string `json:"service"`
	Status    string `json:"status"`   // alive, degraded, unhealthy
	Timestamp string `json:"_timestamp"`
	Details   map[string]interface{} `json:"details"`
}

// Trade represents a completed trade record
type Trade struct {
	ID           int64     `json:"id"`
	Symbol       string    `json:"symbol"`
	OrderID      string    `json:"order_id"`
	Direction    string    `json:"direction"`
	EntryTime    time.Time `json:"entry_time"`
	EntryPrice   float64   `json:"entry_price"`
	ExitTime     time.Time `json:"exit_time"`
	ExitPrice    float64   `json:"exit_price"`
	Quantity     float64   `json:"quantity"`
	GrossPnL     float64   `json:"gross_pnl"`
	NetPnL       float64   `json:"net_pnl"`
	Fees         float64   `json:"fees"`
	Slippage     float64   `json:"slippage"`
	Status       string    `json:"status"`       // OPEN, CLOSED, CANCELLED
	ExitReason   string    `json:"exit_reason"`  // TP, SL, SIGNAL, MANUAL
	Confidence   float64   `json:"confidence"`
	Regime       string    `json:"regime"`
	ModelVersion string    `json:"model_version"`
}

// RiskMetrics contains current risk measurements
type RiskMetrics struct {
	CurrentDrawdownPct     float64 `json:"current_drawdown_pct"`
	MaxDrawdownPct         float64 `json:"max_drawdown_pct"`
	DailyLossPct           float64 `json:"daily_loss_pct"`
	ConsecutiveLosses      int     `json:"consecutive_losses"`
	OpenPositionValue      float64 `json:"open_position_value"`
	ExposurePct            float64 `json:"exposure_pct"`
	ShouldHalt             bool    `json:"should_halt"`
	HaltReason             string  `json:"halt_reason"`
}
