// Package config provides configuration loading for AURIX services.
package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config represents the complete AURIX configuration
type Config struct {
	Exchange   ExchangeConfig   `yaml:"exchange"`
	Trading    TradingConfig    `yaml:"trading"`
	Risk       RiskConfig       `yaml:"risk"`
	Database   DatabaseConfig   `yaml:"database"`
	Redis      RedisConfig      `yaml:"redis"`
	Logging    LoggingConfig    `yaml:"logging"`
}

// ExchangeConfig contains exchange-specific settings
type ExchangeConfig struct {
	Name                   string `yaml:"name"`
	Testnet                bool   `yaml:"testnet"`
	APIKey                 string `yaml:"api_key"`
	APISecret              string `yaml:"api_secret"`
	ReconnectDelayMs       int    `yaml:"websocket_reconnect_delay_ms"`
	MaxReconnectAttempts   int    `yaml:"websocket_max_reconnect_attempts"`
	PingIntervalSeconds    int    `yaml:"websocket_ping_interval_seconds"`
	OrdersPerSecond        int    `yaml:"orders_per_second"`
	OrdersPerMinute        int    `yaml:"orders_per_minute"`
}

// TradingConfig contains trading parameters
type TradingConfig struct {
	Symbol     string   `yaml:"symbol"`
	Leverage   int      `yaml:"leverage"`
	MarginType string   `yaml:"margin_type"`
	Timeframes []string `yaml:"timeframes"`
}

// RiskConfig contains risk management settings
type RiskConfig struct {
	InitialCapital             float64 `yaml:"initial_capital"`
	RiskPerTradePercent        float64 `yaml:"risk_per_trade_percent"`
	MaxPositionSizePercent     float64 `yaml:"max_position_size_percent"`
	MaxDailyLossPercent        float64 `yaml:"max_daily_loss_percent"`
	MaxDrawdownPercent         float64 `yaml:"max_drawdown_percent"`
	MaxConsecutiveLosses       int     `yaml:"max_consecutive_losses"`
	PauseAfterConsecutiveLosses int    `yaml:"pause_after_consecutive_losses"`
	PauseDurationMinutes       int     `yaml:"pause_duration_minutes"`
	CooldownAfterLossMinutes   int     `yaml:"cooldown_after_loss_minutes"`
	MaxSlippagePercent         float64 `yaml:"max_slippage_percent"`
	ExpectedSlippageBps        int     `yaml:"expected_slippage_bps"`
}

// DatabaseConfig contains database settings
type DatabaseConfig struct {
	Type                string `yaml:"type"`
	Path                string `yaml:"path"`
	BackupEnabled       bool   `yaml:"backup_enabled"`
	BackupIntervalHours int    `yaml:"backup_interval_hours"`
}

// RedisConfig contains Redis settings
type RedisConfig struct {
	Host               string `yaml:"host"`
	Port               int    `yaml:"port"`
	Password           string `yaml:"password"`
	DB                 int    `yaml:"db"`
	ChannelSignals     string `yaml:"channel_signals"`
	ChannelHeartbeat   string `yaml:"channel_heartbeat"`
	ChannelControl     string `yaml:"channel_control"`
	PublishTimeoutMs   int    `yaml:"publish_timeout_ms"`
	SubscribeTimeoutMs int    `yaml:"subscribe_timeout_ms"`
}

// LoggingConfig contains logging settings
type LoggingConfig struct {
	Level       string `yaml:"level"`
	Format      string `yaml:"format"`
	File        string `yaml:"file"`
	MaxSizeMB   int    `yaml:"max_size_mb"`
	BackupCount int    `yaml:"backup_count"`
}

// Load reads configuration from a YAML file
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	// Set defaults
	cfg.SetDefaults()

	return &cfg, nil
}

// SetDefaults sets default values for unspecified fields
func (c *Config) SetDefaults() {
	if c.Exchange.Name == "" {
		c.Exchange.Name = "binance"
	}
	if c.Exchange.ReconnectDelayMs == 0 {
		c.Exchange.ReconnectDelayMs = 5000
	}
	if c.Exchange.MaxReconnectAttempts == 0 {
		c.Exchange.MaxReconnectAttempts = 10
	}
	if c.Exchange.PingIntervalSeconds == 0 {
		c.Exchange.PingIntervalSeconds = 30
	}
	
	if c.Trading.Symbol == "" {
		c.Trading.Symbol = "BTCUSDT"
	}
	if c.Trading.Leverage == 0 {
		c.Trading.Leverage = 5
	}
	if c.Trading.MarginType == "" {
		c.Trading.MarginType = "CROSSED"
	}

	if c.Risk.InitialCapital == 0 {
		c.Risk.InitialCapital = 10000
	}
	if c.Risk.RiskPerTradePercent == 0 {
		c.Risk.RiskPerTradePercent = 1.0
	}
	if c.Risk.MaxPositionSizePercent == 0 {
		c.Risk.MaxPositionSizePercent = 5.0
	}
	if c.Risk.MaxDailyLossPercent == 0 {
		c.Risk.MaxDailyLossPercent = 3.0
	}
	if c.Risk.MaxDrawdownPercent == 0 {
		c.Risk.MaxDrawdownPercent = 5.0
	}
	if c.Risk.MaxConsecutiveLosses == 0 {
		c.Risk.MaxConsecutiveLosses = 8
	}

	if c.Database.Type == "" {
		c.Database.Type = "sqlite"
	}
	if c.Database.Path == "" {
		c.Database.Path = "data/aurix.db"
	}

	if c.Redis.Host == "" {
		c.Redis.Host = "localhost"
	}
	if c.Redis.Port == 0 {
		c.Redis.Port = 6379
	}
	if c.Redis.ChannelSignals == "" {
		c.Redis.ChannelSignals = "aurix:signals"
	}
	if c.Redis.ChannelHeartbeat == "" {
		c.Redis.ChannelHeartbeat = "aurix:heartbeat"
	}
	if c.Redis.ChannelControl == "" {
		c.Redis.ChannelControl = "aurix:control"
	}

	if c.Logging.Level == "" {
		c.Logging.Level = "INFO"
	}
	if c.Logging.File == "" {
		c.Logging.File = "data/logs/aurix.log"
	}
}

// GetWebSocketURL returns the appropriate WebSocket URL based on testnet setting
func (c *Config) GetWebSocketURL() string {
	if c.Exchange.Testnet {
		return "wss://fstream.binancefuture.com/ws"
	}
	return "wss://fstream.binance.com/ws"
}

// GetRESTURL returns the appropriate REST API URL based on testnet setting
func (c *Config) GetRESTURL() string {
	if c.Exchange.Testnet {
		return "https://testnet.binancefuture.com"
	}
	return "https://fapi.binance.com"
}
