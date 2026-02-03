// AURIX Market Data Collector
//
// Collects real-time candle data from Binance Futures via WebSocket,
// aggregates to multiple timeframes, and stores in SQLite.
package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"aurix/internal/binance"
	"aurix/internal/candle"
	"aurix/internal/db"
	"aurix/internal/redis"
	"aurix/internal/types"
	"aurix/pkg/config"
)

func main() {
	// Parse command line flags
	configPath := flag.String("config", "config/config.yaml", "Path to config file")
	flag.Parse()

	log.Println("===========================================")
	log.Println("       AURIX Market Data Collector")
	log.Println("===========================================")

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	log.Printf("Symbol: %s", cfg.Trading.Symbol)
	log.Printf("Timeframes: %v", cfg.Trading.Timeframes)
	log.Printf("Testnet: %v", cfg.Exchange.Testnet)

	// Resolve database path relative to config file directory
	dbPath := cfg.Database.Path
	if !filepath.IsAbs(dbPath) {
		configDir := filepath.Dir(*configPath)
		dbPath = filepath.Join(configDir, "..", dbPath)
	}
	
	// Ensure data directory exists
	dbDir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		log.Fatalf("Failed to create database directory: %v", err)
	}
	
	log.Printf("Database path: %s", dbPath)
	
	// Initialize database
	database, err := db.NewClient(dbPath)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer database.Close()
	log.Println("✅ Database connected")

	// Initialize Redis
	redisClient, err := redis.NewClient(
		cfg.Redis.Host,
		cfg.Redis.Port,
		cfg.Redis.Password,
		cfg.Redis.DB,
	)
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	defer redisClient.Close()
	log.Println("✅ Redis connected")

	// Initialize candle aggregator
	aggregator := candle.NewAggregator(cfg.Trading.Symbol, 1000)
	
	// Add higher timeframe aggregation buffers
	// 15m = 15 x 1m candles, 1h = 60 x 1m candles
	aggregator.AddBuffer("1m", "15m", 15)
	aggregator.AddBuffer("1m", "1h", 60)

	// Initialize WebSocket client
	wsClient := binance.NewWebSocketClient(
		cfg.GetWebSocketURL(),
		cfg.Exchange.ReconnectDelayMs,
		cfg.Exchange.MaxReconnectAttempts,
		cfg.Exchange.PingIntervalSeconds,
	)

	// Subscribe to kline streams
	symbol := strings.ToLower(cfg.Trading.Symbol)
	for _, tf := range cfg.Trading.Timeframes {
		wsClient.Subscribe(symbol, tf)
		log.Printf("Subscribed to %s@kline_%s", symbol, tf)
	}

	// Handle incoming candles
	wsClient.OnCandle(func(c types.Candle) {
		log.Printf("📊 Candle: %s %s O:%.2f H:%.2f L:%.2f C:%.2f V:%.0f",
			c.Symbol, c.Timeframe, c.Open, c.High, c.Low, c.Close, c.Volume)

		// Aggregate and store
		completedCandles := aggregator.AddCandle(c)
		
		for _, completed := range completedCandles {
			// Save to database
			if err := database.InsertCandle(completed); err != nil {
				log.Printf("Failed to save candle: %v", err)
			}

			// Publish to Redis for Python service
			if completed.Timeframe == "15m" {
				redisClient.PublishWithTimestamp(cfg.Redis.ChannelSignals, map[string]interface{}{
					"type":      "NEW_CANDLE",
					"symbol":    completed.Symbol,
					"timeframe": completed.Timeframe,
					"open":      completed.Open,
					"high":      completed.High,
					"low":       completed.Low,
					"close":     completed.Close,
					"volume":    completed.Volume,
					"open_time": completed.OpenTime,
				})
			}
		}
	})

	// Handle errors
	wsClient.OnError(func(err error) {
		log.Printf("❌ WebSocket error: %v", err)
		database.LogEvent("WEBSOCKET_ERROR", "ERROR", err.Error(), "")
	})

	// Handle reconnections
	wsClient.OnReconnect(func(attempt int) {
		log.Printf("🔄 Reconnecting... attempt %d", attempt)
		database.LogEvent("WEBSOCKET_RECONNECT", "WARNING", 
			"Reconnection attempt", 
			`{"attempt": ` + string(rune(attempt)) + `}`)
	})

	// Connect to WebSocket
	if err := wsClient.Connect(); err != nil {
		log.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	log.Println("✅ WebSocket connected")

	// Start heartbeat
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			redisClient.PublishHeartbeat(cfg.Redis.ChannelHeartbeat, "collector", "alive")
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Println("🚀 Collector running. Press Ctrl+C to stop.")

	<-sigChan
	log.Println("\n⏹️ Shutting down...")

	wsClient.Close()
	log.Println("Collector stopped")
}
