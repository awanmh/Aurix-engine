// AURIX Order Executor
//
// Receives trading signals from Python ML service via Redis,
// executes orders on Binance Futures with risk management.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"aurix/internal/binance"
	"aurix/internal/db"
	"aurix/internal/redis"
	"aurix/internal/risk"
	"aurix/internal/types"
	"aurix/pkg/config"
)

type Executor struct {
	config      *config.Config
	db          *db.Client
	redis       *redis.Client
	exchange    *binance.RESTClient
	riskManager *risk.Manager
}

func main() {
	// Parse command line flags
	configPath := flag.String("config", "config/config.yaml", "Path to config file")
	flag.Parse()

	log.Println("===========================================")
	log.Println("          AURIX Order Executor")
	log.Println("===========================================")

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if cfg.Exchange.Testnet {
		log.Println("⚠️  RUNNING ON TESTNET")
	} else {
		log.Println("🔴 RUNNING ON MAINNET - REAL MONEY!")
	}
	log.Printf("Symbol: %s", cfg.Trading.Symbol)
	log.Printf("Leverage: %dx", cfg.Trading.Leverage)

	// Initialize database
	database, err := db.NewClient(cfg.Database.Path)
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

	// Initialize exchange client
	exchange := binance.NewRESTClient(
		cfg.GetRESTURL(),
		cfg.Exchange.APIKey,
		cfg.Exchange.APISecret,
	)

	// Set leverage and margin type
	if err := exchange.SetLeverage(cfg.Trading.Symbol, cfg.Trading.Leverage); err != nil {
		log.Printf("Warning: Failed to set leverage: %v", err)
	}
	if err := exchange.SetMarginType(cfg.Trading.Symbol, cfg.Trading.MarginType); err != nil {
		log.Printf("Warning: Failed to set margin type: %v", err)
	}

	// Initialize risk manager
	riskManager := risk.NewManager(cfg.Risk, cfg.Risk.InitialCapital)

	// Set up halt callback
	riskManager.OnHalt(func(reason string) {
		log.Printf("🚨 KILL SWITCH TRIGGERED: %s", reason)
		
		// Close all positions
		exchange.CancelAllOrders(cfg.Trading.Symbol)
		
		// Notify via Redis
		redisClient.PublishControl(cfg.Redis.ChannelControl, "HALT", reason)
		
		// Log to database
		database.LogEvent("KILL_SWITCH", "CRITICAL", reason, "")
	})

	// Create executor
	executor := &Executor{
		config:      cfg,
		db:          database,
		redis:       redisClient,
		exchange:    exchange,
		riskManager: riskManager,
	}

	// Subscribe to signals
	redisClient.Subscribe(cfg.Redis.ChannelSignals, executor.handleSignal)
	
	// Subscribe to control commands
	redisClient.Subscribe(cfg.Redis.ChannelControl, executor.handleControl)

	// Start heartbeat
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			// Check Redis connection
			if !redisClient.CheckConnection() {
				database.LogEvent("REDIS_DISCONNECT", "WARNING", "Redis connection lost", "")
			}
			
			// Send heartbeat
			redisClient.PublishHeartbeat(cfg.Redis.ChannelHeartbeat, "executor", "alive")
			
			// Update equity from exchange
			executor.updateEquity()
		}
	}()

	// Start daily reset goroutine
	go func() {
		for {
			// Wait until midnight UTC
			now := time.Now().UTC()
			next := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
			time.Sleep(time.Until(next))
			
			riskManager.ResetDaily()
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Println("🚀 Executor running. Press Ctrl+C to stop.")

	<-sigChan
	log.Println("\n⏹️ Shutting down...")

	// Cancel all open orders on shutdown
	if err := exchange.CancelAllOrders(cfg.Trading.Symbol); err != nil {
		log.Printf("Warning: Failed to cancel orders: %v", err)
	}

	log.Println("Executor stopped")
}

// handleSignal processes incoming trading signals
func (e *Executor) handleSignal(channel string, message []byte) {
	var signal types.Signal
	if err := json.Unmarshal(message, &signal); err != nil {
		log.Printf("Failed to parse signal: %v", err)
		return
	}

	// Ignore non-trading signals
	if signal.Type != "OPEN" && signal.Type != "CLOSE" {
		return
	}

	log.Printf("📨 Signal: %s %s %.1f%% @ %.2f", 
		signal.Type, signal.Direction, signal.Confidence*100, signal.EntryPrice)

	// Check if trading is allowed
	canTrade, reason := e.riskManager.CanTrade()
	if !canTrade {
		log.Printf("⛔ Trading blocked: %s", reason)
		return
	}

	// Execute based on signal type
	switch signal.Type {
	case "OPEN":
		e.openPosition(signal)
	case "CLOSE":
		e.closePosition(signal)
	}
}

// openPosition opens a new position
func (e *Executor) openPosition(signal types.Signal) {
	// Check for existing position
	position, err := e.exchange.GetPosition(signal.Symbol)
	if err != nil {
		log.Printf("Failed to check position: %v", err)
		return
	}
	if position != nil {
		log.Printf("Position already open: %s %.4f @ %.2f", 
			position.Direction, position.Quantity, position.EntryPrice)
		return
	}

	// Calculate position size
	quantity := e.riskManager.CalculatePositionSize(signal.EntryPrice, signal.StopLoss)
	if quantity <= 0 {
		log.Println("Invalid position size")
		return
	}

	// Determine order side
	side := "BUY"
	positionSide := "LONG"
	if signal.Direction == "SHORT" {
		side = "SELL"
		positionSide = "SHORT"
	}

	log.Printf("📈 Opening %s position: %.4f @ market", signal.Direction, quantity)

	// Place market order
	order, err := e.exchange.PlaceMarketOrder(signal.Symbol, side, quantity, positionSide)
	if err != nil {
		log.Printf("❌ Order failed: %v", err)
		e.db.LogEvent("ORDER_FAILED", "ERROR", err.Error(), "")
		return
	}

	log.Printf("✅ Order filled: %s @ %.2f", order.OrderID, parseFloat(order.AvgPrice))

	// Place stop-loss
	stopSide := "SELL"
	if signal.Direction == "SHORT" {
		stopSide = "BUY"
	}
	
	_, err = e.exchange.PlaceStopLoss(signal.Symbol, stopSide, quantity, signal.StopLoss, positionSide)
	if err != nil {
		log.Printf("Warning: Failed to place stop-loss: %v", err)
	}

	// Place take-profit
	_, err = e.exchange.PlaceTakeProfit(signal.Symbol, stopSide, quantity, signal.TakeProfit, positionSide)
	if err != nil {
		log.Printf("Warning: Failed to place take-profit: %v", err)
	}

	// Record trade
	trade := types.Trade{
		Symbol:       signal.Symbol,
		OrderID:      fmt.Sprintf("%d", order.OrderID),
		Direction:    signal.Direction,
		EntryTime:    time.Now(),
		EntryPrice:   parseFloat(order.AvgPrice),
		Quantity:     quantity,
		Status:       "OPEN",
		Confidence:   signal.Confidence,
		Regime:       signal.Regime,
		ModelVersion: signal.ModelVersion,
	}
	e.db.InsertTrade(trade)

	// Notify via Redis
	e.redis.PublishWithTimestamp(e.config.Redis.ChannelSignals, map[string]interface{}{
		"type":        "TRADE_OPENED",
		"order_id":    order.OrderID,
		"direction":   signal.Direction,
		"quantity":    quantity,
		"entry_price": parseFloat(order.AvgPrice),
	})
}

// closePosition closes an existing position
func (e *Executor) closePosition(signal types.Signal) {
	position, err := e.exchange.GetPosition(signal.Symbol)
	if err != nil {
		log.Printf("Failed to check position: %v", err)
		return
	}
	if position == nil {
		log.Println("No position to close")
		return
	}

	// Cancel existing SL/TP orders
	e.exchange.CancelAllOrders(signal.Symbol)

	// Determine close side
	side := "SELL"
	positionSide := "LONG"
	if position.Direction == "SHORT" {
		side = "BUY"
		positionSide = "SHORT"
	}

	log.Printf("📉 Closing %s position: %.4f @ market", position.Direction, position.Quantity)

	// Place market close order
	order, err := e.exchange.PlaceMarketOrder(signal.Symbol, side, position.Quantity, positionSide)
	if err != nil {
		log.Printf("❌ Close order failed: %v", err)
		return
	}

	exitPrice := parseFloat(order.AvgPrice)
	
	// Calculate PnL
	var grossPnL float64
	if position.Direction == "LONG" {
		grossPnL = (exitPrice - position.EntryPrice) * position.Quantity
	} else {
		grossPnL = (position.EntryPrice - exitPrice) * position.Quantity
	}
	
	// Estimate fees
	fees := (position.EntryPrice + exitPrice) * position.Quantity * 0.0004 // 0.04% taker fee
	netPnL := grossPnL - fees

	log.Printf("✅ Position closed @ %.2f | PnL: $%.2f (net: $%.2f)", exitPrice, grossPnL, netPnL)

	// Update risk manager
	e.riskManager.RecordTrade(netPnL)

	// Update trade record
	// Note: In production, you'd look up the trade by position/order ID
	
	// Notify via Redis
	e.redis.PublishWithTimestamp(e.config.Redis.ChannelSignals, map[string]interface{}{
		"type":       "TRADE_COMPLETE",
		"direction":  position.Direction,
		"entry":      position.EntryPrice,
		"exit":       exitPrice,
		"gross_pnl":  grossPnL,
		"net_pnl":    netPnL,
		"confidence": signal.Confidence,
	})
}

// handleControl processes control commands
func (e *Executor) handleControl(channel string, message []byte) {
	var cmd types.ControlCommand
	if err := json.Unmarshal(message, &cmd); err != nil {
		return
	}

	log.Printf("🎮 Control command: %s", cmd.Command)

	switch cmd.Command {
	case "HALT":
		e.riskManager.UpdateEquity(0) // Force halt
	case "RESUME":
		e.riskManager.Resume()
	case "CLOSE_ALL":
		e.exchange.CancelAllOrders(e.config.Trading.Symbol)
		// Also close any open position
		if pos, _ := e.exchange.GetPosition(e.config.Trading.Symbol); pos != nil {
			e.closePosition(types.Signal{Symbol: e.config.Trading.Symbol})
		}
	}
}

// updateEquity fetches current equity from exchange
func (e *Executor) updateEquity() {
	info, err := e.exchange.GetAccountInfo()
	if err != nil {
		log.Printf("Failed to get account info: %v", err)
		return
	}

	equity := parseFloat(info.TotalWalletBalance) + parseFloat(info.TotalUnrealizedProfit)
	metrics := e.riskManager.UpdateEquity(equity)

	// Save to database
	state := e.riskManager.GetState()
	state.AvailableBalance = parseFloat(info.AvailableBalance)
	state.UnrealizedPnL = parseFloat(info.TotalUnrealizedProfit)
	e.db.SaveAccountState(state)

	// Log if approaching limits
	if metrics.CurrentDrawdownPct > metrics.MaxDrawdownPct*0.7 {
		log.Printf("⚠️ Drawdown warning: %.2f%%", metrics.CurrentDrawdownPct)
	}
}

// parseFloat converts string to float64
func parseFloat(s string) float64 {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}
