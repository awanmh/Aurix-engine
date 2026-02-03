// Package binance provides WebSocket client for Binance Futures.
package binance

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"aurix/internal/types"
)

// KlineMessage represents a kline/candlestick message from WebSocket
// Note: Binance API may send prices as either string or number depending on endpoint
type KlineMessage struct {
	EventType string `json:"e"`
	EventTime int64  `json:"E"`
	Symbol    string `json:"s"`
	Kline     struct {
		StartTime    int64       `json:"t"`
		CloseTime    int64       `json:"T"`
		Symbol       string      `json:"s"`
		Interval     string      `json:"i"`
		Open         json.Number `json:"o"`
		Close        json.Number `json:"c"`
		High         json.Number `json:"h"`
		Low          json.Number `json:"l"`
		Volume       json.Number `json:"v"`
		TradeCount   int         `json:"n"`
		IsClosed     bool        `json:"x"`
	} `json:"k"`
}

// WebSocketClient manages WebSocket connection to Binance
type WebSocketClient struct {
	url              string
	conn             *websocket.Conn
	mu               sync.RWMutex
	isConnected      bool
	reconnectDelay   time.Duration
	maxReconnects    int
	pingInterval     time.Duration
	
	// Callbacks
	onCandle         func(candle types.Candle)
	onError          func(err error)
	onReconnect      func(attempt int)
	
	// Control
	done             chan struct{}
	subscriptions    []string
}

// NewWebSocketClient creates a new WebSocket client
func NewWebSocketClient(
	url string,
	reconnectDelayMs int,
	maxReconnects int,
	pingIntervalSeconds int,
) *WebSocketClient {
	return &WebSocketClient{
		url:            url,
		reconnectDelay: time.Duration(reconnectDelayMs) * time.Millisecond,
		maxReconnects:  maxReconnects,
		pingInterval:   time.Duration(pingIntervalSeconds) * time.Second,
		done:           make(chan struct{}),
		subscriptions:  make([]string, 0),
	}
}

// OnCandle sets the callback for new candles
func (c *WebSocketClient) OnCandle(handler func(candle types.Candle)) {
	c.onCandle = handler
}

// OnError sets the callback for errors
func (c *WebSocketClient) OnError(handler func(err error)) {
	c.onError = handler
}

// OnReconnect sets the callback for reconnection attempts
func (c *WebSocketClient) OnReconnect(handler func(attempt int)) {
	c.onReconnect = handler
}

// Subscribe adds a subscription to a symbol/timeframe stream
func (c *WebSocketClient) Subscribe(symbol, interval string) {
	stream := fmt.Sprintf("%s@kline_%s", symbol, interval)
	c.subscriptions = append(c.subscriptions, stream)
}

// Connect establishes the WebSocket connection
func (c *WebSocketClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if len(c.subscriptions) == 0 {
		return fmt.Errorf("no subscriptions configured")
	}
	
	// Build stream URL
	url := c.url
	for i, stream := range c.subscriptions {
		if i == 0 {
			url += "/" + stream
		} else {
			url += "/" + stream
		}
	}
	
	log.Printf("Connecting to WebSocket: %s", url)
	
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	
	c.conn = conn
	c.isConnected = true
	
	// Start goroutines
	go c.readLoop()
	go c.pingLoop()
	
	log.Println("WebSocket connected successfully")
	return nil
}

// Close closes the WebSocket connection
func (c *WebSocketClient) Close() error {
	close(c.done)
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.conn != nil {
		err := c.conn.WriteMessage(websocket.CloseMessage, 
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		if err != nil {
			log.Printf("Error sending close message: %v", err)
		}
		c.conn.Close()
		c.isConnected = false
	}
	
	return nil
}

// IsConnected returns true if WebSocket is connected
func (c *WebSocketClient) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.isConnected
}

// readLoop continuously reads messages from WebSocket
func (c *WebSocketClient) readLoop() {
	for {
		select {
		case <-c.done:
			return
		default:
		}
		
		c.mu.RLock()
		conn := c.conn
		c.mu.RUnlock()
		
		if conn == nil {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		
		_, message, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			
			if c.onError != nil {
				c.onError(err)
			}
			
			c.handleDisconnect()
			continue
		}
		
		c.handleMessage(message)
	}
}

// handleMessage processes incoming WebSocket messages
func (c *WebSocketClient) handleMessage(message []byte) {
	var kline KlineMessage
	if err := json.Unmarshal(message, &kline); err != nil {
		log.Printf("Failed to parse message: %v", err)
		return
	}
	
	// Only process kline events
	if kline.EventType != "kline" {
		return
	}
	
	// Only emit on closed candles
	if !kline.Kline.IsClosed {
		return
	}
	
	// Parse price values
	open := parseNumber(kline.Kline.Open)
	high := parseNumber(kline.Kline.High)
	low := parseNumber(kline.Kline.Low)
	close := parseNumber(kline.Kline.Close)
	volume := parseNumber(kline.Kline.Volume)
	
	candle := types.Candle{
		Symbol:    kline.Symbol,
		Timeframe: kline.Kline.Interval,
		OpenTime:  kline.Kline.StartTime,
		Open:      open,
		High:      high,
		Low:       low,
		Close:     close,
		Volume:    volume,
		CloseTime: kline.Kline.CloseTime,
		Timestamp: time.Now(),
	}
	
	if c.onCandle != nil {
		c.onCandle(candle)
	}
}

// handleDisconnect manages reconnection logic
func (c *WebSocketClient) handleDisconnect() {
	c.mu.Lock()
	c.isConnected = false
	c.conn = nil
	c.mu.Unlock()
	
	for attempt := 1; attempt <= c.maxReconnects; attempt++ {
		select {
		case <-c.done:
			return
		default:
		}
		
		log.Printf("Reconnecting... attempt %d/%d", attempt, c.maxReconnects)
		
		if c.onReconnect != nil {
			c.onReconnect(attempt)
		}
		
		time.Sleep(c.reconnectDelay)
		
		if err := c.Connect(); err != nil {
			log.Printf("Reconnect failed: %v", err)
			continue
		}
		
		log.Println("Reconnected successfully")
		return
	}
	
	log.Println("Max reconnection attempts reached")
	if c.onError != nil {
		c.onError(fmt.Errorf("max reconnection attempts reached"))
	}
}

// pingLoop sends periodic pings to keep connection alive
func (c *WebSocketClient) pingLoop() {
	ticker := time.NewTicker(c.pingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.done:
			return
		case <-ticker.C:
			c.mu.RLock()
			conn := c.conn
			c.mu.RUnlock()
			
			if conn != nil {
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					log.Printf("Ping failed: %v", err)
				}
			}
		}
	}
}

// parseNumber converts json.Number to float64
func parseNumber(n json.Number) float64 {
	f, err := n.Float64()
	if err != nil {
		return 0
	}
	return f
}
