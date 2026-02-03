// Package redis provides pub/sub client for inter-service communication.
package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/go-redis/redis/v8"
)

// Client handles Redis pub/sub operations
type Client struct {
	client *redis.Client
	ctx    context.Context
	cancel context.CancelFunc
}

// NewClient creates a new Redis client
func NewClient(host string, port int, password string, db int) (*Client, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	rdb := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", host, port),
		Password: password,
		DB:       db,
	})
	
	// Test connection
	if err := rdb.Ping(ctx).Err(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}
	
	return &Client{
		client: rdb,
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

// Close closes the Redis connection
func (c *Client) Close() error {
	c.cancel()
	return c.client.Close()
}

// Publish publishes a message to a channel
func (c *Client) Publish(channel string, data interface{}) error {
	message, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}
	
	return c.client.Publish(c.ctx, channel, message).Err()
}

// PublishWithTimestamp publishes a message with timestamp
func (c *Client) PublishWithTimestamp(channel string, data map[string]interface{}) error {
	data["_timestamp"] = time.Now().Format(time.RFC3339)
	return c.Publish(channel, data)
}

// Subscribe subscribes to a channel and calls handler for each message
func (c *Client) Subscribe(channel string, handler func(channel string, message []byte)) {
	pubsub := c.client.Subscribe(c.ctx, channel)
	
	go func() {
		ch := pubsub.Channel()
		
		for {
			select {
			case <-c.ctx.Done():
				pubsub.Close()
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				handler(msg.Channel, []byte(msg.Payload))
			}
		}
	}()
}

// SubscribeMultiple subscribes to multiple channels
func (c *Client) SubscribeMultiple(channels []string, handler func(channel string, message []byte)) {
	pubsub := c.client.Subscribe(c.ctx, channels...)
	
	go func() {
		ch := pubsub.Channel()
		
		for {
			select {
			case <-c.ctx.Done():
				pubsub.Close()
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				handler(msg.Channel, []byte(msg.Payload))
			}
		}
	}()
}

// Set sets a key-value pair with optional expiration
func (c *Client) Set(key string, value interface{}, expiration time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return c.client.Set(c.ctx, key, data, expiration).Err()
}

// Get gets a value by key
func (c *Client) Get(key string) (string, error) {
	return c.client.Get(c.ctx, key).Result()
}

// GetJSON gets and unmarshal JSON value
func (c *Client) GetJSON(key string, dest interface{}) error {
	data, err := c.client.Get(c.ctx, key).Bytes()
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dest)
}

// Delete deletes a key
func (c *Client) Delete(key string) error {
	return c.client.Del(c.ctx, key).Err()
}

// PublishHeartbeat publishes a service heartbeat
func (c *Client) PublishHeartbeat(channel, serviceName, status string) error {
	return c.PublishWithTimestamp(channel, map[string]interface{}{
		"service": serviceName,
		"status":  status,
	})
}

// PublishSignal publishes a trading signal
func (c *Client) PublishSignal(channel string, signalType, symbol, direction string, 
	confidence, entryPrice, takeProfit, stopLoss, quantity float64, 
	regime, modelVersion string) error {
	
	return c.PublishWithTimestamp(channel, map[string]interface{}{
		"type":          signalType,
		"symbol":        symbol,
		"direction":     direction,
		"confidence":    confidence,
		"entry_price":   entryPrice,
		"take_profit":   takeProfit,
		"stop_loss":     stopLoss,
		"quantity":      quantity,
		"regime":        regime,
		"model_version": modelVersion,
	})
}

// PublishControl publishes a control command
func (c *Client) PublishControl(channel, command, reason string) error {
	return c.PublishWithTimestamp(channel, map[string]interface{}{
		"command": command,
		"reason":  reason,
	})
}

// CheckConnection checks if Redis is connected
func (c *Client) CheckConnection() bool {
	err := c.client.Ping(c.ctx).Err()
	if err != nil {
		log.Printf("Redis connection check failed: %v", err)
		return false
	}
	return true
}
