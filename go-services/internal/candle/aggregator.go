// Package candle provides multi-timeframe candle aggregation.
package candle

import (
	"sync"
	"time"

	"aurix/internal/types"
)

// Aggregator handles candle aggregation across multiple timeframes
type Aggregator struct {
	mu        sync.RWMutex
	symbol    string
	candles   map[string][]types.Candle // timeframe -> candles
	buffers   map[string]*CandleBuffer  // for aggregating higher timeframes
	maxCandles int
}

// CandleBuffer accumulates base candles to form higher timeframe candles
type CandleBuffer struct {
	BaseTimeframe   string
	TargetTimeframe string
	Multiplier      int
	CurrentCandle   *types.Candle
	CandleCount     int
}

// NewAggregator creates a new candle aggregator
func NewAggregator(symbol string, maxCandles int) *Aggregator {
	return &Aggregator{
		symbol:     symbol,
		candles:    make(map[string][]types.Candle),
		buffers:    make(map[string]*CandleBuffer),
		maxCandles: maxCandles,
	}
}

// AddBuffer adds a higher timeframe aggregation buffer
func (a *Aggregator) AddBuffer(base, target string, multiplier int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	a.buffers[target] = &CandleBuffer{
		BaseTimeframe:   base,
		TargetTimeframe: target,
		Multiplier:      multiplier,
	}
}

// AddCandle adds a new candle and aggregates to higher timeframes
func (a *Aggregator) AddCandle(candle types.Candle) []types.Candle {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	// Add to storage for this timeframe
	tf := candle.Timeframe
	a.candles[tf] = append(a.candles[tf], candle)
	
	// Trim if too many
	if len(a.candles[tf]) > a.maxCandles {
		a.candles[tf] = a.candles[tf][len(a.candles[tf])-a.maxCandles:]
	}
	
	// Check for higher timeframe aggregation
	var completedCandles []types.Candle
	completedCandles = append(completedCandles, candle)
	
	for targetTF, buffer := range a.buffers {
		if buffer.BaseTimeframe != candle.Timeframe {
			continue
		}
		
		if completed := a.aggregateCandle(buffer, candle); completed != nil {
			a.candles[targetTF] = append(a.candles[targetTF], *completed)
			if len(a.candles[targetTF]) > a.maxCandles {
				a.candles[targetTF] = a.candles[targetTF][len(a.candles[targetTF])-a.maxCandles:]
			}
			completedCandles = append(completedCandles, *completed)
		}
	}
	
	return completedCandles
}

// aggregateCandle accumulates base candles into a higher timeframe candle
func (a *Aggregator) aggregateCandle(buffer *CandleBuffer, candle types.Candle) *types.Candle {
	if buffer.CurrentCandle == nil {
		// Start new aggregated candle
		buffer.CurrentCandle = &types.Candle{
			Symbol:    candle.Symbol,
			Timeframe: buffer.TargetTimeframe,
			OpenTime:  candle.OpenTime,
			Open:      candle.Open,
			High:      candle.High,
			Low:       candle.Low,
			Close:     candle.Close,
			Volume:    candle.Volume,
			CloseTime: candle.CloseTime,
			Timestamp: time.Now(),
		}
		buffer.CandleCount = 1
		return nil
	}
	
	// Update aggregated candle
	buffer.CurrentCandle.High = max(buffer.CurrentCandle.High, candle.High)
	buffer.CurrentCandle.Low = min(buffer.CurrentCandle.Low, candle.Low)
	buffer.CurrentCandle.Close = candle.Close
	buffer.CurrentCandle.CloseTime = candle.CloseTime
	buffer.CurrentCandle.Volume += candle.Volume
	buffer.CandleCount++
	
	// Check if complete
	if buffer.CandleCount >= buffer.Multiplier {
		completed := *buffer.CurrentCandle
		buffer.CurrentCandle = nil
		buffer.CandleCount = 0
		return &completed
	}
	
	return nil
}

// GetCandles returns candles for a timeframe
func (a *Aggregator) GetCandles(timeframe string, limit int) []types.Candle {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	candles := a.candles[timeframe]
	if len(candles) == 0 {
		return nil
	}
	
	if limit >= len(candles) {
		result := make([]types.Candle, len(candles))
		copy(result, candles)
		return result
	}
	
	result := make([]types.Candle, limit)
	copy(result, candles[len(candles)-limit:])
	return result
}

// GetLatestCandle returns the most recent candle for a timeframe
func (a *Aggregator) GetLatestCandle(timeframe string) *types.Candle {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	candles := a.candles[timeframe]
	if len(candles) == 0 {
		return nil
	}
	
	result := candles[len(candles)-1]
	return &result
}

// Count returns the number of candles for a timeframe
func (a *Aggregator) Count(timeframe string) int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.candles[timeframe])
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
