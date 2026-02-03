// Package risk provides risk management and kill-switch functionality.
package risk

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"aurix/internal/types"
	"aurix/pkg/config"
)

// Manager handles risk management and kill-switch logic
type Manager struct {
	mu                sync.RWMutex
	config            config.RiskConfig
	
	// State
	equity            float64
	peakEquity        float64
	dailyStartEquity  float64
	dailyPnL          float64
	consecutiveLosses int
	isHalted          bool
	haltReason        string
	pauseUntil        time.Time
	
	// Trade tracking
	lastTradeTime     time.Time
	recentTrades      []tradeResult
	
	// Callbacks
	onHalt            func(reason string)
	onPause           func(duration time.Duration, reason string)
}

type tradeResult struct {
	pnl      float64
	isWin    bool
	time     time.Time
}

// NewManager creates a new risk manager
func NewManager(cfg config.RiskConfig, initialEquity float64) *Manager {
	return &Manager{
		config:           cfg,
		equity:           initialEquity,
		peakEquity:       initialEquity,
		dailyStartEquity: initialEquity,
		recentTrades:     make([]tradeResult, 0),
	}
}

// OnHalt sets the callback for trading halt
func (m *Manager) OnHalt(handler func(reason string)) {
	m.onHalt = handler
}

// OnPause sets the callback for trading pause
func (m *Manager) OnPause(handler func(duration time.Duration, reason string)) {
	m.onPause = handler
}

// UpdateEquity updates current equity and checks for drawdown limits
func (m *Manager) UpdateEquity(equity float64) *types.RiskMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.equity = equity
	
	// Update peak equity
	if equity > m.peakEquity {
		m.peakEquity = equity
	}
	
	// Calculate metrics
	metrics := m.calculateMetrics()
	
	// Check for halt conditions
	m.checkHaltConditions(metrics)
	
	return metrics
}

// RecordTrade records a trade result and checks consecutive losses
func (m *Manager) RecordTrade(pnl float64) *types.RiskMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	isWin := pnl > 0
	m.recentTrades = append(m.recentTrades, tradeResult{
		pnl:   pnl,
		isWin: isWin,
		time:  time.Now(),
	})
	
	// Keep only recent trades
	if len(m.recentTrades) > 100 {
		m.recentTrades = m.recentTrades[1:]
	}
	
	m.lastTradeTime = time.Now()
	m.dailyPnL += pnl
	
	// Update consecutive losses
	if isWin {
		m.consecutiveLosses = 0
	} else {
		m.consecutiveLosses++
	}
	
	// Calculate metrics
	metrics := m.calculateMetrics()
	
	// Check for halt/pause conditions
	m.checkHaltConditions(metrics)
	m.checkConsecutiveLosses()
	
	return metrics
}

// calculateMetrics calculates current risk metrics
func (m *Manager) calculateMetrics() *types.RiskMetrics {
	// Current drawdown
	currentDD := 0.0
	if m.peakEquity > 0 {
		currentDD = (m.peakEquity - m.equity) / m.peakEquity * 100
	}
	
	// Daily loss
	dailyLoss := 0.0
	if m.dailyStartEquity > 0 && m.dailyPnL < 0 {
		dailyLoss = (-m.dailyPnL / m.dailyStartEquity) * 100
	}
	
	return &types.RiskMetrics{
		CurrentDrawdownPct: currentDD,
		MaxDrawdownPct:     m.config.MaxDrawdownPercent,
		DailyLossPct:       dailyLoss,
		ConsecutiveLosses:  m.consecutiveLosses,
		ShouldHalt:         m.isHalted,
		HaltReason:         m.haltReason,
	}
}

// checkHaltConditions checks if trading should be halted
func (m *Manager) checkHaltConditions(metrics *types.RiskMetrics) {
	if m.isHalted {
		return
	}
	
	// Check max drawdown
	if metrics.CurrentDrawdownPct >= m.config.MaxDrawdownPercent {
		m.halt(fmt.Sprintf("Max drawdown exceeded: %.2f%% >= %.2f%%", 
			metrics.CurrentDrawdownPct, m.config.MaxDrawdownPercent))
		return
	}
	
	// Check daily loss limit
	if metrics.DailyLossPct >= m.config.MaxDailyLossPercent {
		m.halt(fmt.Sprintf("Daily loss limit exceeded: %.2f%% >= %.2f%%",
			metrics.DailyLossPct, m.config.MaxDailyLossPercent))
		return
	}
}

// checkConsecutiveLosses checks for consecutive loss conditions
func (m *Manager) checkConsecutiveLosses() {
	// Hard halt on max consecutive losses
	if m.consecutiveLosses >= m.config.MaxConsecutiveLosses {
		m.halt(fmt.Sprintf("Max consecutive losses: %d", m.consecutiveLosses))
		return
	}
	
	// Pause on threshold
	if m.consecutiveLosses >= m.config.PauseAfterConsecutiveLosses {
		m.pause(
			time.Duration(m.config.PauseDurationMinutes)*time.Minute,
			fmt.Sprintf("Consecutive losses: %d", m.consecutiveLosses),
		)
	}
}

// halt triggers a trading halt
func (m *Manager) halt(reason string) {
	m.isHalted = true
	m.haltReason = reason
	
	log.Printf("🚨 TRADING HALT: %s", reason)
	
	if m.onHalt != nil {
		m.onHalt(reason)
	}
}

// pause temporarily pauses trading
func (m *Manager) pause(duration time.Duration, reason string) {
	m.pauseUntil = time.Now().Add(duration)
	
	log.Printf("⏸️ TRADING PAUSED for %v: %s", duration, reason)
	
	if m.onPause != nil {
		m.onPause(duration, reason)
	}
}

// CanTrade checks if trading is currently allowed
func (m *Manager) CanTrade() (bool, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.isHalted {
		return false, m.haltReason
	}
	
	if time.Now().Before(m.pauseUntil) {
		remaining := time.Until(m.pauseUntil).Round(time.Second)
		return false, fmt.Sprintf("Paused for %v", remaining)
	}
	
	// Check cooldown after loss
	if m.consecutiveLosses > 0 && m.config.CooldownAfterLossMinutes > 0 {
		cooldown := time.Duration(m.config.CooldownAfterLossMinutes) * time.Minute
		if time.Since(m.lastTradeTime) < cooldown {
			remaining := cooldown - time.Since(m.lastTradeTime)
			return false, fmt.Sprintf("Cooldown: %v remaining", remaining.Round(time.Second))
		}
	}
	
	return true, ""
}

// GetState returns current risk state
func (m *Manager) GetState() types.AccountState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	return types.AccountState{
		Equity:            m.equity,
		PeakEquity:        m.peakEquity,
		DailyPnL:          m.dailyPnL,
		CurrentDrawdown:   (m.peakEquity - m.equity) / m.peakEquity * 100,
		ConsecutiveLosses: m.consecutiveLosses,
		IsHalted:          m.isHalted,
		HaltReason:        m.haltReason,
		RecordedAt:        time.Now(),
	}
}

// ResetDaily resets daily counters (call at midnight)
func (m *Manager) ResetDaily() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.dailyStartEquity = m.equity
	m.dailyPnL = 0
	
	log.Println("Daily risk counters reset")
}

// Resume resumes trading after a manual review
func (m *Manager) Resume() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.isHalted = false
	m.haltReason = ""
	m.consecutiveLosses = 0
	m.pauseUntil = time.Time{}
	
	log.Println("✅ Trading resumed")
}

// CalculatePositionSize calculates safe position size
func (m *Manager) CalculatePositionSize(entryPrice, stopLoss float64) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	// Risk amount in USD
	riskAmount := m.equity * (m.config.RiskPerTradePercent / 100)
	
	// Distance to stop loss
	stopDistance := abs(entryPrice - stopLoss)
	if stopDistance == 0 {
		return 0
	}
	
	// Position size based on risk
	positionSize := riskAmount / stopDistance
	
	// Apply max position size limit
	maxPositionValue := m.equity * (m.config.MaxPositionSizePercent / 100)
	maxPositionSize := maxPositionValue / entryPrice
	
	if positionSize > maxPositionSize {
		positionSize = maxPositionSize
	}
	
	return positionSize
}

// ToJSON returns the current state as JSON
func (m *Manager) ToJSON() ([]byte, error) {
	state := m.GetState()
	return json.Marshal(state)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
