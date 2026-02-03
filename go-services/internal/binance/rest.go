// Package binance provides REST API client for Binance Futures.
package binance

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"aurix/internal/types"
)

// RESTClient handles REST API calls to Binance Futures
type RESTClient struct {
	baseURL   string
	apiKey    string
	apiSecret string
	client    *http.Client
}

// NewRESTClient creates a new REST client
func NewRESTClient(baseURL, apiKey, apiSecret string) *RESTClient {
	return &RESTClient{
		baseURL:   baseURL,
		apiKey:    apiKey,
		apiSecret: apiSecret,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// OrderResponse represents the response from order placement
type OrderResponse struct {
	OrderID       int64   `json:"orderId"`
	Symbol        string  `json:"symbol"`
	Status        string  `json:"status"`
	ClientOrderID string  `json:"clientOrderId"`
	Price         string  `json:"price"`
	AvgPrice      string  `json:"avgPrice"`
	OrigQty       string  `json:"origQty"`
	ExecutedQty   string  `json:"executedQty"`
	Type          string  `json:"type"`
	Side          string  `json:"side"`
	PositionSide  string  `json:"positionSide"`
	UpdateTime    int64   `json:"updateTime"`
}

// AccountInfo represents account information
type AccountInfo struct {
	TotalWalletBalance    string `json:"totalWalletBalance"`
	TotalUnrealizedProfit string `json:"totalUnrealizedProfit"`
	TotalMarginBalance    string `json:"totalMarginBalance"`
	AvailableBalance      string `json:"availableBalance"`
	Positions             []struct {
		Symbol           string `json:"symbol"`
		PositionAmt      string `json:"positionAmt"`
		EntryPrice       string `json:"entryPrice"`
		UnrealizedProfit string `json:"unrealizedProfit"`
		PositionSide     string `json:"positionSide"`
	} `json:"positions"`
}

// PlaceMarketOrder places a market order
func (c *RESTClient) PlaceMarketOrder(
	symbol string,
	side string,     // BUY, SELL
	quantity float64,
	positionSide string, // LONG, SHORT
) (*OrderResponse, error) {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("side", side)
	params.Set("type", "MARKET")
	params.Set("quantity", strconv.FormatFloat(quantity, 'f', -1, 64))
	params.Set("positionSide", positionSide)
	
	return c.placeOrder(params)
}

// PlaceStopLoss places a stop-loss order
func (c *RESTClient) PlaceStopLoss(
	symbol string,
	side string,
	quantity float64,
	stopPrice float64,
	positionSide string,
) (*OrderResponse, error) {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("side", side)
	params.Set("type", "STOP_MARKET")
	params.Set("quantity", strconv.FormatFloat(quantity, 'f', -1, 64))
	params.Set("stopPrice", strconv.FormatFloat(stopPrice, 'f', 2, 64))
	params.Set("positionSide", positionSide)
	params.Set("closePosition", "true")
	
	return c.placeOrder(params)
}

// PlaceTakeProfit places a take-profit order
func (c *RESTClient) PlaceTakeProfit(
	symbol string,
	side string,
	quantity float64,
	stopPrice float64,
	positionSide string,
) (*OrderResponse, error) {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("side", side)
	params.Set("type", "TAKE_PROFIT_MARKET")
	params.Set("quantity", strconv.FormatFloat(quantity, 'f', -1, 64))
	params.Set("stopPrice", strconv.FormatFloat(stopPrice, 'f', 2, 64))
	params.Set("positionSide", positionSide)
	params.Set("closePosition", "true")
	
	return c.placeOrder(params)
}

// placeOrder executes the order placement
func (c *RESTClient) placeOrder(params url.Values) (*OrderResponse, error) {
	body, err := c.signedRequest("POST", "/fapi/v1/order", params)
	if err != nil {
		return nil, err
	}
	
	var resp OrderResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return &resp, nil
}

// CancelOrder cancels an existing order
func (c *RESTClient) CancelOrder(symbol string, orderID int64) error {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("orderId", strconv.FormatInt(orderID, 10))
	
	_, err := c.signedRequest("DELETE", "/fapi/v1/order", params)
	return err
}

// CancelAllOrders cancels all open orders for a symbol
func (c *RESTClient) CancelAllOrders(symbol string) error {
	params := url.Values{}
	params.Set("symbol", symbol)
	
	_, err := c.signedRequest("DELETE", "/fapi/v1/allOpenOrders", params)
	return err
}

// GetAccountInfo fetches account information
func (c *RESTClient) GetAccountInfo() (*AccountInfo, error) {
	params := url.Values{}
	
	body, err := c.signedRequest("GET", "/fapi/v2/account", params)
	if err != nil {
		return nil, err
	}
	
	var info AccountInfo
	if err := json.Unmarshal(body, &info); err != nil {
		return nil, fmt.Errorf("failed to parse account info: %w", err)
	}
	
	return &info, nil
}

// GetPosition gets position for a symbol
func (c *RESTClient) GetPosition(symbol string) (*types.Position, error) {
	info, err := c.GetAccountInfo()
	if err != nil {
		return nil, err
	}
	
	for _, pos := range info.Positions {
		if pos.Symbol == symbol && pos.PositionAmt != "0" {
			qty := parseFloat(pos.PositionAmt)
			direction := "LONG"
			if qty < 0 {
				direction = "SHORT"
				qty = -qty
			}
			
			return &types.Position{
				Symbol:        pos.Symbol,
				Direction:     direction,
				EntryPrice:    parseFloat(pos.EntryPrice),
				Quantity:      qty,
				UnrealizedPnL: parseFloat(pos.UnrealizedProfit),
			}, nil
		}
	}
	
	return nil, nil // No position
}

// SetLeverage sets the leverage for a symbol
func (c *RESTClient) SetLeverage(symbol string, leverage int) error {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("leverage", strconv.Itoa(leverage))
	
	_, err := c.signedRequest("POST", "/fapi/v1/leverage", params)
	return err
}

// SetMarginType sets the margin type for a symbol
func (c *RESTClient) SetMarginType(symbol, marginType string) error {
	params := url.Values{}
	params.Set("symbol", symbol)
	params.Set("marginType", marginType)
	
	_, err := c.signedRequest("POST", "/fapi/v1/marginType", params)
	return err
}

// signedRequest makes a signed API request
func (c *RESTClient) signedRequest(method, endpoint string, params url.Values) ([]byte, error) {
	// Add timestamp
	params.Set("timestamp", strconv.FormatInt(time.Now().UnixMilli(), 10))
	
	// Create signature
	signature := c.sign(params.Encode())
	params.Set("signature", signature)
	
	// Build URL
	reqURL := c.baseURL + endpoint
	
	var req *http.Request
	var err error
	
	if method == "GET" || method == "DELETE" {
		reqURL += "?" + params.Encode()
		req, err = http.NewRequest(method, reqURL, nil)
	} else {
		req, err = http.NewRequest(method, reqURL, bytes.NewBufferString(params.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	
	if err != nil {
		return nil, err
	}
	
	// Add headers
	req.Header.Set("X-MBX-APIKEY", c.apiKey)
	
	// Execute request
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	
	// Check for errors
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}
	
	return body, nil
}

// sign creates HMAC SHA256 signature
func (c *RESTClient) sign(data string) string {
	h := hmac.New(sha256.New, []byte(c.apiSecret))
	h.Write([]byte(data))
	return hex.EncodeToString(h.Sum(nil))
}

// parseFloat converts string to float64
func parseFloat(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return f
}
