# AURIX Azure Deployment Guide

## Overview

Deploy AURIX trading bot to Azure for a 14-day trial run using your $100 Azure Student credit.

**Estimated Cost**: ~$20-25 for 14 days

## Prerequisites

1. **Azure CLI** - Install from [docs.microsoft.com](https://docs.microsoft.com/cli/azure/install-azure-cli)
2. **Docker Desktop** - Install from [docker.com](https://www.docker.com/products/docker-desktop)
3. **Azure Student Account** - Login with `az login`

## Quick Start

### Option 1: PowerShell (Windows)

```powershell
cd D:\pribadi\Machine_Learning\Aurix
.\azure\deploy.ps1
```

### Option 2: Bash (WSL/Linux/Mac)

```bash
cd /mnt/d/pribadi/Machine_Learning/Aurix
chmod +x azure/deploy.sh
./azure/deploy.sh
```

## What Gets Deployed

| Resource                    | SKU        | Est. Cost/Day  |
| --------------------------- | ---------- | -------------- |
| Azure Container Registry    | Basic      | $0.17          |
| Azure Cache for Redis       | Basic C0   | $0.55          |
| PostgreSQL Flexible         | B1ms       | $0.50          |
| Container Instance (Go)     | 1 CPU, 1GB | $0.04          |
| Container Instance (Python) | 2 CPU, 4GB | $0.12          |
| **Total**                   |            | **~$1.40/day** |

## Monitor Your Bot

```powershell
# View Go Collector logs
az container logs -g aurix-trading-bot-rg -n aurix-go-collector --follow

# View Decision Engine logs
az container logs -g aurix-trading-bot-rg -n aurix-python-engine --follow
```

## Stop to Save Money

```powershell
az container stop -g aurix-trading-bot-rg -n aurix-go-collector
az container stop -g aurix-trading-bot-rg -n aurix-python-engine
```

## Resume

```powershell
az container start -g aurix-trading-bot-rg -n aurix-go-collector
az container start -g aurix-trading-bot-rg -n aurix-python-engine
```

## Cleanup (Delete Everything)

```powershell
az group delete -n aurix-trading-bot-rg --yes
```

## Configuration

Edit `config/config.yaml` before deployment:

- Set your Binance API keys (testnet first!)
- Adjust risk parameters
- Set trading pairs

## Safety Notes

> ⚠️ **ALWAYS TEST ON TESTNET FIRST**

1. Set `exchange.testnet: true` in config
2. Use Binance Testnet API keys
3. Monitor for at least 3-5 days before real trading
4. Start with minimal position sizes
