# AURIX Azure Deployment Script (PowerShell)
# Deploys the trading bot to Azure using Container Instances
#
# Prerequisites:
# - Azure CLI installed and logged in (az login)
# - Docker installed for building images
# - $100 Azure Student credit available

$ErrorActionPreference = "Stop"

# Configuration
$RESOURCE_GROUP = "aurix-trading-bot-rg"
$LOCATION = "southeastasia"  # Closest to Indonesia

Write-Host "🚀 AURIX Azure Deployment" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Check Azure CLI
try {
    $null = Get-Command az -ErrorAction Stop
} catch {
    Write-Host "❌ Azure CLI not found. Please install: https://docs.microsoft.com/cli/azure/install-azure-cli" -ForegroundColor Red
    exit 1
}

# Check login status
try {
    $null = az account show 2>$null
} catch {
    Write-Host "📌 Please login to Azure..." -ForegroundColor Yellow
    az login
}

Write-Host "📦 Current Azure Account:" -ForegroundColor Green
az account show --query "{name:name, id:id}" -o table

# Create Resource Group
Write-Host ""
Write-Host "📁 Creating Resource Group: $RESOURCE_GROUP" -ForegroundColor Green
az group create --name $RESOURCE_GROUP --location $LOCATION

# Get credentials
Write-Host ""
Write-Host "🔑 Enter your credentials:" -ForegroundColor Yellow
$BINANCE_API_KEY = Read-Host "Binance API Key"
$BINANCE_API_SECRET = Read-Host "Binance API Secret" -AsSecureString
$BINANCE_API_SECRET_PLAIN = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($BINANCE_API_SECRET))

# Generate PostgreSQL password
$POSTGRES_PASSWORD = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object {[char]$_})

# Deploy Azure resources
Write-Host ""
Write-Host "☁️  Deploying Azure resources (this may take 5-10 minutes)..." -ForegroundColor Green
az deployment group create `
    --resource-group $RESOURCE_GROUP `
    --template-file azure/main.bicep `
    --parameters `
        postgresPassword="$POSTGRES_PASSWORD" `
        binanceApiKey="$BINANCE_API_KEY" `
        binanceApiSecret="$BINANCE_API_SECRET_PLAIN"

# Get ACR info
$deployment = az deployment group show `
    --resource-group $RESOURCE_GROUP `
    --name main `
    --query properties.outputs -o json | ConvertFrom-Json

$ACR_SERVER = $deployment.acrLoginServer.value
$ACR_NAME = $ACR_SERVER -replace '.azurecr.io', ''

Write-Host ""
Write-Host "🐳 Building and pushing Docker images..." -ForegroundColor Green

# Login to ACR
az acr login --name $ACR_NAME

# Build and push Go collector
Write-Host "Building Go collector..."
docker build -t "$ACR_SERVER/aurix-go-collector:latest" ./go-services
docker push "$ACR_SERVER/aurix-go-collector:latest"

# Build and push Python decision engine
Write-Host "Building Python decision engine..."
docker build -t "$ACR_SERVER/aurix-decision-engine:latest" ./python-services
docker push "$ACR_SERVER/aurix-decision-engine:latest"

# Restart containers
Write-Host ""
Write-Host "🔄 Restarting containers..." -ForegroundColor Green
az container restart --resource-group $RESOURCE_GROUP --name aurix-go-collector 2>$null
az container restart --resource-group $RESOURCE_GROUP --name aurix-python-engine 2>$null

Write-Host ""
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Resources deployed:" -ForegroundColor Cyan
Write-Host "   - Azure Container Registry: $ACR_SERVER"
Write-Host "   - Redis Cache: $($deployment.redisHostName.value)"
Write-Host "   - PostgreSQL: $($deployment.postgresHostName.value)"
Write-Host ""
Write-Host "💰 Estimated cost: ~`$1.5-2/day (~`$20-25 for 14 days)" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔍 Monitor logs:" -ForegroundColor Cyan
Write-Host "   az container logs -g $RESOURCE_GROUP -n aurix-go-collector --follow"
Write-Host "   az container logs -g $RESOURCE_GROUP -n aurix-python-engine --follow"
Write-Host ""
Write-Host "🛑 To stop and save money:" -ForegroundColor Yellow
Write-Host "   az container stop -g $RESOURCE_GROUP -n aurix-go-collector"
Write-Host "   az container stop -g $RESOURCE_GROUP -n aurix-python-engine"
Write-Host ""
Write-Host "🗑️  To delete everything when done:" -ForegroundColor Red
Write-Host "   az group delete -n $RESOURCE_GROUP --yes"
