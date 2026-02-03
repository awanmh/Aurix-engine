#!/bin/bash
# AURIX Azure Deployment Script
# Deploys the trading bot to Azure using Container Instances
#
# Prerequisites:
# - Azure CLI installed and logged in (az login)
# - Docker installed for building images
# - $100 Azure Student credit available

set -e

# Configuration
RESOURCE_GROUP="aurix-trading-bot-rg"
LOCATION="southeastasia"  # Closest to Indonesia
POSTGRES_PASSWORD=$(openssl rand -base64 24)

echo "🚀 AURIX Azure Deployment"
echo "========================="
echo ""

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Please install: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# Check login status
if ! az account show &> /dev/null; then
    echo "📌 Please login to Azure..."
    az login
fi

echo "📦 Current Azure Account:"
az account show --query "{name:name, id:id}" -o table


# Create Resource Group
echo ""
echo "📁 Creating Resource Group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Get Binance API credentials
echo ""
echo "🔑 Enter your Binance API credentials:"
read -p "Binance API Key: " BINANCE_API_KEY
read -sp "Binance API Secret: " BINANCE_API_SECRET
echo ""

# Deploy Azure resources
echo ""
echo "☁️  Deploying Azure resources (this may take 5-10 minutes)..."
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file azure/main.bicep \
    --parameters \
        postgresPassword="$POSTGRES_PASSWORD" \
        binanceApiKey="$BINANCE_API_KEY" \
        binanceApiSecret="$BINANCE_API_SECRET"

# Get ACR credentials
ACR_NAME=$(az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query properties.outputs.acrLoginServer.value -o tsv | sed 's/.azurecr.io//')

echo ""
echo "🐳 Building and pushing Docker images..."

# Login to ACR
az acr login --name $ACR_NAME

ACR_SERVER="$ACR_NAME.azurecr.io"

# Build and push Go collector
echo "Building Go collector..."
docker build -t $ACR_SERVER/aurix-go-collector:latest ./go-services
docker push $ACR_SERVER/aurix-go-collector:latest

# Build and push Python decision engine
echo "Building Python decision engine..."
docker build -t $ACR_SERVER/aurix-decision-engine:latest ./python-services
docker push $ACR_SERVER/aurix-decision-engine:latest

# Restart container instances to pick up new images
echo ""
echo "🔄 Restarting containers..."
az container restart --resource-group $RESOURCE_GROUP --name aurix-go-collector || true
az container restart --resource-group $RESOURCE_GROUP --name aurix-python-engine || true

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Resources deployed:"
echo "   - Azure Container Registry: $ACR_SERVER"
echo "   - Redis Cache: $(az redis show -g $RESOURCE_GROUP --name $(az redis list -g $RESOURCE_GROUP --query '[0].name' -o tsv) --query hostName -o tsv 2>/dev/null || echo 'Deploying...')"
echo "   - PostgreSQL: $(az postgres flexible-server show -g $RESOURCE_GROUP -n $(az postgres flexible-server list -g $RESOURCE_GROUP --query '[0].name' -o tsv) --query fullyQualifiedDomainName -o tsv 2>/dev/null || echo 'Deploying...')"
echo ""
echo "💰 Estimated cost: ~$1.5-2/day (~$20-25 for 14 days)"
echo ""
echo "🔍 Monitor logs:"
echo "   az container logs -g $RESOURCE_GROUP -n aurix-go-collector --follow"
echo "   az container logs -g $RESOURCE_GROUP -n aurix-python-engine --follow"
echo ""
echo "🛑 To stop and save money:"
echo "   az container stop -g $RESOURCE_GROUP -n aurix-go-collector"
echo "   az container stop -g $RESOURCE_GROUP -n aurix-python-engine"
echo ""
echo "🗑️  To delete everything when done:"
echo "   az group delete -n $RESOURCE_GROUP --yes"
