// Azure Container Instances deployment for AURIX Trading Bot
// Estimated cost: ~$15-25/14 days with B1 tier containers

@description('Location for all resources')
param location string = resourceGroup().location

@description('Container registry name')
param acrName string = 'aurixacr${uniqueString(resourceGroup().id)}'

@description('Redis cache name')
param redisName string = 'aurix-redis-${uniqueString(resourceGroup().id)}'

@description('PostgreSQL server name')
param postgresName string = 'aurix-postgres-${uniqueString(resourceGroup().id)}'

@description('PostgreSQL admin password')
@secure()
param postgresPassword string

@description('Binance API Key')
@secure()
param binanceApiKey string

@description('Binance API Secret')
@secure()
param binanceApiSecret string

// Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Azure Cache for Redis (Basic tier for cost savings)
resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: redisName
  location: location
  properties: {
    sku: {
      name: 'Basic'
      family: 'C'
      capacity: 0  // Smallest tier
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
  }
}

// Azure Database for PostgreSQL Flexible Server
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: postgresName
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '15'
    administratorLogin: 'aurix'
    administratorLoginPassword: postgresPassword
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

// PostgreSQL Firewall rule for Azure services
resource postgresFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-03-01-preview' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// PostgreSQL Database
resource postgresDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-03-01-preview' = {
  parent: postgres
  name: 'aurix'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// Container Instance for Go Collector
resource goCollector 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: 'aurix-go-collector'
  location: location
  properties: {
    containers: [
      {
        name: 'go-collector'
        properties: {
          image: '${acr.properties.loginServer}/aurix-go-collector:latest'
          resources: {
            requests: {
              cpu: 1
              memoryInGB: 1
            }
          }
          environmentVariables: [
            {
              name: 'REDIS_URL'
              value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380'
            }
            {
              name: 'BINANCE_API_KEY'
              secureValue: binanceApiKey
            }
            {
              name: 'BINANCE_API_SECRET'
              secureValue: binanceApiSecret
            }
          ]
        }
      }
    ]
    osType: 'Linux'
    restartPolicy: 'Always'
    imageRegistryCredentials: [
      {
        server: acr.properties.loginServer
        username: acr.listCredentials().username
        password: acr.listCredentials().passwords[0].value
      }
    ]
  }
}

// Container Instance for Python Decision Engine
resource pythonEngine 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: 'aurix-python-engine'
  location: location
  properties: {
    containers: [
      {
        name: 'decision-engine'
        properties: {
          image: '${acr.properties.loginServer}/aurix-decision-engine:latest'
          resources: {
            requests: {
              cpu: 2
              memoryInGB: 4
            }
          }
          environmentVariables: [
            {
              name: 'REDIS_URL'
              value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380'
            }
            {
              name: 'DATABASE_URL'
              value: 'postgresql://aurix:${postgresPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/aurix?sslmode=require'
            }
            {
              name: 'BINANCE_API_KEY'
              secureValue: binanceApiKey
            }
            {
              name: 'BINANCE_API_SECRET'
              secureValue: binanceApiSecret
            }
          ]
        }
      }
    ]
    osType: 'Linux'
    restartPolicy: 'Always'
    imageRegistryCredentials: [
      {
        server: acr.properties.loginServer
        username: acr.listCredentials().username
        password: acr.listCredentials().passwords[0].value
      }
    ]
  }
}

// Outputs
output acrLoginServer string = acr.properties.loginServer
output redisHostName string = redis.properties.hostName
output postgresHostName string = postgres.properties.fullyQualifiedDomainName
