# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -o /aurix-collector ./cmd/collector

# Runtime stage
FROM alpine:3.18

WORKDIR /app

# Install certificates for HTTPS
RUN apk --no-cache add ca-certificates tzdata

# Copy binary
COPY --from=builder /aurix-collector /app/aurix-collector

# Create non-root user
RUN adduser -D -g '' aurix
USER aurix

EXPOSE 8080

ENTRYPOINT ["/app/aurix-collector"]
