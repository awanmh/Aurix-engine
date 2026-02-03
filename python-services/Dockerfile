# Python Decision Engine Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m aurix
USER aurix

# Create data directories
RUN mkdir -p /app/data /app/models

EXPOSE 8081

CMD ["python", "decision_engine.py"]
