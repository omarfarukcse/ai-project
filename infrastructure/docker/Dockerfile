# infrastructure/docker/Dockerfile
# Multi-stage build for production

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/feature_store models outputs logs

# Create non-root user
RUN useradd -m -u 1000 cdss && chown -R cdss:cdss /app
USER cdss

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
ENTRYPOINT ["python", "-m", "src.api.app"]

# Default command (can be overridden)
CMD ["--host", "0.0.0.0", "--port", "8000"]