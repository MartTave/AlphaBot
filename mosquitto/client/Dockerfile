FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install runtime dependencies including Bluetooth tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    bluez \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml gate_handler.py ./

CMD ["uv", "run", "gate_handler.py"]
