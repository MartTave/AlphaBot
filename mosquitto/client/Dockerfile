FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app/src

# Install runtime dependencies including Bluetooth tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    bluez \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Change to the src directory where mounted files are located
WORKDIR /app/src

CMD ["uv", "run", "gate_handler.py"]
