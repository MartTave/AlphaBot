FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install runtime dependencies (no build tools needed for wheels!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./*.py ./src/
COPY ./*.toml ./src/
COPY ./*.python-version ./src/

# Sync dependencies (will use pre-built wheels for OpenCV)
RUN uv sync --project ./src

CMD ["uv", "run", "--project", "./src/", "-m", "src"]
