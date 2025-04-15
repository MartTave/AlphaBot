FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install runtime dependencies (no build tools needed for wheels!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./*.py ./src/
COPY ./*.toml ./src/
COPY ./*.python-version ./src/

# Sync dependencies (will use pre-built wheels for OpenCV)
RUN uv sync --project ./src
RUN uv sync --project ./src -P camera_sync

CMD ["uv", "run", "--project", "./src/", "-m", "src"]