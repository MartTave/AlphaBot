FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install runtime dependencies (no build tools needed for wheels!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./*.py ./src/
COPY ./*.toml ./src/
COPY ./*.python-version ./src/

# Sync dependencies (will use pre-built wheels for OpenCV)
RUN uv sync --project ./src

CMD ["uv", "run", "--project", "./src/", "-m", "src"]
