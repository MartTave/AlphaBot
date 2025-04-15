FROM ghcr.io/astral-sh/uv:alpine

WORKDIR /app

RUN apk update && apk upgrade

RUN apk add --no-cache \
    musl-dev \
    clang \
    build-base \
    mesa-gl \
    glib \
    python3-dev \
    libffi-dev

COPY ./*.py ./src/
COPY ./*.toml ./src/
COPY ./*.python-version ./src/

RUN uv sync --project ./src

CMD ["uv", "run","--project", "./src/", "-m", "src"]
