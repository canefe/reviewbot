FROM python:3.12-slim

WORKDIR /app

# base Linux tooling + git
RUN apt-get update && apt-get install -y \
    git \
    bash \
    findutils \
    coreutils \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install --no-cache-dir uv

# copy project
COPY . .

# install deps
RUN uv sync --frozen

EXPOSE 8122

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8122"]