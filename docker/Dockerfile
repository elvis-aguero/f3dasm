FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends grep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src/ src/

RUN pip install uv --quiet \
    && uv pip install -e ".[agentic]" --system --no-cache

WORKDIR /study
ENTRYPOINT ["python", "-m", "f3dasm._src.agentic.run"]
