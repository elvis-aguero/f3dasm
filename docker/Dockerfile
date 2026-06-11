FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends grep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src/ src/

# Lean by default (torch-free orchestration core; literature reviewer runs on
# pymupdf + BM25). For the heavy quality upgrade (docling layout-aware PDF→MD +
# fastembed dense retrieval), build with:
#   docker build --build-arg EXTRAS=agentic,agentic-extra ...
ARG EXTRAS=agentic
RUN pip install uv --quiet \
    && uv pip install -e ".[${EXTRAS}]" --system --no-cache

WORKDIR /study
ENTRYPOINT ["python", "-m", "f3dasm._src.agentic.run"]
