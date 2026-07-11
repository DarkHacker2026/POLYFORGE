FROM ubuntu:22.04

# Avoid tzdata interactive prompt during build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    cmake \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Python pipeline and oracle
COPY pyproject.toml .
COPY README_ORACLE.md .
COPY oracle_standalone.py .
COPY cuda_parser.py .
COPY reference_isa.py .
COPY discovery_agent.py .
COPY vortex_compile.py .
COPY test_llm_comprehension.py .
COPY data/ ./data/
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -e .

# Set the default command to the oracle CLI
ENTRYPOINT ["parallel-oracle"]
