FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libclang-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# NOTE: Hardware execution (Stage 6) requires WSL2 on Windows host.
# This image runs Stages 1-5: Oracle + Lowering demo.
ENTRYPOINT ["python", "demo_offline.py"]
