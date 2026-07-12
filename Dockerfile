FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (none needed beyond what's in slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the FastAPI port
EXPOSE 10000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
