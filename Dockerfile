# ── LearnClear — Dockerfile for Google Cloud Run ──────────────────────────────
# Python 3.11 slim keeps the image small and fast to deploy

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (Docker layer caching —
# if requirements.txt hasn't changed, this layer is reused on redeploy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Cloud Run injects PORT as an environment variable (default 8080)
ENV PORT=8080

# Start the bot
CMD ["python", "main.py"]
